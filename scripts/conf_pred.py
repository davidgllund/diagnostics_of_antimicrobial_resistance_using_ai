#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from sys import argv
import yaml
import torch
import pickle as pkl

from train_model import load_previous_model, create_model
from aux import ABTextDataset, expand_samples, to_device
from sys import argv

def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', '-c', required=True, help='Configuration file to use for building model.')
    parser.add_argument('--folder', '-f', required=True, help='Folder where config file is located. Output files will be generated here.')

    arguments = parser.parse_args()

    return arguments

def evaluate_model(df, config, amr_model, epoch_idx, device, vocab, thresholds=None):
    model_cfg = config['model']
    data_cfg = config['test']

    max_seq_len = model_cfg['max_seq_len']
    batch_size = data_cfg['batch_size']
    obs_per_epoch = data_cfg['obs_per_epoch']

    n_metadata = config['train']['n_metadata']
    pos_response = model_cfg['pos_response']
    antibiotics = [k.split('_')[0] for k in pos_response.keys()]
    n_batches = int(obs_per_epoch / batch_size)
    isolates_per_batch = data_cfg['isolates_to_sample_per_batch']

    x, y = expand_samples(df, isolates_per_batch, n_metadata, epoch_idx)

    idx = np.random.choice(range(len(x)), size=min(len(x), obs_per_epoch), replace=False)
    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, pos_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    perform = torch.zeros(2, n_batches, dtype=torch.float32) - 1.0

    amr_model.eval()

    preds_list = []
    softmax_list = []
    trues_list = []
    all_species = []
    all_antibiotics = []
    all_n_ast = []

    batch_times = []

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch (matches ABTextDataset.__getitem__)
        token_ids, year, month, age, species_batch, sample_type_batch, x_resp, x_pos, y_resp, y_pos = to_device(batch, device)

        # Build attention mask (3 continuous features prepended)
        attention_mask = torch.cat([
            torch.ones(token_ids.size(0), 3, device=device, dtype=torch.long),
            (token_ids != vocab.token_to_idx['<pad>']).long()
        ], dim=1)

        batch_species = np.concatenate([np.repeat(species_batch[i], int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()
        batch_n_ast = np.concatenate([np.repeat(int(x_pos[i].sum().cpu().item()), int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()

        with torch.set_grad_enabled(False):
            # Run unified model: returns [batch_size, n_antibiotics, n_outputs]
            amr_pred = amr_model(token_ids, year, month, age, y_pos, attention_mask)

            # Select only positive positions (flatten to [n_pos, n_outputs])
            classification_pred = amr_pred[y_pos]

            softmax_scores = torch.softmax(classification_pred, dim=1)

            ids = torch.nonzero(y_pos, as_tuple=False)[:, 1]
            targets = y_resp[y_pos].to(device)

        ids_cpu = ids.cpu().numpy()
        batch_antibiotics = [antibiotics[int(i)] for i in ids_cpu]
        
        batch_softmax_np = softmax_scores.detach().cpu().numpy()
        batch_targets_np = targets.detach().cpu().numpy()

        if thresholds is None:
            batch_preds_np = torch.argmax(classification_pred, dim=1).detach().cpu().numpy()
        else:
            # thresholds may be keyed by (antibiotic, species, n_ast) tuples; fall back to antibiotic-only key or 0.5
            batch_preds = []
            for s, ab, sp, n_ast in zip(batch_softmax_np[:, 1], batch_antibiotics, batch_species, batch_n_ast):
                th = None
                # try tuple key first
                if isinstance(thresholds, dict):
                    th = thresholds.get((ab, sp, n_ast))
                    if th is None:
                        th = thresholds.get(ab)
                if th is None:
                    th = 0.5
                batch_preds.append(1 if s >= th else 0)
            batch_preds_np = np.array(batch_preds)

        preds_list.append(batch_preds_np)
        softmax_list.append(batch_softmax_np)
        trues_list.append(batch_targets_np)
        all_antibiotics.extend(batch_antibiotics)
        all_species.extend(batch_species)
        all_n_ast.extend(batch_n_ast)

        if batch_idx + 1 == n_batches:
            break

    if preds_list:
        preds = np.concatenate(preds_list, axis=0)
        softmaxes = np.concatenate(softmax_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
    else:
        preds = np.array([], dtype=int)
        softmaxes = np.array([]).reshape(0, 2)
        trues = np.array([], dtype=int)

    results_df = pd.DataFrame({
        'species': all_species,
        'antibiotic': all_antibiotics,
        'true': trues,
        'pred': preds,
        'n_ast_results': all_n_ast,
        'softmax_score': softmaxes[:, 1],
        'epoch': epoch_idx
    })

    del token_ids, x_resp, x_pos, y_resp, y_pos, targets, amr_pred, classification_pred, x, y, dataset, dataloader
    
    return perform, results_df

def conformal_set_binary(softmax_prob, q_hat):
    nonconf_pos = 1 - softmax_prob
    nonconf_neg = softmax_prob
    pred_set = []
    if nonconf_pos <= q_hat:
        pred_set.append("Resistant")
    if nonconf_neg <= q_hat:
        pred_set.append("Susceptible")
    return pred_set

def conf_pred(val_results, test_results):
    val_results["nonconformity"] = 1 - val_results["softmax_score"].where(
        val_results["true"] == 1, 1 - val_results["softmax_score"]
    )

    confidences = [0.90, 0.95, 0.975]

    label_map = {0: "Susceptible", 1: "Resistant"}
    test_results["true_label_str"] = test_results["true"].map(label_map)
    combined_group_summaries = []

    for confidence in confidences:
        alpha = 1 - confidence

        q_hats = {}
        grouped_val = val_results.groupby(["species", "antibiotic"])
        for (sp, abx), g in grouped_val:
            if len(g) >= 5:
                q_hat = np.quantile(g["nonconformity"], 1 - alpha)
                q_hats[(sp, abx)] = q_hat
 
        global_q_hat = np.quantile(val_results["nonconformity"], 1 - alpha)

        conformal_sets = []
        q_used = []
        for _, row in test_results.iterrows():
            key = (row["species"], row["antibiotic"])
            q_hat = q_hats.get(key, global_q_hat)
            conformal_sets.append(conformal_set_binary(row["softmax_score"], q_hat))
            q_used.append(q_hat)

        test_results[f"conformal_set_{int(confidence*1000)}"] = conformal_sets
        test_results[f"q_hat_used_{int(confidence*1000)}"] = q_used
        test_results[f"set_size_{int(confidence*1000)}"] = test_results[f"conformal_set_{int(confidence*1000)}"].apply(len)
        test_results[f"covered_{int(confidence*1000)}"] = [
            test_results["true_label_str"].iloc[i] in test_results[f"conformal_set_{int(confidence*1000)}"].iloc[i]
            for i in range(len(test_results))
        ]

        rows = []
        grouped_test = test_results.groupby(["species", "antibiotic"])
        for (sp, abx), g in grouped_test:
            cs_col = f"conformal_set_{int(confidence*1000)}"
            set_size_col = f"set_size_{int(confidence*1000)}"
            covered_col = f"covered_{int(confidence*1000)}"
            q_col = f"q_hat_used_{int(confidence*1000)}"

            # ============================
            # CONFIDENT (set size = 1)
            # ============================
            confident = g[g[set_size_col] == 1]
            n_confident = len(confident)
            confident_correct = confident_VME = confident_ME = 0
            for _, r in confident.iterrows():
                pred_label = r[cs_col][0]
                true_label = r["true"]
                if pred_label == label_map[true_label]:
                    confident_correct += 1
                elif pred_label == "Susceptible" and true_label == 1:
                    confident_VME += 1
                elif pred_label == "Resistant" and true_label == 0:
                    confident_ME += 1

            # ============================
            # UNCERTAIN (set size > 1)
            # ============================
            uncertain = g[g[set_size_col] > 1]
            n_uncertain = len(uncertain)
            uncertain_true_resistant = int((uncertain["true"] == 1).sum())
            uncertain_true_susceptible = int((uncertain["true"] == 0).sum())

            # ============================
            # EMPTY (set size = 0)
            # ============================
            empty = g[g[set_size_col] == 0]
            n_empty = len(empty)
            empty_true_resistant = int((empty["true"] == 1).sum())   # NEW
            empty_true_susceptible = int((empty["true"] == 0).sum()) # NEW

            # ============================
            # Summary stats
            # ============================
            coverage = g[covered_col].mean() if len(g) > 0 else np.nan
            avg_set_size = g[set_size_col].mean() if len(g) > 0 else np.nan
            mean_q_used = g[q_col].mean() if len(g) > 0 else np.nan

            rows.append({
                "species": sp,
                "antibiotic": abx,
                "confidence": confidence,
                "n_samples": len(g),
                "n_confident": n_confident,
                "confident_correct": confident_correct,
                "confident_VME": confident_VME,
                "confident_ME": confident_ME,
                "n_uncertain": n_uncertain,
                "uncertain_true_resistant": uncertain_true_resistant,
                "uncertain_true_susceptible": uncertain_true_susceptible,
                "n_empty": n_empty,
                "empty_true_resistant": empty_true_resistant,
                "empty_true_susceptible": empty_true_susceptible,
                "coverage": coverage,
                "avg_set_size": avg_set_size,
                "mean_q_used": mean_q_used
            })

        group_summary_df = pd.DataFrame(rows)
        combined_group_summaries.append(group_summary_df.assign(confidence=confidence))

    output = pd.concat(combined_group_summaries, ignore_index=True)
    return output

def main():
    arguments = parse_args(argv)
    with open(os.path.join(arguments.folder, arguments.config), 'r') as f:
        config = yaml.safe_load(f)

    vocab = torch.load(config['vocab']['vocab'], weights_only=False)
    vocab_size = len(vocab.idx_to_token)
    config['vocab']['vocab_size'] = vocab_size
    pad_id = vocab.token_to_idx['<pad>']

    df = pd.read_csv(config['data']['dataset'], sep='\t')
    df['sentence'] = [x.split(' ') for x in df['sentence']]
    df_val = df.loc[df['fold'].isin(['val'])]
    df_test = df.loc[df['fold'].isin(['test'])]

    rank = 0
    print("Initializing model...")
    amr_model = create_model(config, pad_id, rank)

    state = load_previous_model('model', amr_model, rank, load_previous=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_model.to(device)

    with open('thresholds.pkl', 'rb') as f:
        thresholds = pkl.load(f)

    perform_val, val_results_df = evaluate_model(df_val, config, amr_model, 1000, device, vocab, thresholds=None)
    perform_test, test_results_df = evaluate_model(df_test, config, amr_model, 1000, device, vocab, thresholds=thresholds)
    test_results_df.to_csv(os.path.join(arguments.folder, "test_results_eval.tsv"), sep="\t", index=False)

    val_results_df = val_results_df.loc[val_results_df['n_ast_results'] >= 3]
    test_results_df = test_results_df.loc[test_results_df['n_ast_results'] >= 3]

    conf_pred_results = conf_pred(val_results_df, test_results_df)
    conf_pred_results.to_csv(os.path.join(arguments.folder, "conf_pred_results.tsv"), sep="\t", index=False)

if __name__ == '__main__':
    main()