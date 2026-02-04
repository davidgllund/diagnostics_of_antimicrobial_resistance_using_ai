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

def compute_final_metrics(results_df):
    records = []
    for (sp, abx, tpe, n_ast), g in results_df.groupby(['species', 'antibiotic', 'sample_type', 'n_ast_results']):
        y_true = g['true']
        y_pred = g['pred']

        TP = ((y_true == 'R') & (y_pred == 'R')).sum()
        TN = ((y_true == 'S') & (y_pred == 'S')).sum()
        FP = ((y_true == 'S') & (y_pred == 'R')).sum()
        FN = ((y_true == 'R') & (y_pred == 'S')).sum()

        records.append({
            'species': sp,
            'antibiotic': abx,
            'sample_type': tpe,
            'n_ast_results': n_ast,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'n_samples': len(g)
        })
        
    return pd.DataFrame(records)

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

    x, y = expand_samples(df, config, len(df), n_metadata, epoch_idx)

    idx = np.random.choice(range(len(x)), size=min(len(x), obs_per_epoch), replace=False)
    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, pos_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    perform = torch.zeros(2, n_batches, dtype=torch.float32) - 1.0

    amr_model.eval()

    preds_list = []
    logits_list = []
    softmax_list = []
    trues_list = []
    all_species = []
    all_antibiotics = []
    all_sample_types = []
    all_n_ast = []

    for batch_idx, batch in enumerate(dataloader):
        token_ids, year, month, age, species_batch, sample_type_batch, x_resp, x_pos, y_resp, y_pos = to_device(batch, device)
 
        attention_mask = torch.cat([
            torch.ones(token_ids.size(0), 3, device=device, dtype=torch.long),
            (token_ids != vocab.token_to_idx['<pad>']).long()
        ], dim=1)

        batch_species = np.concatenate([np.repeat(species_batch[i], int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()
        batch_sample_type = np.concatenate([np.repeat(sample_type_batch[i], int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()
        batch_n_ast = np.concatenate([np.repeat(int(x_pos[i].sum().cpu().item()), int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()

        with torch.set_grad_enabled(False):
            amr_pred = amr_model(token_ids, year, month, age, y_pos, attention_mask)
            classification_pred = amr_pred[y_pos]
            softmax_scores = torch.softmax(classification_pred, dim=1)

            ids = torch.nonzero(y_pos, as_tuple=False)[:, 1]
            targets = y_resp[y_pos].to(device)

        ids_cpu = ids.cpu().numpy()
        batch_antibiotics = [antibiotics[int(i)] for i in ids_cpu]

        batch_logits_np = classification_pred.detach().cpu().numpy()
        batch_softmax_np = softmax_scores.detach().cpu().numpy()
        batch_targets_np = targets.detach().cpu().numpy()
 
        if thresholds is None:
            batch_preds_num = torch.argmax(classification_pred, dim=1).detach().cpu().numpy()
        else:
            batch_preds_list = []
            for s, ab, sp, n_ast in zip(batch_softmax_np[:, 1], batch_antibiotics, batch_species, batch_n_ast):
                th = None
                if isinstance(thresholds, dict):
                    th = thresholds.get((ab, sp, n_ast))
                    if th is None:
                        th = thresholds.get(ab)
                if th is None:
                    th = 0.5
                batch_preds_list.append(1 if s >= th else 0)
            batch_preds_num = np.array(batch_preds_list)

        batch_preds_np = np.array(['R' if p == 1 else 'S' for p in batch_preds_num])
        batch_targets_str = np.array(['R' if t == 1 else 'S' for t in batch_targets_np])

        logits_list.append(batch_logits_np)
        preds_list.append(batch_preds_np)
        softmax_list.append(batch_softmax_np)
        trues_list.append(batch_targets_str)
        all_antibiotics.extend(batch_antibiotics)
        all_species.extend(batch_species)
        all_n_ast.extend(batch_n_ast)
        all_sample_types.extend(batch_sample_type)

        if batch_idx + 1 == n_batches:
            break

    trues = np.concatenate(trues_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    softmaxes = np.concatenate(softmax_list, axis=0)

    results_df = pd.DataFrame({
        'species': all_species,
        'antibiotic': all_antibiotics,
        'sample_type': all_sample_types,
        'n_ast_results': all_n_ast,
        'true': trues,
        'pred': preds,
        'logits': logits[:, 1],
        'softmax': softmaxes[:, 1]
    })
    
    del token_ids, x_resp, x_pos, y_resp, y_pos, targets, amr_pred, classification_pred, x, y, dataset, dataloader
    
    return perform, results_df

def conf_pred(cal_results, eval_results): 
    confidences = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    min_group_size = 20

    cal_results = cal_results.copy()
    cal_results['nonconf_R'] = 1.0 - cal_results['softmax']
    cal_results['nonconf_S'] = cal_results['softmax']

    results_out = eval_results.copy()

    for confidence in confidences:
        alpha = 1 - confidence

        all_nonconfs = np.sort(np.concatenate([cal_results['nonconf_R'].values, cal_results['nonconf_S'].values]))
        n_all = len(all_nonconfs)
        if n_all == 0:
            raise ValueError('No calibration samples available to compute global q_hat')
        k_all = int(np.ceil((n_all + 1) * (1 - alpha)))
        k_all = min(max(k_all, 1), n_all)
        global_q_hat = float(all_nonconfs[k_all - 1])

        label_q_hats = {}
        for lbl, col in [('R', 'nonconf_R'), ('S', 'nonconf_S')]:
            vals_lbl = np.sort(cal_results.loc[cal_results['true'] == lbl, col].values)
            n_lbl = len(vals_lbl)
            if n_lbl >= min_group_size:
                k_lbl = int(np.ceil((n_lbl + 1) * (1 - alpha)))
                k_lbl = min(max(k_lbl, 1), n_lbl)
                label_q_hats[lbl] = float(vals_lbl[k_lbl - 1])
            else:
                label_q_hats[lbl] = None

        q_hats = {}
        grouped = cal_results.groupby('antibiotic')
        for abx, g in grouped:
            for lbl, col in [('R', 'nonconf_R'), ('S', 'nonconf_S')]:
                vals = np.sort(g.loc[g['true'] == lbl, col].values)
                n = len(vals)
                if n >= min_group_size:
                    k = int(np.ceil((n + 1) * (1 - alpha)))
                    k = min(max(k, 1), n)
                    q_hats[(abx, lbl)] = float(vals[k - 1])
                else:
                    q_hats[(abx, lbl)] = None

        conformal_sets = []
        q_used = []
        for _, row in results_out.iterrows():
            abx = row['antibiotic']
            pR = row['softmax']
            
            nonconf_R = 1.0 - pR
            nonconf_S = pR

            qR = q_hats.get((abx, 'R'))
            if qR is None:
                qR = label_q_hats.get('R') if label_q_hats.get('R') is not None else global_q_hat

            qS = q_hats.get((abx, 'S'))
            if qS is None:
                qS = label_q_hats.get('S') if label_q_hats.get('S') is not None else global_q_hat

            include_R = nonconf_R <= qR
            include_S = nonconf_S <= qS

            if include_R and not include_S:
                out = 'R'
            elif include_S and not include_R:
                out = 'S'
            elif include_R and include_S:
                out = 'R/S'
            else:
                out = ''

            conformal_sets.append(out)
            q_used.append((qR, qS))

        results_out[f'cp_{int(confidence*100)}'] = conformal_sets
        results_out[f'q_hat_used_{int(confidence*100)}'] = q_used

    return results_out

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
    df_cal = df.loc[df['fold'].isin(['calibration'])]
    df_eval = df.loc[df['fold'].isin(['test'])]

    rank = 0
    print('Initializing model...')
    amr_model = create_model(config, pad_id, rank)

    _ = load_previous_model('model', amr_model, rank, load_previous=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_model.to(device)

    with open('thresholds.pkl', 'rb') as f:
        thresholds = pkl.load(f)

    _, cal_results_df = evaluate_model(df_cal, config, amr_model, 1000, device, vocab, thresholds=None)
    _, eval_results_df = evaluate_model(df_eval, config, amr_model, 1000, device, vocab, thresholds=thresholds)

    class_metrics_df = compute_final_metrics(eval_results_df) 
    class_metrics_df.to_csv(os.path.join(arguments.folder, 'eval_results.tsv'), sep='\t', index=False)

    conf_pred_results = conf_pred(cal_results_df, eval_results_df)
    conf_pred_results.to_csv(os.path.join(arguments.folder, 'conf_pred_results.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    main()