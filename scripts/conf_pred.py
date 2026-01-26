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

    x, y = expand_samples(df, len(df), n_metadata, epoch_idx)

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
    all_n_ast = []

    for batch_idx, batch in enumerate(dataloader):
        token_ids, year, month, age, species_batch, _, x_resp, x_pos, y_resp, y_pos = to_device(batch, device)
 
        attention_mask = torch.cat([
            torch.ones(token_ids.size(0), 3, device=device, dtype=torch.long),
            (token_ids != vocab.token_to_idx['<pad>']).long()
        ], dim=1)

        batch_species = np.concatenate([np.repeat(species_batch[i], int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()
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

        if batch_idx + 1 == n_batches:
            break

    trues = np.concatenate(trues_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    softmaxes = np.concatenate(softmax_list, axis=0)

    results_df = pd.DataFrame({
        'species': all_species,
        'antibiotic': all_antibiotics,
        'n_ast_results': all_n_ast,
        'true': trues,
        'pred': preds,
        'logits': logits[:, 1],
        'softmax': softmaxes[:, 1]
    })
    
    del token_ids, x_resp, x_pos, y_resp, y_pos, targets, amr_pred, classification_pred, x, y, dataset, dataloader
    
    return perform, results_df

def generate_conformal_set(softmax_prob, q_hat):
    nonconf_pos = 1 - softmax_prob
    nonconf_neg = softmax_prob
    
    if nonconf_pos <= q_hat and nonconf_neg > q_hat:
        output = 'R'
    elif nonconf_neg <= q_hat and nonconf_pos > q_hat:
        output = 'S'
    elif nonconf_pos <= q_hat and nonconf_neg <= q_hat:
        output = 'R/S'
    else:
        output = ''
        
    return output

def conf_pred(cal_results, eval_results): 
    cal_results['nonconformity'] = np.where(
        cal_results['true'] == 'R',
        1 - cal_results['softmax'],
        cal_results['softmax']
    )

    confidences = [0.90, 0.95, 0.975] 

    for confidence in confidences:
        alpha = 1 - confidence

        q_hats = {}
        grouped_cal = cal_results.groupby('antibiotic')
        min_group_size = 20
        for abx, g in grouped_cal:
            n = len(g)
            if n >= min_group_size:
                values = np.sort(g['nonconformity'].values)
                k = int(np.ceil((n + 1) * (1 - alpha)))
                k = min(max(k, 1), n)
                q_hat = float(values[k - 1])
                q_hats[abx] = q_hat
            else: 
                print(f"Warning: calibration group {abx} has only {n} samples (<{min_group_size}); using global q_hat")

        vals_all = np.sort(cal_results['nonconformity'].values)
        n_all = len(vals_all)

        if n_all == 0:
            raise ValueError('No calibration samples available to compute global q_hat')
        
        k_all = int(np.ceil((n_all + 1) * (1 - alpha)))
        k_all = min(max(k_all, 1), n_all)
        global_q_hat = float(vals_all[k_all - 1])

        conformal_sets = []
        q_used = []
        for _, row in eval_results.iterrows():
            abx = row['antibiotic']
            q_hat = q_hats.get(abx, global_q_hat)
            conformal_sets.append(generate_conformal_set(row['softmax'], q_hat))
            q_used.append(q_hat)

        eval_results[f'cp_{int(confidence*100)}'] = conformal_sets
        eval_results[f'q_hat_used_{int(confidence*100)}'] = q_used

    return eval_results

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
    df = df.loc[df['fold'].isin(['test'])]
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df_cal = df_shuffled.iloc[:len(df_shuffled)//2]
    df_eval = df_shuffled.iloc[len(df_shuffled)//2:]

    rank = 0
    print('Initializing model...')
    amr_model = create_model(config, pad_id, rank)

    state = load_previous_model('model', amr_model, rank, load_previous=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_model.to(device)

    with open('thresholds.pkl', 'rb') as f:
        thresholds = pkl.load(f)

    _, cal_results_df = evaluate_model(df_cal, config, amr_model, 1000, device, vocab, thresholds=None)
    _, eval_results_df = evaluate_model(df_eval, config, amr_model, 1000, device, vocab, thresholds=thresholds)
    eval_results_df.to_csv(os.path.join(arguments.folder, 'eval_results.tsv'), sep='\t', index=False)

    cal_results_df = cal_results_df.loc[cal_results_df['n_ast_results'] >= 3]
    eval_results_df = eval_results_df.loc[eval_results_df['n_ast_results'] >= 3]

    conf_pred_results = conf_pred(cal_results_df, eval_results_df)
    conf_pred_results.to_csv(os.path.join(arguments.folder, 'conf_pred_results.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    main()