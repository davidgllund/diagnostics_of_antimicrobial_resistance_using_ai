#!/usr/bin/env python
import os
import re
import argparse
import numpy as np
import polars as pl
import yaml
import torch
import pickle as pkl

from dire.model_utils import load_previous_model, create_model 
from dire.aux import ABTextDataset, to_device
from sys import argv

N_CONTINUOUS_FEATURES = 3

def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', required=True, help='')
    parser.add_argument('--config', '-c', required=True, help='Configuration file to use for building model.')
    parser.add_argument('--folder', '-f', required=True, help='Folder where config file is located. Output files will be generated here.')
    parser.add_argument('--target', '-t', required=True, help='')
    parser.add_argument('--predictors', '-p', required=True, help='')

    arguments = parser.parse_args(argv)

    return arguments

def predict_phenotypes(df, config, amr_prediction_model, pred_antibiotic, rsp_antibiotic, pad_id, device, vocab, thresholds=None):
    model_cfg = config['model']
    data_cfg = config['test']

    max_seq_len = model_cfg['max_seq_len']
    batch_size = data_cfg['batch_size']
    n_metadata = config['train']['n_metadata']
    encode_response = model_cfg['encode_response']

    antibiotics = list(dict.fromkeys(k.split('_')[0] for k in encode_response))

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    x = []
    y = []

    for isolate in df['sentence']:
        metadata = isolate[:n_metadata]
        predictors = [w for w in isolate[n_metadata:] if any(t in w for t in pred_antibiotic)]
        x.append(['<cls>', *metadata, *predictors])
        y.append([w for w in isolate[n_metadata:] if any(t in w for t in [rsp_antibiotic])])

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, encode_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)

    amr_prediction_model.eval()

    rows = []

    for batch in dataloader:
        token_ids, year, month, age, species_batch, sample_type_batch, gender_batch, lab_batch, x_pos, y_resp, y_pos = to_device(batch, device)

        attention_mask = torch.cat([
            torch.ones(token_ids.size(0), N_CONTINUOUS_FEATURES, device=device, dtype=torch.long),
            (token_ids != pad_id).long()
        ], dim=1)

        logits_list = amr_prediction_model(token_ids, year, month, age, attention_mask)

        for ab_idx, logits_ab in enumerate(logits_list):
            mask = y_pos[:, ab_idx] == 1
            if mask.sum() == 0:
                continue

            logits = logits_ab[mask]
            targets = y_resp[mask, ab_idx]
            softmax = torch.softmax(logits, dim=1)

            sample_indices = mask.nonzero(as_tuple=False).squeeze(1)
            batch_species_flat = [species_batch[i] for i in sample_indices]
            batch_sample_type_flat = [sample_type_batch[i] for i in sample_indices]
            batch_gender_flat = [gender_batch[i] for i in sample_indices]
            batch_age_flat = [age[i].cpu().item() for i in sample_indices]
            batch_lab_flat = [lab_batch[i] for i in sample_indices]
            batch_n_ast_flat = [int(x_pos[i].sum().cpu()) for i in sample_indices]

            n_outputs = config['model']['n_outputs_per_antibiotic'][antibiotics[ab_idx]]

            if n_outputs == 2:
                pS = softmax[:, 0]
                pR = softmax[:, 1]
                pI = torch.zeros_like(pS)
                if thresholds is None:
                    preds = torch.argmax(softmax, dim=1).cpu().numpy()
                else:
                    softmax_np = softmax[:, 1].cpu().numpy()
                    threshold_vals = np.array([
                        thresholds.get((antibiotics[ab_idx], sp, n_ast), 0.5)
                        for sp, n_ast in zip(batch_species_flat, batch_n_ast_flat)
                    ])
                    preds = np.where(softmax_np >= threshold_vals, 1, 0)
            else:
                pS = softmax[:, 0]
                pI = softmax[:, 1]
                pR = softmax[:, 2]
                if thresholds is None:
                    preds = torch.argmax(softmax, dim=1).cpu().numpy()
                else:
                    softmax_np = softmax[:, 2].cpu().numpy()
                    preds_max = torch.argmax(softmax, dim=1).cpu().numpy()
                    threshold_vals = np.array([
                        thresholds.get((antibiotics[ab_idx], sp, n_ast), 0.5)
                        for sp, n_ast in zip(batch_species_flat, batch_n_ast_flat)
                    ])
                    preds = np.where(softmax_np >= threshold_vals, 2, 0)
                    preds = np.where((preds == 0) & (preds_max == 1), 1, preds)

            for i in range(len(batch_species_flat)):
                rows.append({
                    'lab': batch_lab_flat[i],
                    'age': batch_age_flat[i],
                    'gender': batch_gender_flat[i],
                    'species': batch_species_flat[i],
                    'antibiotic': antibiotics[ab_idx],
                    'sample_type': batch_sample_type_flat[i],
                    'n_ast_results': batch_n_ast_flat[i],
                    'true': int(targets[i].item()),
                    'pred': int(preds[i]),
                    'pS': float(pS[i].item()),
                    'pI': float(pI[i].item()),
                    'pR': float(pR[i].item()),
                })

    if not rows:
        return pl.DataFrame({
            'lab': [], 'age': [], 'gender': [], 'species': [], 'antibiotic': [],
            'sample_type': [], 'n_ast_results': [], 'true': [], 'pred': [],
            'pS': [], 'pI': [], 'pR': []
        })

    results_df = pl.DataFrame(rows)
    n_outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']
    for abx, n_cls in n_outputs_per_antibiotic.items():
        mapping = {0: 'S', 1: 'R'} if n_cls == 2 else {0: 'S', 1: 'I', 2: 'R'}
        results_df = results_df.with_columns(
            pl.when(pl.col('antibiotic') == abx)
            .then(pl.col('true').map_elements(lambda v: mapping.get(int(v), v), return_dtype=pl.String))
            .otherwise(pl.col('true')).alias('true'),
            pl.when(pl.col('antibiotic') == abx)
            .then(pl.col('pred').map_elements(lambda v: mapping.get(int(v), v), return_dtype=pl.String))
            .otherwise(pl.col('pred')).alias('pred')
        )
    return results_df


def apply_q_hat_prediction(eval_results_df, q_hats, config):
    n_outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']
    output_rows = []

    for row in eval_results_df.iter_rows(named=True):
        abx = row['antibiotic']
        sp = row['species']
        p = {'S': row['pS'], 'I': row['pI'], 'R': row['pR']}

        if n_outputs_per_antibiotic[abx] == 2:
            labels = ['S', 'R']
        else:
            labels = ['S', 'I', 'R']

        valid_labels = [lbl for lbl in labels if q_hats.get((abx, sp, lbl)) is not None]
        pred_set = []
        for lbl in valid_labels:
            q = q_hats[(abx, sp, lbl)]
            if (1.0 - p[lbl]) <= q:
                pred_set.append(lbl)

        output_rows.append({
            **row,
            'conformal_set': '/'.join(pred_set) if pred_set else '',
            'q_hat': tuple(q_hats.get((abx, sp, lbl)) for lbl in labels),
        })

    return pl.DataFrame(output_rows)


def main():
    arguments = parse_args(argv)
    with open(os.path.join(arguments.folder, arguments.config), 'r') as f:
        config = yaml.safe_load(f)

    vocab = torch.load(config['vocab']['vocab'], weights_only=False)
    vocab_size = len(vocab.idx_to_token)
    config['vocab']['vocab_size'] = vocab_size
    pad_id = vocab.token_to_idx['<pad>']

    df = pl.read_csv(arguments.input, separator='\t')
    df = df.with_columns(pl.col('sentence').str.split(' '))

    with open(arguments.predictors, 'r') as f:
        pred_antibiotics = [x.strip() for x in f.readlines()]

    rank = 0
    print('Initializing model...')
    amr_prediction_model = create_model(config, pad_id, rank)

    _ = load_previous_model(arguments.folder, amr_prediction_model, rank, load_previous=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_prediction_model.to(device)

    thresholds_path = config['model']['thresholds']
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, 'rb') as f:
                thresholds = pkl.load(f)
        except Exception as e:
            print(f'Warning: Failed to load thresholds from {thresholds_path}: {e}')
            thresholds = None

    eval_results_df = predict_phenotypes(df, config, amr_prediction_model, pred_antibiotics, arguments.target, pad_id, device, vocab, thresholds=thresholds)

    with open(config['model']['q_hats'], 'rb') as f:
        q_hats = pkl.load(f)

    df_eval_cp = apply_q_hat_prediction(eval_results_df, q_hats, config)
    df_eval_cp.write_csv(os.path.join(arguments.folder, 'predictions.tsv'), separator='\t', include_header=True)

if __name__ == '__main__':
    main()
