#!/usr/bin/env python
import argparse
import numpy as np
import os
import pickle as pkl 
import polars as pl
import torch
import yaml

from sys import argv
from dire.model_utils import load_previous_training_state, create_model
from dire.aux import ABTextDataset, expand_samples_eval, to_device

N_CONTINUOUS_FEATURES = 3 

def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', '-c', required=True, help='Configuration file to use for building model.')
    parser.add_argument('--folder', '-f', required=True, help='Folder where config file is located. Output files will be generated here.')

    arguments = parser.parse_args(argv)

    return arguments


def predict_ast_results(df, config, amr_prediction_model, device, vocab, pad_id, thresholds=None):
    model_cfg = config['model']
    data_cfg = config['test']

    max_n_tokens = model_cfg['max_n_tokens']
    batch_size = data_cfg['batch_size'] 
    n_metadata = config['train']['n_metadata']
    encode_response = model_cfg['encode_response']

    antibiotics = list(dict.fromkeys(k.split('_')[0] for k in encode_response))

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    x, y = expand_samples_eval(df, config, n_metadata, 0, 6)

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_n_tokens, encode_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)

    amr_prediction_model.eval()

    rows = []

    with torch.no_grad():
        for batch in dataloader:
            token_ids, year, month, age, species_batch, sample_type_batch, gender_batch, lab_batch, x_pos, y_resp, y_pos = to_device(batch, device)

            attention_mask = (token_ids != pad_id).long()
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


def conformal_prediction(cal_results, eval_results, config):
    min_group_size = 100
    cal_results = cal_results.clone()
    results_out = eval_results.clone()
    n_outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']

    cal_results = cal_results.with_columns(
        (1.0 - pl.col('pS')).alias('nonconf_S'),
        (1.0 - pl.col('pI')).alias('nonconf_I'),
        (1.0 - pl.col('pR')).alias('nonconf_R'),
    )

    alpha = 1 - 0.99
    q_hats = {}

    for (abx, sp), g in cal_results.group_by(['antibiotic', 'species']):
        if n_outputs_per_antibiotic[abx] == 2:
            labels = ['S', 'R']
        else:
            labels = ['S', 'I', 'R']

        for lbl in labels:
            vals = g.filter(pl.col('true') == lbl).get_column(f'nonconf_{lbl}').sort().to_list()
            n = len(vals)
            if n >= min_group_size:
                k = int(np.ceil((n + 1) * (1 - alpha)))
                k = min(max(k, 1), n)
                q_hats[(abx, sp, lbl)] = float(vals[k - 1])
            else:
                q_hats[(abx, sp, lbl)] = None

    conformal_sets = []
    q_used = []

    for row in results_out.iter_rows(named=True):
        abx = row['antibiotic']
        sp = row['species']
        p = {'S': row['pS'], 'I': row['pI'], 'R': row['pR']}

        if n_outputs_per_antibiotic[abx] == 2:
            labels = ['S', 'R']
        else:
            labels = ['S', 'I', 'R']

        nonconf = {k: 1.0 - v for k, v in p.items()}
        valid_labels = [lbl for lbl in labels if q_hats.get((abx, sp, lbl)) is not None]
        pred_set = [lbl for lbl in valid_labels if nonconf[lbl] <= q_hats[(abx, sp, lbl)]]
        conformal_sets.append('/'.join(pred_set) if pred_set else '')
        q_used.append(tuple(q_hats.get((abx, sp, lbl)) for lbl in labels))

    q_hat_str = [",".join("" if v is None else str(v) for v in tup) for tup in q_used]
    results_out = results_out.with_columns(
        pl.Series('conformal_set', conformal_sets),
        pl.Series('q_hat', q_hat_str),
    )
    return results_out, q_hats


def main():
    arguments = parse_args(argv[1:])
    with open(os.path.join(arguments.folder, arguments.config), 'r') as f:
        config = yaml.safe_load(f)

    vocab = torch.load(config['vocab']['vocab'], weights_only=False)
    vocab_size = len(vocab.idx_to_token)
    config['vocab']['vocab_size'] = vocab_size
    pad_id = vocab.token_to_idx['<pad>']

    df = pl.read_csv(config['data']['dataset'], separator='\t')
    df = df.with_columns(pl.col('sentence').str.split(' '))
    df_cal = df.filter(pl.col('fold').is_in(['calibration']))
    df_eval = df.filter(pl.col('fold').is_in(['test']))

    rank = 0
    print('Initializing model...')
    amr_prediction_model = create_model(config, pad_id, rank)

    n_epochs = config['train']['n_epochs']
    _ = load_previous_training_state(arguments.folder, amr_prediction_model, n_epochs, rank)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_prediction_model.to(device)

    thresholds_path = os.path.join(arguments.folder, config['model']['thresholds'])
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, 'rb') as f:
                thresholds = pkl.load(f)
        except Exception as e:
            print(f'Warning: Failed to load thresholds from {thresholds_path}: {e}')
            thresholds = None

    print('Predicting AST results...')
    cal_results_df = predict_ast_results(df_cal, config, amr_prediction_model, device, vocab, pad_id, thresholds)
    eval_results_df = predict_ast_results(df_eval, config, amr_prediction_model, device, vocab, pad_id, thresholds)

    print('Performing conformal prediction...')
    df_eval_cp, q_hats = conformal_prediction(cal_results_df, eval_results_df, config)
    df_eval_cp.write_csv(os.path.join(arguments.folder, 'eval_results.tsv'), separator='\t', include_header=True)

    # Save q_hats (conformal thresholds) so other scripts can load them
    qh_path = os.path.join(arguments.folder, config['model'].get('q_hats', 'q_hats.pkl'))
    try:
        with open(qh_path, 'wb') as f:
            pkl.dump(q_hats, f)
        print(f'Saved q_hats to {qh_path}')
    except Exception as e:
        print(f'Warning: Failed to save q_hats to {qh_path}: {e}')

    print('Evaluation complete!')

if __name__ == '__main__':
    main()