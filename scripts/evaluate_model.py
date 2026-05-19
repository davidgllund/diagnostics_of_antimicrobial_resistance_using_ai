#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from sys import argv
import yaml
import torch
import pickle as pkl 

from model_utils import load_previous_model, create_model
from aux import ABTextDataset, expand_samples, get_balanced_indices, to_device

N_CONTINUOUS_FEATURES = 3 

def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', '-c', required=True, help='Configuration file to use for building model.')
    parser.add_argument('--folder', '-f', required=True, help='Folder where config file is located. Output files will be generated here.')

    arguments = parser.parse_args()

    return arguments

def predict_ast_results(df, config, amr_prediction_model, epoch_idx, device, vocab, pad_id, thresholds=None):
    model_cfg = config['model']
    data_cfg = config['test']

    max_seq_len = model_cfg['max_seq_len']
    batch_size = data_cfg['batch_size']
    obs_per_epoch = data_cfg['obs_per_epoch']
    isolates_per_batch = data_cfg['isolates_to_sample_per_batch']
    n_metadata = config['train']['n_metadata']
    encode_response = model_cfg['encode_response']

    antibiotics = list(dict.fromkeys(k.split('_')[0] for k in encode_response))

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    x, y = expand_samples(df, config, isolates_per_batch, n_metadata, epoch_idx, 6, False)
    idx = get_balanced_indices(x, y, obs_per_epoch)

    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, encode_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, 
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    amr_prediction_model.eval()

    preds_list = []
    trues_list = []

    pS_list = []
    pI_list = []
    pR_list = []

    all_species = []
    all_antibiotics = []
    all_sample_types = []
    all_genders = []
    all_ages = []
    all_labs = []
    all_n_ast = []

    with torch.no_grad():
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

                    if thresholds == None:
                        preds = torch.argmax(softmax, dim=1).cpu().numpy()
                    else:
                        softmax_np = softmax[:, 1].cpu().numpy()
                        threshold_vals = np.array([thresholds.get((antibiotics[ab_idx], sp, n_ast), 0.5) 
                                                for sp, n_ast in zip(batch_species_flat, batch_n_ast_flat)])
                        preds = np.where(softmax_np >= threshold_vals, 1, 0)


                else:
                    pS = softmax[:, 0]
                    pI = softmax[:, 1]
                    pR = softmax[:, 2]

                    if thresholds == None:
                        preds = torch.argmax(softmax, dim=1).cpu().numpy()
                    else:
                        softmax_np = softmax[:, 2].cpu().numpy()
                        preds_max = torch.argmax(softmax, dim=1).cpu().numpy()
                        threshold_vals = np.array([thresholds.get((antibiotics[ab_idx], sp, n_ast), 0.5) 
                                                for sp, n_ast in zip(batch_species_flat, batch_n_ast_flat)])
                        preds = np.where(softmax_np >= threshold_vals, 2, 0)
                        preds = np.where((preds == 0) & (preds_max == 1), 1, preds)

                all_species.extend(batch_species_flat)
                all_sample_types.extend(batch_sample_type_flat)
                all_genders.extend(batch_gender_flat)
                all_ages.extend(batch_age_flat)
                all_labs.extend(batch_lab_flat)
                all_n_ast.extend(batch_n_ast_flat)
                all_antibiotics.extend([antibiotics[ab_idx]] * len(batch_species_flat))

                preds_list.append(preds)
                trues_list.append(targets.cpu().numpy())
                pS_list.append(pS.cpu().numpy())
                pI_list.append(pI.cpu().numpy())
                pR_list.append(pR.cpu().numpy())

    if preds_list:
        preds = np.concatenate(preds_list)
        trues = np.concatenate(trues_list)
        pS = np.concatenate(pS_list)
        pI = np.concatenate(pI_list)
        pR = np.concatenate(pR_list)
    else:
        preds = np.array([], dtype=int)
        trues = np.array([], dtype=int)
        pS = np.array([])
        pI = np.array([])
        pR = np.array([])

    # Collect predictions
    results_df = pd.DataFrame({
        "lab": all_labs,
        "age": all_ages,
        "gender": all_genders,
        "species": all_species,
        "antibiotic": all_antibiotics,
        "sample_type": all_sample_types,
        "n_ast_results": all_n_ast,
        "true": trues,
        "pred": preds,
        "pS": pS,
        "pI": pI,
        "pR": pR
    })

    # Convert from numeric to phenotype
    n_outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']
    for abx, n_cls in n_outputs_per_antibiotic.items():
        mask = results_df['antibiotic'] == abx 

        if n_cls == 2:
            map_dict = {0:'S',1:'R'}
        else:
            map_dict = {0:'S',1:'I',2:'R'}

        results_df.loc[mask,'true'] = results_df.loc[mask,'true'].map(map_dict)
        results_df.loc[mask,'pred'] = results_df.loc[mask,'pred'].map(map_dict)

    return results_df

def conf_pred(cal_results, eval_results, config):  
    min_group_size = 100 
    cal_results = cal_results.copy()
    results_out = eval_results.copy()
    n_outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']
    
    cal_results['nonconf_S'] = 1.0 - cal_results['pS']
    cal_results['nonconf_I'] = 1.0 - cal_results['pI']
    cal_results['nonconf_R'] = 1.0 - cal_results['pR']
    
    alpha = 1 - 0.95
    q_hats = {}

    grouped = cal_results.groupby(['antibiotic', 'species'])
    for (abx, sp), g in grouped:
        if n_outputs_per_antibiotic[abx] == 2:
            labels = ['S', 'R']
        else:
            labels = ['S', 'I', 'R']

        for lbl in labels:
            col = f"nonconf_{lbl}"
            vals = np.sort(g.loc[g['true'] == lbl, col].values)
            n = len(vals)

            if n >= min_group_size:
                k = int(np.ceil((n + 1) * (1 - alpha)))
                k = min(max(k, 1), n)
                q_hats[(abx, sp, lbl)] = float(vals[k - 1])
            else:
                q_hats[(abx, sp, lbl)] = None

    conformal_sets = []
    q_used = []

    for _, row in results_out.iterrows():
        abx = row['antibiotic']
        sp = row['species']
        p = {"S": row["pS"], "I": row["pI"], "R": row["pR"]}

        if n_outputs_per_antibiotic[abx] == 2:
            labels = ['S', 'R']
        else:
            labels = ['S', 'I', 'R']

        nonconf = {k: 1.0 - v for k, v in p.items()}
        include = {}
        qvals = []

        for lbl in labels:
            q = q_hats.get((abx, sp, lbl))

            if q is None:
                include[lbl] = False
            else:
                include[lbl] = nonconf[lbl] <= q

            qvals.append(q)

        valid_labels = [lbl for lbl in labels if q_hats.get((abx, sp, lbl)) is not None]
        pred_set = [lbl for lbl in valid_labels if nonconf[lbl] <= q_hats[(abx, sp, lbl)]]

        if len(pred_set) == 0:
            out = ''
        else:
            out = '/'.join(pred_set)

        conformal_sets.append(out)
        q_used.append(tuple(qvals))

    results_out['conformal_set'] = conformal_sets
    results_out['q_hat'] = q_used
 
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

    print('Predicting AST results...')
    cal_results_df = predict_ast_results(df_cal, config, amr_prediction_model, 100, device, vocab, pad_id, thresholds)
    eval_results_df = predict_ast_results(df_eval, config, amr_prediction_model, 100, device, vocab, pad_id, thresholds)

    print('Performing conformal prediction...')
    df_eval_cp = conf_pred(cal_results_df, eval_results_df, config)
    df_eval_cp.to_csv(os.path.join(arguments.folder, 'eval_results.tsv'), sep='\t', index=False)

    print('Evaluation complete!')

if __name__ == '__main__':
    main()