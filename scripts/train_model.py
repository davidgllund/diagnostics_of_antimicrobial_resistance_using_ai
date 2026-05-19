#!/usr/bin/env python
import torch 
import os
import time 
import argparse
import yaml
import pickle as pkl
import numpy as np
import pandas as pd
import random
import gc
from sys import argv

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_curve
from aux import ABTextDataset, expand_samples, get_balanced_indices, to_device, append_df_to_tsv, update_ema, print_progress, numeric_to_phenotype 
from model_utils import create_model, load_previous_model, load_previous_state, start_optimizer_loss, save_model
from evaluate_model import predict_ast_results

N_CONTINUOUS_FEATURES = 3 

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train transformer model for AST prediction.')
    parser.add_argument('--config', '-c', required=True, help='Config file name.')
    parser.add_argument('--folder', '-f', required=True, help='Config/output folder.')
    return parser.parse_args()

def _apply_regularization(logits, class_ids, lambda_reg, device):
    if logits.size(0) <= 1:
        return torch.tensor(0.0, device=device)

    n_classes = class_ids.max() + 1
    sum_logits = torch.zeros(n_classes, logits.size(1), device=device)
    sum_logits.index_add_(0, class_ids, logits)
    
    counts = torch.bincount(class_ids, minlength=n_classes).clamp(min=1).unsqueeze(1)
    class_means = sum_logits / counts
    
    means_per_logit = class_means[class_ids]
    reg_loss = torch.mean((logits - means_per_logit) ** 2)
    
    return lambda_reg * reg_loss


def _compute_loss(logits_list, y_pos, y_resp, criterion, train, config, antibiotics, species, device):
    loss_total = 0.0
    n_total = 0
    
    reg_loss_class = torch.tensor(0.0, device=device)
    reg_loss_grp = torch.tensor(0.0, device=device)

    lambda_class = config['train'].get('lambda_class_reg', 0.0)
    lambda_grp = config['train'].get('lambda_grp_reg', 0.0)
    
    sp_grps = config['vocab'].get('sp_grp', {})
    species_to_idx = {sp: i for i, (grp, list_sp) in enumerate(sp_grps.items()) for sp in list_sp}

    for ab_idx, (ab_name, logits_ab) in enumerate(zip(antibiotics, logits_list)):
        mask = y_pos[:, ab_idx] == 1
        if not mask.any():
            continue

        logits = logits_ab[mask]
        targets = y_resp[mask, ab_idx]
        
        loss_total += criterion[ab_name](logits, targets).sum()
        n_total += targets.numel()

        if train:
            if lambda_class > 0:
                class_ids = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                reg_loss_class += _apply_regularization(logits, class_ids, lambda_class, device)

            if lambda_grp > 0:
                current_species = np.array(species)[mask.cpu().numpy()]
                grp_ids = torch.tensor([species_to_idx.get(s, -1) for s in current_species], 
                                     device=device, dtype=torch.long)
                
                valid_mask = grp_ids != -1
                if valid_mask.any():
                    reg_loss_grp += _apply_regularization(logits[valid_mask], 
                                                          grp_ids[valid_mask], 
                                                          lambda_grp, device)

    if n_total == 0:
        return None

    return (loss_total / n_total) + reg_loss_class + reg_loss_grp
    
def compute_classification_thresholds(df, config):
    outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']
    thresholds = {}
    for ab, g in df.groupby(['antibiotic', 'species', 'n_ast_results']):
        n_outputs = outputs_per_antibiotic[ab[0]]
        y_score = g['pR'].values

        if n_outputs == 2:
            y_true = np.array(g['true'].replace({'S':0,'R':1}), dtype=int)

        elif n_outputs == 3:
            y_true = np.array(g['true'].replace({'S':0,'I':0,'R':1}), dtype=int)

        fpr, tpr, thr = roc_curve(y_true, y_score)

        dist = np.sqrt((1 - tpr)**2 + fpr**2)
        best_point = thr[np.argmin(dist)]

        thresholds[ab] = best_point
        
    return thresholds

def define_thresholds(pS, pR, y, n_classes, error_limit, phenotype):
    taus = np.linspace(0,1,1000)
    y_true = numeric_to_phenotype(y, n_classes)

    if phenotype == 'R':
        pred_R = pR[:,None] >= taus
        true_R = y_true == 'R'

        vme = ((true_R[:,None]) & (~pred_R) & (pS[:,None] >= 0.5)).sum(axis=0)
        error_rate = vme / true_R.sum()
    
    elif phenotype == 'S':
        pred_S = pS[:,None] >= taus
        true_S = y_true == 'S'

        me = ((true_S[:,None]) & (~pred_S) & (pR[:,None] >= 0.5)).sum(axis=0)
        error_rate = me / true_S.sum()

    valid = taus[error_rate <= error_limit]

    return valid

def forward_epoch(df, config, amr_prediction_model, criterion, optimizer, scheduler, epoch_idx, device, vocab, pad_id, train=True):
    model_cfg = config['model']
    data_cfg = config['train'] if train else config['test']

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
     
    x, y = expand_samples(df, config, isolates_per_batch, n_metadata, epoch_idx, 6, train)
    idx = get_balanced_indices(x, y, obs_per_epoch)

    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, encode_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=train, num_workers=num_workers, 
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    n_batches = int(np.ceil(min(len(dataset), obs_per_epoch) / batch_size))

    amr_prediction_model.train() if train else amr_prediction_model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    batch_times = []

    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        token_ids, year, month, age, species_batch, _, _, _, _, y_resp, y_pos = to_device(batch, device)
        
        attention_mask = torch.cat([
            torch.ones(token_ids.size(0), N_CONTINUOUS_FEATURES, device=device, dtype=torch.long),
            (token_ids != pad_id).long()
        ], dim=1)

        with torch.set_grad_enabled(train):
            logits_list = amr_prediction_model(token_ids, year, month, age, attention_mask)
            loss = _compute_loss(logits_list, y_pos, y_resp, criterion, train, config, antibiotics, species_batch, device)

            if loss is None:
                continue

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(amr_prediction_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()


        correct = 0
        count = 0

        for ab_idx, logits_ab in enumerate(logits_list):
            mask = y_pos[:, ab_idx] == 1
            if mask.sum() == 0:
                continue

            logits = logits_ab[mask]
            targets = y_resp[mask, ab_idx]

            softmax = torch.softmax(logits, dim=1)
            preds = torch.argmax(softmax, dim=1)

            correct += (preds == targets).sum().item()
            count += targets.numel()
 
        total_loss += loss.item() * count
        total_correct += correct
        total_samples += count

        batch_times.append(time.time() - start_time)

        if batch_idx + 1 == n_batches:
            break

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    epoch_time = sum(batch_times)
    batch_count = len(batch_times)

    return avg_loss, avg_acc, epoch_time, batch_count

def main():
    arguments = parse_args(argv)
    os.makedirs(arguments.folder, exist_ok=True)
    
    with open(os.path.join(arguments.folder, arguments.config), 'r') as f:
        config = yaml.safe_load(f)

    vocab = torch.load(config['vocab']['vocab'], weights_only=False)
    vocab_size = len(vocab.idx_to_token)
    config['vocab']['vocab_size'] = vocab_size
    pad_id = vocab.token_to_idx['<pad>']

    df = pd.read_csv(config['data']['dataset'], sep='\t')
    df['sentence'] = [x.split(' ') for x in df['sentence']]
    df_train = df.loc[df['fold'].isin(['train'])]
    df_val = df.loc[df['fold'].isin(['validation'])]

    rank = 0
    print('Initializing model...')

    amr_prediction_model = create_model(config, pad_id, rank)

    state = load_previous_model(arguments.folder, amr_prediction_model, rank, load_previous=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_prediction_model.to(device)

    criterion, optimizer = start_optimizer_loss(config, amr_prediction_model, rank)

    n_epochs = config['train']['n_epochs']
    _, _, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss = load_previous_state(
        arguments.folder, n_epochs, rank
    )

    np.random.seed(2025)
    torch.manual_seed(2025)
    random.seed(2025)

    train_perf_path = os.path.join(arguments.folder, 'avg_train_performance.tsv')
    val_perf_path = os.path.join(arguments.folder, 'avg_val_performance.tsv')

    ema_batch_time = None
    obs_train = config['train']['obs_per_epoch']
    obs_val = config['test']['obs_per_epoch']
    bs_train = config['train']['batch_size']
    bs_val = config['test']['batch_size']

    train_batches_per_epoch = int(np.ceil(obs_train / bs_train))
    val_batches_per_epoch = int(np.ceil(obs_val / bs_val))

    batches_per_epoch = train_batches_per_epoch + val_batches_per_epoch
    total_training_batches = n_epochs * batches_per_epoch
    processed_batches_global = start_epoch * batches_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 1000,
        num_training_steps = n_epochs * batches_per_epoch
    )

    for epoch in range(start_epoch, n_epochs):
        step_in_epoch = 0
        epoch_start = time.time()

        train_loss, train_acc, train_time, train_batch_count = forward_epoch(
            df_train, config, amr_prediction_model, criterion, 
            optimizer, scheduler, epoch, device, vocab, pad_id, train=True
        )

        per_batch_time = train_time / train_batch_count
        for _ in range(train_batch_count):
            ema_batch_time = update_ema(ema_batch_time, per_batch_time)
            step_in_epoch += 1
            processed_batches_global += 1

            print_progress(epoch, step_in_epoch, batches_per_epoch,
                        processed_batches_global, total_training_batches,
                        ema_batch_time, epoch_start)
 
        avg_train_results = pd.DataFrame({
            'avg_loss': [train_loss],
            'avg_acc': [train_acc],
            'epoch': [epoch]
        })

        append_df_to_tsv(train_perf_path, avg_train_results)

        gc.collect()
        torch.cuda.empty_cache()

        val_loss, val_acc, _, val_batch_count = forward_epoch(
            df_val, config, amr_prediction_model, criterion, 
            optimizer, None, epoch, device, vocab, pad_id, train=False
        )

        for _ in range(val_batch_count):
            ema_batch_time = update_ema(ema_batch_time, per_batch_time)
            step_in_epoch += 1
            processed_batches_global += 1
            print_progress(epoch, step_in_epoch, batches_per_epoch,
                        processed_batches_global, total_training_batches,
                        ema_batch_time, epoch_start)

        avg_val_results = pd.DataFrame({
            'avg_loss': [val_loss],
            'avg_acc': [val_acc],
            'epoch': [epoch]
        })

        append_df_to_tsv(val_perf_path, avg_val_results)

        del avg_train_results, avg_val_results
        gc.collect()
        torch.cuda.empty_cache()

        print()

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(os.path.join(arguments.folder, 'pretrain_amr_prediction_model.torch'), amr_prediction_model)

        state = {
            'epoch': epoch,
            'avg_batch_time': avg_batch_time,
            'total_elapsed_time': total_time,
            'processed_batches': processed_batches,
            'min_val_loss': min_val_loss
        }

        try:
            with open(os.path.join(arguments.folder, 'pretrain_state.pkl'), 'wb') as f:
                pkl.dump(state, f)
        except Exception as e:
            print(f'Warning: Failed to save training state: {e}')

    print('Training complete!')

    print('Computing classification thresholds...')
    val_results_df = predict_ast_results(df_val, config, amr_prediction_model, n_epochs, device, vocab, pad_id, thresholds=None)
    thresholds = compute_classification_thresholds(val_results_df, config)

    with open(config['model']['thresholds'], 'wb') as f:
            pkl.dump(thresholds, f)

if __name__ == '__main__':
    main()