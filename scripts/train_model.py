#!/usr/bin/env python
import argparse
import gc
import numpy as np
import os
import pickle as pkl
import polars as pl
import random
import time 
import torch 
import yaml

from sys import argv
from transformers import get_cosine_schedule_with_warmup
from dire.aux_utils import ABTextDataset, expand_samples, to_device, append_df_to_tsv, update_ema, print_progress
from dire.model_utils import create_model, load_previous_training_state, start_optimizer_loss, save_model

N_CONTINUOUS_FEATURES = 3 

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train transformer model for AST prediction.')
    parser.add_argument('--config', '-c', required=True, help='Config file name.')
    parser.add_argument('--folder', '-f', required=True, help='Config/output folder.')
    return parser.parse_args(argv)


def _antibiotic_class_regularization(prob_matrix, mask_matrix, lambda_reg, device):
    counts = mask_matrix.sum(dim=1)
    valid_rows = counts >= 2
    if not valid_rows.any():
        return torch.tensor(0.0, device=device)

    row_mean = (prob_matrix * mask_matrix).sum(dim=1) / counts.clamp(min=1)
    deviations = (prob_matrix - row_mean.unsqueeze(1)) ** 2 * mask_matrix
    per_row_loss = deviations.sum(dim=1) / counts.clamp(min=1)

    return lambda_reg * per_row_loss[valid_rows].mean()


def _extract_sr_probs(logits):
    probs = torch.softmax(logits, dim=1)
    if logits.size(1) == 3:
        sr_probs = probs[:, [0, 2]]
    else:
        sr_probs = probs[:, [0, 1]]
    return sr_probs / sr_probs.sum(dim=1, keepdim=True).clamp(min=1e-7)


def _compute_loss(logits_list, y_pos, y_resp, criterion, train, config, antibiotics, species, device, epoch_idx):
    ab_losses = []
    n_total = 0

    class_members = {} 

    lambda_class = config['train'].get('lambda_class_reg', 0.0)

    ab_to_class_name = config['vocab'].get('ab_class', {})
    unique_classes = sorted(set(ab_to_class_name.values()))
    class_name_to_id = {name: i for i, name in enumerate(unique_classes)}

    for ab_idx, (ab_name, logits_ab) in enumerate(zip(antibiotics, logits_list)):
        mask = y_pos[:, ab_idx] == 1
        if not mask.any():
            continue

        logits = logits_ab[mask]
        targets = y_resp[mask, ab_idx]

        per_sample_loss = criterion[ab_name](logits, targets)
        ab_losses.append(per_sample_loss.mean())
        n_total += targets.numel()

        if not train:
            continue

        if lambda_class > 0:
            class_name = ab_to_class_name.get(ab_name)
            if class_name:
                class_id = class_name_to_id[class_name]
                r_prob_full = _extract_sr_probs(logits_ab)[:, 1]
                class_members.setdefault(class_id, []).append((r_prob_full, mask.float()))

    if not ab_losses:
        return None, None

    ce_loss = torch.stack(ab_losses).mean()

    if not train:
        return ce_loss, ce_loss.item()

    reg_loss_class = torch.tensor(0.0, device=device)
    if lambda_class > 0 and class_members:
        class_losses = []
        for members in class_members.values():
            if len(members) < 2:
                continue
            prob_matrix = torch.stack([m[0] for m in members], dim=1)
            mask_matrix = torch.stack([m[1] for m in members], dim=1)
            class_losses.append(_antibiotic_class_regularization(prob_matrix, mask_matrix, lambda_class, device))
        if class_losses:
            reg_loss_class = torch.stack(class_losses).mean()

    total_loss = ce_loss + reg_loss_class
    return total_loss, ce_loss.item()


def _recover_es_count(val_perf_path, min_delta):
    """Replay the early-stopping patience logic over the logged validation losses so
    that resuming a run doesn't lose track of how many non-improving epochs preceded it
    (and so `es_count` is always defined, even on the first epoch after a resume)."""
    if not os.path.exists(val_perf_path):
        return 0

    df_val_log = pl.read_csv(val_perf_path, separator='\t')
    if df_val_log.is_empty():
        return 0

    best = float('inf')
    es_count = 0
    for loss in df_val_log.sort('epoch')['avg_loss'].to_list():
        if loss < best - min_delta:
            best = loss
            es_count = 0
        else:
            es_count += 1
    return es_count


def forward_epoch(df, config, amr_prediction_model, criterion, optimizer, scheduler, epoch_idx, device, vocab, pad_id, train=True):
    model_cfg = config['model']
    data_cfg = config['train'] if train else config['test']

    max_n_tokens = model_cfg['max_n_tokens']
    batch_size = data_cfg['batch_size']
    n_metadata = config['train']['n_metadata']
    encode_response = model_cfg['encode_response']

    antibiotics = list(dict.fromkeys(k.split('_')[0] for k in encode_response))

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    # Validation always re-samples the same context/target split (fixed seed) so that
    # val_loss changes reflect model improvement, not a different random split each epoch.
    sample_seed = epoch_idx if train else 0
    x, y = expand_samples(df, n_metadata, sample_seed, 6, train)

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_n_tokens, encode_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=train, num_workers=num_workers, 
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)

    amr_prediction_model.train() if train else amr_prediction_model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    batch_times = []

    for _, batch in enumerate(dataloader):
        start_time = time.time()
        token_ids, year, month, age, species_batch, _, _, _, _, y_resp, y_pos = to_device(batch, device)
        
        attention_mask = (token_ids != pad_id).long()

        with torch.set_grad_enabled(train):
            logits_list = amr_prediction_model(token_ids, year, month, age, attention_mask)
            loss, ce_loss_value = _compute_loss(logits_list, y_pos, y_resp, criterion, train, config, antibiotics, species_batch, device, epoch_idx)

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
 
        total_loss += ce_loss_value * count
        total_correct += correct
        total_samples += count

        batch_times.append(time.time() - start_time)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    epoch_time = sum(batch_times)
    batch_count = len(batch_times)

    return avg_loss, avg_acc, epoch_time, batch_count


def main():
    arguments = parse_args(argv[1:])
    os.makedirs(arguments.folder, exist_ok=True)
    
    with open(os.path.join(arguments.folder, arguments.config), 'r') as f:
        config = yaml.safe_load(f)

    vocab = torch.load(config['vocab']['vocab'], weights_only=False)
    vocab_size = len(vocab.idx_to_token)
    config['vocab']['vocab_size'] = vocab_size
    pad_id = vocab.token_to_idx['<pad>']
    n_epochs = config['train']['n_epochs']

    df = pl.read_csv(config['data']['dataset'], separator='\t')
    df = df.with_columns(pl.col('sentence').str.split(' '))
    df_train = df.filter(pl.col('fold').is_in(['train']))
    df_val = df.filter(pl.col('fold').is_in(['validation']))

    rank = 0
    print('Initializing model...')

    amr_prediction_model = create_model(config, pad_id, rank)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_prediction_model.to(device)

    state = load_previous_training_state(arguments.folder, amr_prediction_model, n_epochs, rank)

    start_epoch = state['start_epoch']
    min_val_loss = state['min_val_loss']
    ema_batch_time = state['avg_batch_time'] or None
    processed_batches_global = state['processed_batches']

    criterion, optimizer = start_optimizer_loss(config, amr_prediction_model, rank)

    np.random.seed(2025)
    torch.manual_seed(2025)
    random.seed(2025)

    train_perf_path = os.path.join(arguments.folder, 'avg_train_performance.tsv')
    val_perf_path = os.path.join(arguments.folder, 'avg_val_performance.tsv')

    early_stopping_patience = config['train'].get('early_stopping_patience', 10)
    early_stopping_min_delta = config['train'].get('early_stopping_min_delta', 0.0)
    es_count = _recover_es_count(val_perf_path, early_stopping_min_delta)

    bs_train = config['train']['batch_size']
    bs_val = config['test']['batch_size']
    n_metadata = config['train']['n_metadata']

    x_train_probe, _ = expand_samples(df_train, n_metadata, 0, 6, train=True)
    x_val_probe, _ = expand_samples(df_val, n_metadata, 0, 6, train=False)

    train_batches_per_epoch = int(np.ceil(len(x_train_probe) / bs_train))
    val_batches_per_epoch = int(np.ceil(len(x_val_probe) / bs_val))
    del x_train_probe, x_val_probe

    batches_per_epoch = train_batches_per_epoch + val_batches_per_epoch
    total_training_batches = n_epochs * batches_per_epoch
    processed_batches_global = start_epoch * batches_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=7500,
        num_training_steps=n_epochs * train_batches_per_epoch
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

            print_progress(epoch, step_in_epoch, train_batches_per_epoch,
                           processed_batches_global, (n_epochs * train_batches_per_epoch),
                           ema_batch_time, epoch_start)
 
        avg_train_results = pl.DataFrame({
            'avg_loss': [train_loss],
            'avg_acc': [train_acc],
            'epoch': [epoch]
        })

        append_df_to_tsv(train_perf_path, avg_train_results)

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

        avg_val_results = pl.DataFrame({
            'avg_loss': [val_loss],
            'avg_acc': [val_acc],
            'epoch': [epoch]
        })

        append_df_to_tsv(val_perf_path, avg_val_results)
        print()

        if val_loss < min_val_loss - early_stopping_min_delta:
            min_val_loss = val_loss
            save_model(os.path.join(arguments.folder, 'amr_prediction_model.torch'), amr_prediction_model)
            es_count = 0
        else:
            es_count += 1

        if es_count >= early_stopping_patience:
            print(f'Validation loss has not improved by more than {early_stopping_min_delta} for '
                  f'{early_stopping_patience} epochs. Early stopping triggered.')
            break

        state = {
            'epoch': epoch,
            'avg_batch_time': ema_batch_time,
            'total_elapsed_time': time.time() - epoch_start,
            'processed_batches': processed_batches_global,
            'min_val_loss': min_val_loss
        }

        try:
            with open(os.path.join(arguments.folder, 'training_state.pkl'), 'wb') as f:
                pkl.dump(state, f)
        except Exception as e:
            print(f'Warning: Failed to save training state: {e}')

    print('Training complete!')


if __name__ == '__main__':
    main()