#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time 
import datetime
import argparse
import yaml
import pickle as pkl
import numpy as np
import pandas as pd
import random
import gc

from sklearn.metrics import f1_score, roc_curve, matthews_corrcoef
from sys import argv
from aux import ABTextDataset, expand_samples, to_device, append_df_to_tsv, update_ema, print_progress
from model import Encoder, ABPred

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Build a transformer model for prediction of AST results.')
    parser.add_argument('--config', '-c', required=True, help='Configuration file to use for building model.')
    parser.add_argument('--folder', '-f', required=True, help='Folder where config file is located. Output files will be generated here.')

    arguments = parser.parse_args()

    return arguments

def create_transformer_model(config, pad_id, rank=0):
    if rank == 0:
        print('Creating transformer model...')

    model_cfg = config['model']
    vocab_cfg = config['vocab']

    transformer_model = Encoder(
        model_cfg['encoder_stack_depth'],
        model_cfg['hidden_size'],
        model_cfg['n_heads'],
        model_cfg['max_seq_len'],
        model_cfg['self_attn_size'],
        model_cfg['ffn_dimension'],
        model_cfg['layer_norm_eps'],
        model_cfg['dropout_rate_attn'],
        model_cfg['dropout_rate_pwff'],
        vocab_cfg['vocab_size'],
        pad_id
    )
    
    n_trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    if rank == 0:
        print(f'Transformer model parameters: {n_trainable_params:,}')

    return transformer_model

def create_classification_model(config, rank=0):
    if rank == 0:
        print('Creating classification model...')

    model_cfg = config['model']
    
    classification_model = ABPred(model_cfg['hidden_size'], 
                                  model_cfg['n_antibiotics'], 
                                  model_cfg['n_outputs'], 
                                  model_cfg['layer_norm_eps']
                                )

    n_trainable_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
    if rank == 0:
        print(f'Classification model paramaters: {n_trainable_params:,}')

    return classification_model

def load_previous_model(model_folder, transformer_model, classification_model, rank=0, load_previous=True):
    def log(msg: str):
        if rank == 0:
            print(msg)

    log('Loading previous model (?)')
    if not load_previous:
        log('No')
        return None   
    log('Yes')

    transformer_path = os.path.join(model_folder, 'pretrain_transformer.torch')
    classifier_path = os.path.join(model_folder, 'pretrain_classification.torch')
    state_path = os.path.join(model_folder, 'pretrain_state.pkl')

    state = None

    if os.path.exists(transformer_path):
        log('Loading previous model weights...')
        transformer_model.load_state_dict(torch.load(transformer_path))
        if os.path.exists(classifier_path):
            classification_model.load_state_dict(torch.load(classifier_path))
    
    if os.path.exists(state_path):
        log('Loading previous state...')
        with open(state_path, 'rb') as f:
                state = pkl.load(f)
    else:
        log('No previous state found.')

    return state

def load_previous_state(model_folder, n_epochs, rank=0):
    if rank == 0:
        print('Checking for previous state...')

    train_results_path = os.path.join(model_folder, 'avg_train_performance.tsv')
    val_results_path = os.path.join(model_folder, 'avg_val_performance.tsv')

    train_loss = torch.zeros((n_epochs, 1))
    val_loss = torch.zeros((n_epochs, 1))

    start_epoch = 0
    min_val_loss = 1000
    avg_batch_time = 0
    total_time = 0
    processed_batches = 0

    if not os.path.exists(train_results_path) or not os.path.exists(val_results_path):
        if rank == 0:
            print('No previous logs found. Starting from scratch.')
        return train_loss, val_loss, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss
    
    df_train_log = pd.read_csv(train_results_path, sep="\t")
    df_val_log   = pd.read_csv(val_results_path, sep="\t")

    if df_train_log.empty or df_val_log.empty:
        if rank == 0:
            print('Log files empty. Starting from scratch.')
        return train_loss, val_loss, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss
    
    last_train_epoch = df_train_log['epoch'].max()
    last_val_epoch = df_val_log['epoch'].max()
    last_epoch = min(last_train_epoch, last_val_epoch)

    for e in range(last_epoch + 1):
        g_train = df_train_log[df_train_log['epoch'] == e]
        g_val   = df_val_log[df_val_log['epoch'] == e]

        if not g_train.empty:
            train_loss[e] = g_train['avg_loss'].mean()

        if not g_val.empty:
            val_loss[e] = g_val['avg_loss'].mean()

    min_val_loss = float(val_loss[: last_epoch + 1].min())

    if rank == 0:
        print(f'Resuming training from epoch {last_epoch + 1} (min val loss: {min_val_loss:.4f})')

    start_epoch = last_epoch + 1
    return train_loss, val_loss, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss

def start_optimizer_loss(config, transformer, classification_model, rank=0):
    if rank == 0:
        print('Initializing optimizer and loss functions...')

    model_cfg = config['model']
    train_cfg = config['train']

    antibiotics = [key.split('_')[0] for key in model_cfg['pos_response'].keys()]

    criterion = [nn.CrossEntropyLoss(reduction='none') for _ in antibiotics]
    optimizer = optim.Adam(params=list(transformer.parameters()) + list(classification_model.parameters()), lr=train_cfg['learning_rate'])
    
    return criterion, optimizer

def _apply_class_regularization(classification_pred, ids, antibiotics, ab_classes, lambda_reg, device):
    abx_to_class = [ab_classes.get(abx, 'unknown') for abx in antibiotics]

    class_to_idxs = {}
    for i, cls in enumerate(abx_to_class):
        class_to_idxs.setdefault(cls, []).append(i)

    class_means = {}
    for cls, idxs in class_to_idxs.items():
        if len(idxs) < 2:
            continue

        mask_cls = torch.isin(ids, torch.tensor(idxs, device=device))
        class_logits = classification_pred[mask_cls]
        if class_logits.size(0) > 0:
            class_means[cls] = class_logits.mean(dim=0)

    reg_loss = torch.tensor(0.0, device=device)
    for cls, idxs in class_to_idxs.items():
        if cls not in class_means:
            continue

        for i in idxs:
            abx_mask = (ids == i)
            if abx_mask.sum() == 0:
                continue

            abx_logits = classification_pred[abx_mask]
            reg_loss += ((abx_logits - class_means[cls]) ** 2).mean()

    reg_loss = lambda_reg * reg_loss
    return reg_loss

def forward_epoch(df, config, transformer_model, classification_model, criterion, optimizer, epoch_idx, device, vocab, train=True, test=False, thresholds=None):

    model_cfg = config['model']
    data_cfg = config['train'] if train else config['test']

    max_seq_len = model_cfg['max_seq_len']
    batch_size = data_cfg['batch_size']
    obs_per_epoch = data_cfg['obs_per_epoch']
    isolates_per_batch = data_cfg['isolates_to_sample_per_batch']
    n_metadata = config['train']['n_metadata']
    pos_response = model_cfg['pos_response']
    antibiotics = [k.split('_')[0] for k in pos_response.keys()]

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
     
    x, y = expand_samples(df, isolates_per_batch, n_metadata, epoch_idx)
    idx = np.random.choice(range(len(x)), size=min(len(x), obs_per_epoch), replace=False)
    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, pos_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=train, num_workers=num_workers, 
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    n_batches = int(np.ceil(min(len(dataset), obs_per_epoch) / batch_size))
    perform = torch.zeros(2, n_batches, dtype=torch.float32) - 1.0

    transformer_model.train() if train else transformer_model.eval()
    classification_model.train() if train else classification_model.eval()

    preds_list = []
    softmax_list = []
    trues_list = []
    losses_list = []
    all_species = []
    all_sample_types = []
    all_antibiotics = []
    all_n_ast = []

    batch_times = []

    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        token_ids, year, month, age, species_batch, sample_type_batch, x_resp, x_pos, y_resp, y_pos = to_device(batch, device)
        valid_lens = None

        ids = torch.nonzero(y_pos, as_tuple=False)[:, 1]
        targets = y_resp[y_pos].to(device)

        batch_species = np.concatenate([np.repeat(species_batch[i], int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()
        batch_sample_type = np.concatenate([np.repeat(sample_type_batch[i], int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()
        batch_n_ast = np.concatenate([np.repeat(int(x_pos[i].sum().cpu().item()), int(y_pos[i].sum())) for i in range(len(y_pos))]).tolist()

        ids_cpu = ids.cpu().numpy()
        batch_antibiotics = [antibiotics[int(i)] for i in ids_cpu]

        with torch.set_grad_enabled(train):
            transformer_pred = transformer_model(token_ids, year, month, age, valid_lens)
            classification_pred = classification_model(transformer_pred[:,3:4,:], y_pos)

            softmax_scores = torch.softmax(classification_pred, dim=1)

            loss = None
            per_sample_losses = torch.zeros((targets.size(0),), dtype=torch.float32, device=device)
            
            for r, crit in enumerate(criterion):
                mask = (ids == r)
                if mask.sum() == 0:
                    continue

                base_losses = crit(classification_pred[mask], targets[mask])
                per_sample_losses[mask] += base_losses.detach()
                
                ab_loss = base_losses.mean()
                loss = ab_loss if loss is None else loss + ab_loss

            if train:
                ab_classes = config['vocab']['ab_class']
                lambda_reg = config['vocab']['lambda_class_reg']

                if lambda_reg > 0:
                    reg_loss = _apply_class_regularization(classification_pred, ids, antibiotics, ab_classes, lambda_reg, device)
                    loss = loss + reg_loss
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_val = loss.item() if loss is not None else float('nan')
        acc = (torch.argmax(classification_pred, dim=1) == targets).float().mean().item()
        perform[0, batch_idx] = loss_val
        perform[1, batch_idx] = acc

        batch_softmax_np = softmax_scores.detach().cpu().numpy()
        batch_targets_np = targets.detach().cpu().numpy()
        batch_losses_np = per_sample_losses.detach().cpu().numpy()

        if thresholds is None:
            batch_preds_np = torch.argmax(classification_pred, dim=1).detach().cpu().numpy()
        else:
            batch_preds_np = np.array([
                1 if s >= thresholds.get((ab, sp), 0.5) else 0
                for s, ab, sp in zip(batch_softmax_np[:, 1], batch_antibiotics, batch_species)
            ])

        preds_list.append(batch_preds_np)
        softmax_list.append(batch_softmax_np)
        trues_list.append(batch_targets_np)
        losses_list.append(batch_losses_np)
        all_antibiotics.extend(batch_antibiotics)
        all_species.extend(batch_species)
        all_sample_types.extend(batch_sample_type)
        all_n_ast.extend(batch_n_ast)

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        if batch_idx + 1 == n_batches:
            break

    if preds_list:
        preds = np.concatenate(preds_list, axis=0)
        softmaxes = np.concatenate(softmax_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        losses = np.concatenate(losses_list, axis=0)
    else:
        preds = np.array([], dtype=int)
        softmaxes = np.array([]).reshape(0, 2)
        trues = np.array([], dtype=int)
        losses = np.array([])

    results_df = pd.DataFrame({
        'species': all_species,
        'antibiotic': all_antibiotics,
        'sample_type': all_sample_types,
        'true': trues,
        'pred': preds,
        'loss': losses,
        'n_ast_results': all_n_ast,
        'softmax_score': softmaxes[:, 1],
        'epoch': epoch_idx
    })

    del token_ids, x_resp, x_pos, y_resp, y_pos, targets, transformer_pred, classification_pred, x, y, dataset, dataloader
    torch.cuda.empty_cache()
    gc.collect()

    class_metrics_df = compute_final_metrics(results_df)
    class_metrics_df['epoch'] = epoch_idx

    if train is False and test is False:
        thresholds = compute_classification_thresholds(results_df)
        with open('thresholds.pkl', "wb") as f:
            pkl.dump(thresholds, f)

    return (perform, sum(batch_times), len(batch_times), class_metrics_df)

def save_model(path, model):
    if isinstance(model,torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, path)

def compute_classification_thresholds(df):
    thresholds = {}
    for ab, g in df.groupby(['antibiotic', 'species']):
        y_true = g['true'].values
        y_score = g['softmax_score'].values

        if len(np.unique(y_true)) < 2:
            thresholds[ab] = 0.5
            continue

        fpr, tpr, thr = roc_curve(y_true, y_score)

        dist = np.sqrt((1 - tpr)**2 + fpr**2)
        best_point = thr[np.argmin(dist)]

        thresholds[ab] = best_point
    
    return thresholds

def compute_final_metrics(results_df):
    records = []
    for (sp, abx, tpe, n_ast), g in results_df.groupby(['species', 'antibiotic', 'sample_type', 'n_ast_results']):
        y_true = g['true'].astype(int)
        y_pred = g['pred'].astype(int)
        losses = g['loss'].astype(float)

        TP = ((y_true == 1) & (y_pred == 1)).sum()
        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()

        records.append({
            'species': sp,
            'antibiotic': abx,
            'sample_type': tpe,
            'n_ast_results': n_ast,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'n_samples': len(g),
            'loss_mean': losses.mean(),
            'loss_median': losses.median(),
            'loss_std': losses.std(),
        })
        
    return pd.DataFrame(records)

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
    df_train = df.loc[~df['fold'].isin(['val', 'test'])]
    df_val = df.loc[df['fold'].isin(['val'])]
    df_test = df.loc[df['fold'].isin(['test'])]

    rank = 0
    print('Initializing model...')
    transformer_model = create_transformer_model(config, pad_id, rank)
    classification_model = create_classification_model(config, rank)

    state = load_previous_model(arguments.folder, transformer_model, classification_model, rank, load_previous=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformer_model.to(device)
    classification_model.to(device)

    criterion, optimizer = start_optimizer_loss(config, transformer_model, classification_model, rank)

    n_epochs = config['train']['n_epochs']
    train_loss, val_loss, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss = load_previous_state(
        arguments.folder, n_epochs, rank
    )

    np.random.seed(2025)
    torch.manual_seed(2025)
    random.seed(2025)

    train_results_path = os.path.join(arguments.folder, 'training_results.tsv')
    train_perf_path = os.path.join(arguments.folder, 'avg_train_performance.tsv')
    val_results_path = os.path.join(arguments.folder, 'validation_results.tsv')
    val_perf_path = os.path.join(arguments.folder, 'avg_val_performance.tsv')

    ema_batch_time = None
    obs_train = config['train']['obs_per_epoch']
    obs_val   = config['test']['obs_per_epoch']
    bs_train  = config['train']['batch_size']
    bs_val    = config['test']['batch_size']

    train_batches_per_epoch = int(np.ceil(obs_train / bs_train))
    val_batches_per_epoch   = int(np.ceil(obs_val   / bs_val))

    batches_per_epoch = train_batches_per_epoch + val_batches_per_epoch
    total_training_batches = n_epochs * batches_per_epoch
    processed_batches_global = start_epoch * batches_per_epoch

    for epoch in range(start_epoch, n_epochs):
        step_in_epoch = 0
        epoch_start = time.time()

        perform_train, train_time, train_batch_count, train_results_df = forward_epoch(
            df_train, config, transformer_model, classification_model, criterion, 
            optimizer, epoch, device, vocab, train=True, test=False, thresholds=None 
        )

        per_batch_time = train_time / train_batch_count
        for _ in range(train_batch_count):
            ema_batch_time = update_ema(ema_batch_time, per_batch_time)
            step_in_epoch += 1
            processed_batches_global += 1
            print_progress(epoch, step_in_epoch, batches_per_epoch,
                        processed_batches_global, total_training_batches,
                        ema_batch_time, epoch_start)

        append_df_to_tsv(train_results_path, train_results_df)

        avg_train_results = pd.DataFrame({
            'avg_loss': perform_train[0, :].mean().unsqueeze(0),
            'std_loss': perform_train[0, :].std().unsqueeze(0),
            'avg_acc': perform_train[1, :].mean().unsqueeze(0),
            'std_acc': perform_train[1, :].std().unsqueeze(0),
            'epoch': torch.tensor([epoch], dtype=torch.int32)
        })

        append_df_to_tsv(train_perf_path, avg_train_results)

        del train_results_df, avg_train_results
        gc.collect()
        torch.cuda.empty_cache()

        perform_val, val_time, val_batch_count, val_results_df = forward_epoch(
            df_val, config, transformer_model, classification_model, criterion, 
            optimizer, epoch, device, vocab, train=False, test=False, thresholds=None 
        )

        for _ in range(val_batch_count):
            ema_batch_time = update_ema(ema_batch_time, per_batch_time)
            step_in_epoch += 1
            processed_batches_global += 1
            print_progress(epoch, step_in_epoch, batches_per_epoch,
                        processed_batches_global, total_training_batches,
                        ema_batch_time, epoch_start)

        append_df_to_tsv(val_results_path, val_results_df)

        avg_val_results = pd.DataFrame({
            'avg_loss': perform_val[0, :].mean().unsqueeze(0),
            'std_loss': perform_val[0, :].std().unsqueeze(0),
            'avg_acc': perform_val[1, :].mean().unsqueeze(0),
            'std_acc': perform_val[1, :].std().unsqueeze(0),
            'epoch': torch.tensor([epoch], dtype=torch.int32)
        })

        append_df_to_tsv(val_perf_path, avg_val_results)

        del val_results_df
        gc.collect()
        torch.cuda.empty_cache()

        print()
        
        avg_val_loss = torch.mean(perform_val[0, :])

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            save_model(os.path.join(arguments.folder, 'pretrain_transformer.torch'), transformer_model)
            save_model(os.path.join(arguments.folder, 'pretrain_classification.torch'), classification_model)

        state = {
            'epoch': epoch,
            'avg_batch_time': avg_batch_time,
            'total_elapsed_time': total_time,
            'processed_batches': processed_batches,
            'min_val_loss': min_val_loss
        }

        with open(os.path.join(arguments.folder, 'pretrain_state.pkl'), 'wb') as f:
            pkl.dump(state, f)

    print('Training complete!')

    if rank == 0:
        print(f'Starting final evaluation...')

    with open(val_results_path, 'rb') as f:
        val_results_df = pd.read_csv(f, sep='\t') 

    with open('thresholds.pkl', 'rb') as f:
        thresholds = pkl.load(f)

    perform_test, _, _, test_results_df = forward_epoch(
        df_test, config, transformer_model, classification_model, criterion,
        optimizer, n_epochs, device, vocab, train=False, test=True, thresholds=thresholds
        )
    
    test_results_df.to_csv(os.path.join(arguments.folder, 'test_results.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    main()
