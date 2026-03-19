#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time 
import argparse
import yaml
import pickle as pkl
import numpy as np
import pandas as pd
import random
import gc
import sys

from transformers import get_linear_schedule_with_warmup
#from sklearn.metrics import roc_curve
from aux import ABTextDataset, expand_samples, get_balanced_indices, to_device, append_df_to_tsv, update_ema, print_progress, numeric_to_phenotype
from model import AMRPredictionModel

N_CONTINUOUS_FEATURES = 3 

def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer model for AST prediction.')
    parser.add_argument('--config', '-c', required=True, help='Config file name.')
    parser.add_argument('--folder', '-f', required=True, help='Config/output folder.')
    return parser.parse_args()

def create_model(config, pad_id, rank=0):
    model_cfg = config['model']
    vocab_cfg = config['vocab']

    print('Creating AMR prediction model...')
    amr_prediction_model = AMRPredictionModel(
        embedding_size=model_cfg['embedding_size'],
        n_antibiotics=model_cfg['n_antibiotics'],
        antibiotics=list(dict.fromkeys(k.split('_')[0] for k in model_cfg['encode_response'])),
        layer_norm_eps=model_cfg['layer_norm_eps'],
        n_heads=model_cfg['n_heads'],
        encoder_stack_depth=model_cfg['encoder_stack_depth'],
        ffn_dimension=model_cfg['ffn_dimension'],
        max_seq_len=model_cfg['max_seq_len'],
        vocab_size=vocab_cfg['vocab_size'],
        pad_id=pad_id,
        dropout_rate_attn=model_cfg['dropout_rate_attn'],
        n_outputs_per_antibiotic=model_cfg['n_outputs_per_antibiotic']
    )
    print(f'Model parameters: {sum(p.numel() for p in amr_prediction_model.parameters() if p.requires_grad):,}')
    return amr_prediction_model

def load_previous_model(model_folder, amr_prediction_model, rank=0, load_previous=True):
    def log(msg: str):
        if rank == 0:
            print(msg)

    log('Loading previous model (?)')
    if not load_previous:
        log('No')
        return None   
    log('Yes')

    model_path = os.path.join(model_folder, 'pretrain_amr_prediction_model.torch')
    state_path = os.path.join(model_folder, 'pretrain_state.pkl')

    state = None

    if os.path.exists(model_path):
        try:
            log('Loading previous model weights...')
            map_loc = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            state_dict = torch.load(model_path, map_location=map_loc)
            amr_prediction_model.load_state_dict(state_dict)
        except Exception as e:
            log(f'Warning: Failed to load model weights: {e}')

    if os.path.exists(state_path):
        try:
            log('Loading previous state...')
            with open(state_path, 'rb') as f:
                state = pkl.load(f)
        except Exception as e:
            log(f'Warning: Failed to load state: {e}')
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

def start_optimizer_loss(config, amr_prediction_model, rank=0):
    if rank == 0:
        print('Initializing optimizer and loss functions...')

    model_cfg = config['model']
    train_cfg = config['train']
    encode_response = model_cfg['encode_response']

    antibiotics = list(dict.fromkeys(k.split('_')[0] for k in encode_response))

    criterion = {}
    for ab in antibiotics:
        criterion[ab] = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.Adam(params=amr_prediction_model.parameters(), lr=train_cfg['learning_rate'])
    
    return criterion, optimizer

def _apply_regularization(logits, antibiotic_indices, words, classes, lambda_reg, device):
    wrd_to_class = [classes.get(wrd, 'unknown') for wrd in words]

    class_to_idxs = {}
    for i, cls in enumerate(wrd_to_class):
        class_to_idxs.setdefault(cls, []).append(i)

    class_means = {}
    for cls, idxs in class_to_idxs.items():
        if len(idxs) < 2:
            continue

        mask_cls = torch.isin(antibiotic_indices, torch.tensor(idxs, device=device))
        class_logits = logits[mask_cls]
        if class_logits.size(0) > 0:
            class_means[cls] = class_logits.mean(dim=0)

    reg_loss = torch.tensor(0.0, device=device)
    for cls, idxs in class_to_idxs.items():
        if cls not in class_means:
            continue

        for i in idxs:
            wrd_mask = (antibiotic_indices == i)
            if wrd_mask.sum() == 0:
                continue

            wrd_logits = logits[wrd_mask]
            reg_loss += ((wrd_logits - class_means[cls]) ** 2).mean()

    reg_loss = lambda_reg * reg_loss
    return reg_loss

def _compute_loss(logits_list, y_pos, y_resp, criterion, train, config, antibiotics, species, device):
    loss_total = torch.tensor(0.0, device=device)
    n_total = 0
    per_sample_losses = []

    lambda_class_reg = config['train'].get('lambda_class_reg', 0.0)
    lambda_grp_reg = config['train'].get('lambda_grp_reg', 0.0)

    ab_classes = config['vocab'].get('ab_class', {})
    sp_grps = config['vocab'].get('sp_grp', {})
    species_to_group = {sp: grp for grp, species_list in sp_grps.items() for sp in species_list}

    for ab_idx, (ab_name, logits_ab) in enumerate(zip(antibiotics, logits_list)):
        mask = y_pos[:, ab_idx] == 1
        if not mask.any():
            continue

        logits = logits_ab[mask]
        targets_ab = y_resp[mask, ab_idx] 
        species_ab = np.array(species)[mask.cpu().numpy()]

        crit = criterion[ab_name]
        sample_losses = crit(logits, targets_ab)

        per_sample_losses.append(sample_losses)

        loss_total += sample_losses.sum()
        n_total += sample_losses.numel()

        if train:
            antibiotic_indices = torch.full(
                (logits.size(0),),
                ab_idx,
                dtype=torch.long,
                device=device
            )

            if lambda_class_reg > 0:
                loss_total += lambda_class_reg * _apply_regularization(logits, antibiotic_indices,
                                                                       [ab_name] * logits.size(0),
                                                                       ab_classes, 1.0, device)

            if lambda_grp_reg > 0:
                loss_total += lambda_grp_reg * _apply_regularization(logits, antibiotic_indices,
                                                                     species_ab, species_to_group, 
                                                                     1.0, device)
                
    if n_total == 0:
        return torch.tensor(0.0, device=device), torch.tensor([], device=device)

    loss = loss_total / n_total
    per_sample_losses = torch.cat(per_sample_losses)

    return loss, per_sample_losses
    
#def compute_classification_thresholds(df):
#    thresholds = {}
#    for ab, g in df.groupby(['antibiotic', 'species', 'n_ast_results']):
#        y_true = g['true'].values
#        y_score = g['softmax_score'].values

#        if len(np.unique(y_true)) < 2:
#            thresholds[ab] = 0.5
#            continue

#        fpr, tpr, thr = roc_curve(y_true, y_score)

#        dist = np.sqrt((1 - tpr)**2 + fpr**2)
#        best_point = thr[np.argmin(dist)]

#        thresholds[ab] = best_point
    
#    return thresholds

#def define_thresholds(pS, pR, y, n_classes, error_limit, phenotype):
#    taus = np.linspace(0,1,1000)
#    y_true = numeric_to_phenotype(y, n_classes)

#    if phenotype == 'R':
#        pred_R = pR[:,None] >= taus
#        true_R = y_true == 'R'

#        vme = ((true_R[:,None]) & (~pred_R) & (pS[:,None] >= 0.5)).sum(axis=0)
#        error_rate = vme / true_R.sum()
    
#    elif phenotype == 'S':
#        pred_S = pS[:,None] >= taus
#        true_S = y_true == 'S'

#        me = ((true_S[:,None]) & (~pred_S) & (pR[:,None] >= 0.5)).sum(axis=0)
#        error_rate = me / true_S.sum()

#    valid = taus[error_rate <= error_limit]

#    return valid

def _compile_output(results_df, config):
    n_outputs_per_antibiotic = config['model']['n_outputs_per_antibiotic']
    records = []

    for (sp, abx, tpe, n_ast), g in results_df.groupby(['species', 'antibiotic', 'sample_type', 'n_ast_results']):
        y_true = g['true'].astype(int).values
        y_pred = g['pred'].astype(int).values

        n_classes = n_outputs_per_antibiotic[abx]
        y_true_phenotype = numeric_to_phenotype(y_true, n_classes)
        y_pred_phenotype = numeric_to_phenotype(y_pred, n_classes)

        pStS = ((y_true_phenotype == 'S') & (y_pred_phenotype == 'S')).sum()
        pItS = ((y_true_phenotype == 'I') & (y_pred_phenotype == 'S')).sum()
        pRtS = ((y_true_phenotype == 'R') & (y_pred_phenotype == 'S')).sum()
        pStI = ((y_true_phenotype == 'S') & (y_pred_phenotype == 'I')).sum()
        pItI = ((y_true_phenotype == 'I') & (y_pred_phenotype == 'I')).sum()
        pRtI = ((y_true_phenotype == 'R') & (y_pred_phenotype == 'I')).sum()
        pStR = ((y_true_phenotype == 'S') & (y_pred_phenotype == 'R')).sum()
        pItR = ((y_true_phenotype == 'I') & (y_pred_phenotype == 'R')).sum()
        pRtR = ((y_true_phenotype == 'R') & (y_pred_phenotype == 'R')).sum()

        records.append({
            'species': sp,
            'antibiotic': abx,
            'sample_type': tpe,
            'n_ast_results': n_ast,
            'pStS': pStS,
            'pItS': pItS,
            'pRtS': pRtS,
            'pStI': pStI,
            'pItI': pItI,
            'pRtI': pRtI,
            'pStR': pStR,
            'pItR': pItR,
            'pRtR': pRtR,
            'n_samples': len(g)
        })
        
    return pd.DataFrame(records)

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
     
    x, y = expand_samples(df, config, isolates_per_batch, n_metadata, epoch_idx)
    idx = get_balanced_indices(x, y, obs_per_epoch)

    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, encode_response)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=train, num_workers=num_workers, 
                                             pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    n_batches = int(np.ceil(min(len(dataset), obs_per_epoch) / batch_size))
    perform = torch.zeros(2, n_batches, dtype=torch.float32) - 1.0

    amr_prediction_model.train() if train else amr_prediction_model.eval()

    preds_list = [] 
    trues_list = []
    losses_list = []

    pS_list = []
    pI_list = []
    pR_list = []
    
    all_species = []
    all_antibiotics = []
    all_sample_types = []
    all_n_ast = []

    batch_times = []

    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        token_ids, year, month, age, species_batch, sample_type_batch, x_resp, x_pos, y_resp, y_pos = to_device(batch, device)
        
        attention_mask = torch.cat([
            torch.ones(token_ids.size(0), N_CONTINUOUS_FEATURES, device=device, dtype=torch.long),
            (token_ids != pad_id).long()
        ], dim=1)

        with torch.set_grad_enabled(train):
            logits_list = amr_prediction_model(token_ids, year, month, age, attention_mask)
            loss, per_sample_losses = _compute_loss(logits_list, y_pos, y_resp, criterion, train, config, antibiotics, species_batch, device)

            if train and loss is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(amr_prediction_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        preds_batch = []
        trues_batch = []

        loss_offset = 0

        for ab_idx, logits_ab in enumerate(logits_list):
            mask = y_pos[:, ab_idx] == 1
            if mask.sum() == 0:
                continue

            logits = logits_ab[mask]
            targets = y_resp[mask, ab_idx]

            softmax = torch.softmax(logits, dim=1)
            preds = torch.argmax(softmax, dim=1)

            n_outputs = config['model']['n_outputs_per_antibiotic'][antibiotics[ab_idx]]

            if n_outputs == 2:
                pS = softmax[:, 0]
                pR = softmax[:, 1]
                pI = torch.zeros_like(pS) 

            else:
                pS = softmax[:, 0]
                pI = softmax[:, 1]
                pR = softmax[:, 2]

            n = logits.size(0)
            n_ast_per_sample = x_pos.sum(dim=1).cpu().numpy()
            sample_indices = mask.nonzero(as_tuple=False).squeeze(1)
            batch_species_flat = [species_batch[i] for i in sample_indices]
            batch_sample_type_flat = [sample_type_batch[i] for i in sample_indices]
            batch_n_ast_flat = [int(n_ast_per_sample[i]) for i in sample_indices]

            all_species.extend(batch_species_flat)
            all_sample_types.extend(batch_sample_type_flat)
            all_n_ast.extend(batch_n_ast_flat)
            all_antibiotics.extend([antibiotics[ab_idx]] * len(batch_species_flat))

            preds_list.append(preds.cpu().numpy())
            trues_list.append(targets.cpu().numpy())
            pS_list.append(pS.detach().cpu().numpy())
            pI_list.append(pI.detach().cpu().numpy())
            pR_list.append(pR.detach().cpu().numpy())

            preds_batch.append(preds)
            trues_batch.append(targets)
  
            losses_all = per_sample_losses[loss_offset:loss_offset + n]
            losses_list.append(losses_all.detach().cpu().numpy())
            loss_offset += n

        preds_all = torch.cat(preds_batch)
        trues_all = torch.cat(trues_batch)

        acc = (preds_all == trues_all).float().mean().item()
        loss_val = loss.item() if loss is not None else float('nan')

        perform[0, batch_idx] = loss_val
        perform[1, batch_idx] = acc

        batch_times.append(time.time() - start_time)

        if batch_idx + 1 == n_batches:
            break

    if preds_list:
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0) 
        losses = np.concatenate(losses_list, axis=0)
        pS = np.concatenate(pS_list)
        pI = np.concatenate(pI_list)
        pR = np.concatenate(pR_list)
    else:
        preds = np.array([], dtype=int)
        trues = np.array([], dtype=int) 
        losses = np.array([])
        pS = np.array([])
        pI = np.array([])
        pR = np.array([])

    results_df = pd.DataFrame({
        'species': all_species,
        'antibiotic': all_antibiotics,
        'sample_type': all_sample_types,
        'true': trues,
        'pred': preds,
        'loss': losses,
        'n_ast_results': all_n_ast,
        "pS": pS,
        "pI": pI,
        "pR": pR,
        'epoch': epoch_idx
    })

    del token_ids, x_resp, x_pos, y_resp, y_pos, targets, x, y, dataset, dataloader
    torch.cuda.empty_cache()
    gc.collect()

    summary_df = _compile_output(results_df, config)
    summary_df['epoch'] = epoch_idx

    return (perform, sum(batch_times), len(batch_times), summary_df)

def save_model(path, model):
    try:
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        torch.save(state_dict, path)
    except Exception as e:
        print(f'Error saving model to {path}: {e}')

def main():
    arguments = parse_args()
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

    train_results_path = os.path.join(arguments.folder, 'training_results.tsv')
    train_perf_path = os.path.join(arguments.folder, 'avg_train_performance.tsv')
    val_results_path = os.path.join(arguments.folder, 'validation_results.tsv')
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

        perform_train, train_time, train_batch_count, train_results_df = forward_epoch(
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

        perform_val, _, val_batch_count, val_results_df = forward_epoch(
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

if __name__ == '__main__':
    main()