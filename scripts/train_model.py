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

from sys import argv
from tqdm import tqdm
from aux import ABTextDataset, expand_samples
from model import Encoder, ABPred

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Build a transformer model for prediction of AST results.')
    parser.add_argument('--config', '-c', required=True,
                        help='Configuration file to use for building model.')
    parser.add_argument('--folder', '-f', required=True,
                        help='Folder where config file is located. Output files will be generated here.')

    arguments = parser.parse_args()

    return arguments

def create_transformer_model(config, pad_id, fake_position=None, rank=0):
    if rank == 0:
        print('Creating transformer model...')

    # Unpack model parameters
    model_cfg = config['model']
    vocab_cfg = config['vocab']

    hidden_size = model_cfg['hidden_size']
    max_seq_len = model_cfg['max_seq_len']
    layer_norm_eps = model_cfg['layer_norm_eps']
    dropout_rate_attn = model_cfg['dropout_rate_attn']
    dropout_rate_pwff = model_cfg['dropout_rate_pwff']
    n_heads = model_cfg['n_heads']
    self_attn_size = model_cfg['self_attn_size']
    ffn_dimension = model_cfg['ffn_dimension']
    encoder_stack_depth = model_cfg['encoder_stack_depth']
    vocab_size = vocab_cfg['vocab_size']

    # Build the transformer encoder
    transformer_model = Encoder(encoder_stack_depth, hidden_size, n_heads, max_seq_len, self_attn_size,
                                ffn_dimension, layer_norm_eps, dropout_rate_attn, dropout_rate_pwff,
                                vocab_size, pad_id, fake_position)
    
    # Report parameter count
    n_trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)

    if rank == 0:
        print(f'Transformer model parameters: {n_trainable_params:,}')

    return transformer_model

def create_classification_model(config, rank=0):
    if rank == 0:
        print('Creating classification model...')

    model_cfg = config['model']

    hidden_size = model_cfg['hidden_size']
    layer_norm_eps = model_cfg['layer_norm_eps']
    n_antibiotics = model_cfg['n_antibiotics']
    n_outputs = model_cfg['n_outputs']
    
    classification_model = ABPred(hidden_size, n_antibiotics, n_outputs, layer_norm_eps)
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

    # Load saved transformer + classifier weights
    if os.path.exists(transformer_path):
        log('Loading previous model weights...')
        transformer_model.load_state_dict(torch.load(transformer_path))
        if os.path.exists(classifier_path):
            classification_model.load_state_dict(torch.load(classifier_path))
    
    # Load optimizer and epoch state
    if os.path.exists(state_path):
        log('Loading previous state...')
        with open(state_path, 'rb') as f:
                state = pkl.load(f)
    else:
        log('No previous state found.')

    return state

def load_previous_state(state, model_folder, n_epochs, rank=0):
    if rank == 0:
        print('Checking for previous state...')

    start_epoch = 0
    avg_batch_time, total_time, processed_batches = 0, 0, 0
    min_val_loss = 1000 

    # Allocate metric containers
    train_loss = torch.zeros((n_epochs, 1))
    train_acc = torch.zeros((n_epochs, 1))
    val_loss = torch.zeros((n_epochs, 1))
    val_acc = torch.zeros((n_epochs, 1))

    # If checkpoint exists, resume
    if state is not None and rank == 0:
        print('Found previous state. Resuming...')
        start_epoch = state['epoch'] + 1
        avg_batch_time = state['avg_batch_time']
        total_time = state['total_time']
        processed_batches = state['processed_batches']
        min_val_loss = state['min_val_loss']

        # Restore metric tensors
        train_loss_prev  = torch.load(os.path.join(model_folder, "pretrain_train_loss.pt"))
        train_acc_prev = torch.load(os.path.join(model_folder, "pretrain_train_acc.pt"))
        val_loss_prev = torch.load(os.path.join(model_folder, "pretrain_val_loss.pt"))
        val_acc_prev  = torch.load(os.path.join(model_folder, "pretrain_val_acc.pt"))

        # Copy historical data
        train_loss[:train_loss_prev.shape[0], :train_loss_prev.shape[1]] = train_loss_prev
        train_acc[:train_acc_prev.shape[0], :train_acc_prev.shape[1]] = train_acc_prev
        val_loss[:val_loss_prev.shape[0], :val_loss_prev.shape[1]] = val_loss_prev
        val_acc[:val_acc_prev.shape[0], :val_acc_prev.shape[1]] = val_acc_prev

        if rank == 0:
            print(f'Resuming training from epoch {start_epoch} (min val loss: {min_val_loss:.4f})\n')

    else:
        if rank == 0:
            print('No previous state found.')

    return train_loss, train_acc, val_loss, val_acc, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss

def start_optimizer_loss(config, transformer, classification_model, device, rank=0):
    if rank == 0:
        print('Initializing optimizer and loss functions...')

    model_cfg = config['model']
    train_cfg = config['train']
    
    # Unpack hyperparameters
    batch_size = train_cfg['batch_size']
    learning_rate = train_cfg['learning_rate']
    learning_rate_warm_up = train_cfg['learning_rate_warm_up']

    pos_response = model_cfg['pos_response']
    antibiotics = [key.split('_')[0] for key in pos_response.keys()]

    # One CrossEntropyLoss per antibiotic
    criterion = [nn.CrossEntropyLoss(reduction='none') for _ in antibiotics]
    
    # Optimizer combines encoder and classifier parameters
    optimizer = optim.Adam(params=list(transformer.parameters()) + list(classification_model.parameters()), lr=learning_rate)

    # Learning rate warm-up scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(step / (learning_rate_warm_up / batch_size), 1.0))
    
    return criterion, optimizer, scheduler

def forward_epoch(df, config, transformer_model, classification_model, criterion, optimizer, scheduler, epoch_idx, total_time,
                  processed_batches, device, n_epochs, rank, vocab, fake_position=None, train=True):

    model_cfg = config['model']
    data_cfg = config['train'] if train else config['test']

    # Key dataset and training parameters
    max_seq_len = model_cfg['max_seq_len']
    batch_size = data_cfg['batch_size']
    obs_per_epoch = data_cfg['obs_per_epoch']

    # Metadata for resampling logic
    n_metadata = config['train']['n_metadata']
    min_predictors = config['train']['minimum_size_of_predictors']
    mean_values = config['train']['d_mean_values']
    max_samples = config['train']['d_max_samples']

    pos_response = model_cfg['pos_response']
    antibiotics = [k.split('_')[0] for k in pos_response.keys()]
    n_batches = int(obs_per_epoch / batch_size)
    isolates_per_batch = data_cfg['isolates_to_sample_per_batch']

    # Resample and expand data for this epoch        
    x, y = expand_samples(df, isolates_per_batch, n_metadata, min_predictors, mean_values, max_samples, epoch_idx)
    
    # Random subsampling
    idx = np.random.choice(range(len(x)), size=min(len(x), obs_per_epoch), replace=False)
    x = [x[j] for j in idx]
    y = [y[j] for j in idx]

    # Build dataset and dataloader
    dataset = ABTextDataset(x, y, vocab, antibiotics, max_seq_len, pos_response, fake_position)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=train, num_workers=0)

    # loss + accuracy for each batch
    perform = torch.zeros(2, len(dataloader)) - 1.0

    transformer_model.train() if train else transformer_model.eval()
    classification_model.train() if train else classification_model.eval()

    # Containers for metrics
    all_species, all_targets, all_preds, all_losses = [], [], [], []

    for batch_idx, (ex, ex_pos, x_resp, x_pos, y_resp, y_pos, length_ab, n_y) in enumerate(dataloader):
        # Move tensors to device
        ex, ex_pos = ex.to(device), ex_pos.to(device)
        x_resp, x_pos = x_resp.to(device), x_pos.to(device)
        y_resp, y_pos = y_resp.to(device), y_pos.to(device)
        targets = y_resp[y_pos].to(device)
        valid_lens = None

        # Forward + backward pass
        with torch.set_grad_enabled(train):
            transformer_pred = transformer_model(ex, ex_pos, valid_lens)
            classification_pred = classification_model(transformer_pred[:,0:1,:], y_pos)

            ids = torch.nonzero(y_pos)[:, 1]
            species_meta = [meta[5] for meta in x]

            # Compute weighted loss per antibiotic/species
            loss = None
            ab_weights = config['train']['ab_weights']
            antibiotics = [k.split('_')[0] for k in config['model']['pos_response'].keys()]

            for r, crit in enumerate(criterion):
                mask = (ids == r)
                if mask.sum() > 0:
                    antibiotic = antibiotics[r]
                    base_losses = crit(classification_pred[mask], targets[mask])

                    # Apply species- and class-specific weighting
                    sample_weights = []
                    masked_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    for idx_in_masked, idx_in_full in enumerate(masked_indices):
                        sp = species_meta[idx_in_full]
                        class_idx = int(targets[mask][idx_in_masked].item())
                        species_weights = ab_weights.get(sp, {})
                        antibiotic_weights = species_weights.get(antibiotic, [1.0, 1.0])
                        sample_weight = antibiotic_weights[class_idx] if class_idx < len(antibiotic_weights) else 1.0
                        sample_weights.append(sample_weight)

                    sample_weights = torch.tensor(sample_weights, dtype=torch.float32, device=device)
                    weighted_loss = (base_losses * sample_weights).mean()
                    loss = weighted_loss if loss is None else loss + weighted_loss

                    weighted_losses = base_losses * sample_weights 
                    sample_losses = weighted_losses.detach().cpu().numpy()
                    all_losses.extend(sample_losses)

            # Backpropagation
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

        # Record performance metrics
        perform[0, batch_idx] = loss.item() if loss is not None else 0.0
        acc = (torch.argmax(classification_pred, dim=1) == targets).float().mean().item()
        perform[1, batch_idx] = acc

        batch_preds = torch.argmax(classification_pred, dim=1).detach().cpu().numpy()
        batch_targets = targets.detach().cpu().numpy()
        batch_species = [species_meta[idx.item()] for idx in torch.nonzero(y_pos)[:, 0]]

        all_species.extend(batch_species)
        all_targets.extend(batch_targets)
        all_preds.extend(batch_preds)
        
        if batch_idx + 1 == n_batches:
            break

    # Aggregate results per species
    results_df = pd.DataFrame({
        "species": all_species,
        "true": all_targets,
        "pred": all_preds,
        "loss": all_losses
    })
    species_metrics = (
        results_df.groupby("species", group_keys=False)
        .apply(lambda g: pd.Series({
            "species": g.name,
            "accuracy": np.mean(g["true"] == g["pred"]),
            "avg_loss": np.mean(g["loss"]),
            "epoch": epoch_idx
        }), include_groups=False)
        .reset_index(drop=True)
    )

    # Clean up memory
    del ex, ex_pos, x_resp, x_pos, y_resp, y_pos, targets, transformer_pred, classification_pred, x, y, dataset, dataloader
    torch.cuda.empty_cache()

    if train:
        return (perform, total_time, processed_batches, species_metrics)
    else:
        return perform, species_metrics

def save_model(path, model):
    if isinstance(model,torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, path)

def main():
    arguments = parse_args(argv)
    os.makedirs(arguments.folder, exist_ok=True)
    
    with open(os.path.join(arguments.folder, arguments.config), 'r') as f:
        config = yaml.safe_load(f)

    # Load vocabulary
    vocab = torch.load(config['vocab']['vocab'], weights_only=False)
    vocab_size = len(vocab.idx_to_token)
    config['vocab']['vocab_size'] = vocab_size
    pad_id = vocab.token_to_idx['<pad>']
    fake_position = config['vocab']['pad_position']

    # Load dataset and split
    df = pd.read_csv(config['data']['dataset'], sep='\t')
    df['sentence'] = [x.split(' ') for x in df['sentence']]
    df_train = df.loc[~df['fold'].isin(['val', 'test'])]
    df_val = df.loc[df['fold'].isin(['val'])]
    df_test = df.loc[df['fold'].isin(['test'])]

    rank = 0
    print("Initializing model...")
    transformer_model = create_transformer_model(config, pad_id, fake_position, rank)
    classification_model = create_classification_model(config, rank)

    # Load previous weights if available
    state = load_previous_model(arguments.folder, transformer_model, classification_model, rank, load_previous=True)

    # Move models to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformer_model.to(device)
    classification_model.to(device)

    # Initialize optimizer/scheduler/loss
    criterion, optimizer, scheduler = start_optimizer_loss(config, transformer_model, classification_model, device, rank)

    n_epochs = config['train']['n_epochs']
    train_loss, train_acc, val_loss, val_acc, start_epoch, avg_batch_time, total_time, processed_batches, min_val_loss = load_previous_state(
        state, arguments.folder, n_epochs, rank
    )

    np.random.seed(2025)
    torch.manual_seed(2025)
    random.seed(2025)

    train_species_metrics = pd.DataFrame()
    val_species_metrics = pd.DataFrame()

    for epoch in tqdm(range(n_epochs), desc="Training Epochs", ncols=100):
        # Train phase
        perform_train, total_time, processed_batches, train_sp_metrics = forward_epoch(
            df_train, config, transformer_model, classification_model, criterion, 
            optimizer, scheduler, epoch, total_time, processed_batches, device, 
            n_epochs, rank, vocab, fake_position, train=True
        )

        # Validation phase
        perform_val, val_sp_metrics = forward_epoch(
            df_val, config, transformer_model, classification_model, criterion, 
            optimizer, scheduler, epoch, total_time, processed_batches, device, 
            n_epochs, rank, vocab, fake_position, train=False
        )

        # Compute averages
        avg_train_loss = torch.mean(perform_train[0, :])
        avg_train_acc = torch.mean(perform_train[1, :])
        avg_val_loss = torch.mean(perform_val[0, :])
        avg_val_acc = torch.mean(perform_val[1, :])

        train_loss[epoch] = avg_train_loss
        train_acc[epoch] = avg_train_acc
        val_loss[epoch] = avg_val_loss
        val_acc[epoch] = avg_val_acc

        # Save checkpoint if validation improves
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

        torch.save(train_loss, os.path.join(arguments.folder, 'pretrain_train_loss.pt'))
        torch.save(train_acc, os.path.join(arguments.folder, 'pretrain_train_acc.pt'))
        torch.save(val_loss, os.path.join(arguments.folder, 'pretrain_val_loss.pt'))
        torch.save(val_acc, os.path.join(arguments.folder, 'pretrain_val_acc.pt'))

        train_species_metrics = pd.concat([train_species_metrics, train_sp_metrics], ignore_index=True)
        val_species_metrics = pd.concat([val_species_metrics, val_sp_metrics], ignore_index=True)

    train_species_metrics.to_csv(os.path.join(arguments.folder, 'train_species_metrics.tsv'), sep='\t', index=False)
    val_species_metrics.to_csv(os.path.join(arguments.folder, 'val_species_metrics.tsv'), sep='\t', index=False)

    print("Training complete!")

    # Final testing phase
    if rank == 0:
        print(f"Starting final test evaluation on {len(df_test)} samples...")
    
    perform_test, test_sp_metrics = forward_epoch(
        df_test, config, transformer_model, classification_model, criterion,
        optimizer, scheduler, n_epochs, total_time, processed_batches, device,
        n_epochs, rank, vocab, fake_position, train=False
        )
    
    avg_test_loss = torch.mean(perform_test[0, :])
    avg_test_acc = torch.mean(perform_test[1, :])

    print(f"\nTest Results: Loss={avg_test_loss:.4f}, Accuracy={avg_test_acc:.4f}")

    test_sp_metrics.to_csv(os.path.join(arguments.folder, 'test_species_metrics.tsv'), sep='\t', index=False)
    print("Test results saved to test_species_metrics.tsv")

if __name__ == '__main__':
    main()