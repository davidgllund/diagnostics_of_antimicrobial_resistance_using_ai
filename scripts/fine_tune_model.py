#!/usr/bin/env python
import argparse
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
from dire.aux import append_df_to_tsv, update_ema, print_progress
from dire.model_utils import create_model, load_previous_training_state, start_optimizer_loss, save_model
from dire.train_model import forward_epoch, compute_classification_thresholds
from dire.evaluate_model import predict_ast_results


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Fine-tune a pretrained AST prediction model on data from additional hospitals/labs.')
    parser.add_argument('--config', '-c', required=True, help='Config file name.')
    parser.add_argument('--folder', '-f', required=True, help='Config/output folder for the fine-tuned model.')
    return parser.parse_args(argv)


def load_base_weights(base_folder, amr_prediction_model, device):
    model_path = os.path.join(base_folder, 'amr_prediction_model.torch')
    amr_prediction_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f'Seeded model weights from base checkpoint: {model_path}')


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

    finetune_cfg = config['finetune']
    labs = finetune_cfg['labs']

    df = pl.read_csv(config['data']['dataset'], separator='\t')
    df = df.with_columns(pl.col('sentence').str.split(' '))
    df = df.filter(pl.col('sentence').list.get(3).is_in(labs))

    df_train = df.filter(pl.col('fold').is_in(['train']))
    df_val = df.filter(pl.col('fold').is_in(['validation']))
    df_cal = df.filter(pl.col('fold').is_in(['calibration']))

    rank = 0
    print('Initializing model...')

    amr_prediction_model = create_model(config, pad_id, rank)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amr_prediction_model.to(device)

    checkpoint_path = os.path.join(arguments.folder, 'amr_prediction_model.torch')
    resuming = os.path.exists(checkpoint_path)

    state = load_previous_training_state(arguments.folder, amr_prediction_model, n_epochs, rank)

    if not resuming:
        load_base_weights(finetune_cfg['base_folder'], amr_prediction_model, device)

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

    obs_train = config['train']['obs_per_epoch']
    obs_val = config['test']['obs_per_epoch']
    bs_train = config['train']['batch_size']
    bs_val = config['test']['batch_size']

    train_batches_per_epoch = int(np.ceil(obs_train / bs_train))
    val_batches_per_epoch = int(np.ceil(obs_val / bs_val))

    batches_per_epoch = train_batches_per_epoch + val_batches_per_epoch
    total_training_batches = n_epochs * batches_per_epoch
    processed_batches_global = start_epoch * batches_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=finetune_cfg.get('num_warmup_steps', 500),
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

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(os.path.join(arguments.folder, 'amr_prediction_model.torch'), amr_prediction_model)

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

    print('Fine-tuning complete!')

    print('Computing classification thresholds...')
    cal_results_df = predict_ast_results(df_cal, config, amr_prediction_model, n_epochs, device, vocab, pad_id, thresholds=None)
    thresholds = compute_classification_thresholds(cal_results_df, config)

    with open(os.path.join(arguments.folder, config['model']['thresholds']), 'wb') as f:
        pkl.dump(thresholds, f)


if __name__ == '__main__':
    main()
