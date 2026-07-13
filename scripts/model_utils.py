import torch
import torch.nn as nn
import torch.optim as optim
import os 
import pickle as pkl 
import pandas as pd 
from model import AMRPredictionModel 

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
        max_n_tokens=model_cfg['max_n_tokens'],
        vocab_size=vocab_cfg['vocab_size'],
        pad_id=pad_id,
        dropout_rate_attn=model_cfg['dropout_rate_attn'],
        n_outputs_per_antibiotic=model_cfg['n_outputs_per_antibiotic']
    )

    if rank == 0:
        print(f'Model parameters: {sum(p.numel() for p in amr_prediction_model.parameters() if p.requires_grad):,}')

    return amr_prediction_model


def load_previous_training_state(model_folder, amr_prediction_model, n_epochs, rank=0):
    def log(msg):
        if rank == 0:
            print(msg)

    defaults = dict(start_epoch=0, min_val_loss=float('inf'), avg_batch_time=0.0, total_elapsed_time=0.0, processed_batches=0)

    model_path = os.path.join(model_folder, 'amr_prediction_model.torch')
    state_path = os.path.join(model_folder, 'training_state.pkl')
    train_log = os.path.join(model_folder, 'avg_train_performance.tsv')
    val_log = os.path.join(model_folder, 'avg_val_performance.tsv')

    if os.path.exists(model_path):
        try:
            map_loc = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            amr_prediction_model.load_state_dict(
                torch.load(model_path, map_location=map_loc, weights_only=True)
            )
            log('Model weights loaded successfully')
        except Exception as e:
            log(f'Warning: Failed to load model weights: {e}')
    else:
        log('No previous model found, starting from random init')

    pkl_state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, 'rb') as f:
                pkl_state = pkl.load(f)
            log('Training state loaded successfully')
        except Exception as e:
            log(f'Warning: Failed to load training state: {e}')

    tsv_state = {}
    if os.path.exists(train_log) and os.path.exists(val_log):
        try:
            df_train_log = pd.read_csv(train_log, sep='\t')
            df_val_log = pd.read_csv(val_log, sep='\t')

            if not df_train_log.empty and not df_val_log.empty:
                last_epoch = int(min(df_train_log['epoch'].max(), df_val_log['epoch'].max()))
                start_epoch = last_epoch + 1

            if start_epoch >= n_epochs:
                log(f'Training already completed ({start_epoch} epochs logged, n_epochs={n_epochs}).')
                start_epoch = n_epochs

            min_val_loss = float(df_val_log['avg_loss'].min())
            tsv_state = dict(start_epoch=start_epoch, min_val_loss=min_val_loss)
            log(f'Resuming from epoch {start_epoch} (min val loss so far: {min_val_loss:.4f}).')
        except Exception as e:
             log(f'Warning: Failed to read log files: {e}')
    else:
        log('No previous log files found, starting from scratch')

    state = {**defaults, **pkl_state, **tsv_state}
    return state


def start_optimizer_loss(config, amr_prediction_model, rank=0):
    if rank == 0:
        print('Initializing optimizer and loss functions...')

    model_cfg = config['model']
    train_cfg = config['train']
    encode_response = model_cfg['encode_response']

    antibiotics = list(dict.fromkeys(k.split('_')[0] for k in encode_response))

    criterion = {}
    for ab in antibiotics:
        criterion[ab] = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

    optimizer = optim.Adam(params=amr_prediction_model.parameters(), lr=train_cfg['learning_rate'])
    
    return criterion, optimizer


def save_model(path, model):
    try:
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        torch.save(state_dict, path)
    except Exception as e:
        print(f'Error saving model to {path}: {e}')