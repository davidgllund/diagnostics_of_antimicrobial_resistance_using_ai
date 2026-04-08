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

def save_model(path, model):
    try:
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        torch.save(state_dict, path)
    except Exception as e:
        print(f'Error saving model to {path}: {e}')