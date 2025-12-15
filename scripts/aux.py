#!/usr/bin/env python
import torch
import pandas as pd
import numpy as np
import random
import os
import time

from collections import Counter
from collections import defaultdict
from torch.utils.data import Dataset

def expand_samples(dataset, isolates_per_batch, n_metadata, seed, max_expansions=5, max_predictor_antibiotics=8):
    if seed is None:
        seed = random.randint(1, 10**18)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    sampled = dataset.sample(n=isolates_per_batch, random_state=seed)

    all_predictors = []
    all_responses = []

    for isolate in sampled['sentence']:
        metadata = isolate[:n_metadata]
        antibiotics = isolate[n_metadata:]
        n_antibiotics = len(antibiotics)

        possible_indices = list(range(n_antibiotics))

        if n_antibiotics > max_expansions:
            chosen_indices = np.random.choice(
                possible_indices, size=max_expansions, replace=False
            )
        else:
            chosen_indices = possible_indices

        for i in chosen_indices:
            i_capped = min(i, max_predictor_antibiotics, n_antibiotics)

            predictor_indices = np.sort(
                np.random.choice(n_antibiotics, size=i_capped, replace=False)
            )
            selected_antibiotics = [antibiotics[j] for j in predictor_indices]

            predictors = ['<cls>', *metadata, *selected_antibiotics]
            predictor_set = set(predictors)

            response = [abx for abx in antibiotics if abx not in predictor_set]

            all_predictors.append(predictors)
            all_responses.append(response)

            if i_capped == max_predictor_antibiotics:
                break
        
    return all_predictors, all_responses

def get_balanced_indices(x, y, obs_per_epoch):
    length_buckets = defaultdict(list)
    for i, xi in enumerate(x):
        length_buckets[len(xi)].append(i)

    total_samples = min(len(y), obs_per_epoch)
    unique_lengths = list(length_buckets.keys())
    k = len(unique_lengths)

    base_n = total_samples // k
    remainder = total_samples % k

    selected_indices = []

    for length in unique_lengths:
        bucket = length_buckets[length]

        target = base_n + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        sample_size = min(len(bucket), target)

        chosen = np.random.choice(bucket, size=sample_size, replace=False)
        selected_indices.extend(chosen.tolist())

    if len(selected_indices) < total_samples:
        remaining_pool = list(set(range(len(y))) - set(selected_indices))
        needed = total_samples - len(selected_indices)
        filler = np.random.choice(remaining_pool, size=needed, replace=False)
        selected_indices.extend(filler.tolist())

    return selected_indices

def to_device(batch, device):
    res = []
    for item in batch:
        if torch.is_tensor(item):
            res.append(item.to(device, non_blocking=True))
        else:
            res.append(item)

    return tuple(res)

def append_df_to_tsv(path, df, sep='\t'):
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', sep=sep, header=write_header, index=False)

def update_ema(ema, value, alpha=0.05):
    if ema is None:
        return value
    
    return (1 - alpha) * ema + alpha * value

def print_progress(epoch, step_in_epoch, batches_per_epoch,
                   processed_batches_global, total_training_batches,
                   ema_bt, epoch_start):
    
    global_pct = 100 * processed_batches_global / total_training_batches
    epoch_pct  = 100 * step_in_epoch / batches_per_epoch

    remaining_batches = total_training_batches - processed_batches_global
    remaining_time = remaining_batches * ema_bt
    eta_min = remaining_time / 60

    print(
        f"\rEpoch {epoch+1} [{epoch_pct:5.1f}%] | "
        f"Training {global_pct:5.1f}% | "
        f"ETA (full run): {eta_min:6.2f} min",
        end=""
    )

class ABTextDataset(Dataset):
    """
    Custom Dataset for predictor-response sequences for antibiotic prediction.
    """
    def __init__(self, X, Y, vocab, antibiotics, max_seq_len, pos_response):
        self.vocab = vocab
        self.df = pd.DataFrame({'x': X, 'y': Y})
        self.ab = antibiotics
        self.max_seq_len = max_seq_len
        self.pos_response = pos_response
        self.pos_response_list  = list(pos_response.keys())
        self.species = [x[6] for x in X]
        self.sample_type = [x[7] for x in X]

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns a single sample with all relevant tensors:
        - token indices
        - position indices
        - predictor/response masks
        - sequence lengths
        """
        token_x = self.df.iloc[idx].x
        token_y = self.df.iloc[idx].y

        year = torch.tensor([int(token_x[1])], dtype=torch.float32)
        month = torch.tensor([int(token_x[2])], dtype=torch.float32)
        age = torch.tensor([int(token_x[3])], dtype=torch.float32)

        del token_x[1:4]
    
        p_x = [t.split('_')[0] for t in token_x if t.split('_')[0] in self.ab]
        x_pos = torch.tensor([t in p_x for t in self.ab], dtype=bool)

 
        x_resp = torch.tensor([1 if t in token_x else 0 for t in self.pos_response_list], dtype=torch.int64)

        p_y = [t.split('_')[0] for t in token_y if t.split('_')[0] in self.ab]
        y_pos = torch.tensor([t in p_y for t in self.ab], dtype=bool)

        y_resp = torch.tensor([1 if t in token_y else 0 for t in self.pos_response_list], dtype=torch.int64)

        token_x = token_x + ['<pad>'] * (self.max_seq_len - len(token_x))
        token_x = torch.tensor(self.vocab[token_x], dtype=torch.int64)

        species = self.species[idx]
        sample_type = self.sample_type[idx]

        return token_x, year, month, age, species, sample_type, x_resp, x_pos, y_resp, y_pos
    
class Vocab:
    """
    Maps tokens to integer indices and vice versa.
    """
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.idx_to_token)
        
    def __getitem__(self, tokens):
        """Returns indices for given token(s)."""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        
        return [self.__getitem__(token) for token in tokens]
        
    def to_tokens(self, indices):
        """Converts indices back to tokens."""
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        
        return self.idx_to_token[indices]
        
    @property
    def unk(self):
        """Index of unknown token <unk>."""
        return self.token_to_idx['<unk>']
    