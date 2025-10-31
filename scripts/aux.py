#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter
import random

def expand_samples(dataset, isolates_per_batch, n_metadata, seed):
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

        for i in range(n_antibiotics):
            predictor_indices = np.sort(
                np.random.choice(n_antibiotics, size=i, replace=False)
            )
            selected_antibiotics = [antibiotics[j] for j in predictor_indices]

            predictors = ['<cls>', *metadata, *selected_antibiotics]
            predictor_set = set(predictors)

            response = [abx for abx in antibiotics if abx not in predictor_set]

            all_predictors.append(predictors)
            all_responses.append(response)
        
    return all_predictors, all_responses

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

        return token_x, year, month, age, species, x_resp, x_pos, y_resp, y_pos
    
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
    