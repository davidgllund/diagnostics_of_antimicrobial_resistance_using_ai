#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(X, valid_lens=None):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    
    batch_pad = torch.where(torch.sum(valid_lens==0,1) > 0)[0]
    pos_pad = torch.sum(valid_lens==0,1)[torch.sum(valid_lens==0,1) > 0]

    batch_pad = torch.repeat_interleave(batch_pad, pos_pad, dim=0)
    pos_pad = torch.where(valid_lens==0, torch.arange(valid_lens.shape[1]), -1).reshape(-1)[torch.where(valid_lens==0, torch.arange(valid_lens.shape[1]), -1).reshape(-1) >= 0]

    X[batch_pad,:,pos_pad,:] = -1e8

    return F.softmax(X, dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, max_seq_len, self_attn_size, dropout_rate_attn, layer_norm_eps) -> None:
        super().__init__()

        assert hidden_size % n_heads == 0, \
            'Model dimension must be divisible by the number of attention heads.'

        self.d_model = hidden_size
        self.num_heads = n_heads
        self.seq_len = max_seq_len
        self.d_k = self_attn_size

        self.W_q = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(self.num_heads * self.d_k, self.d_model, bias=False)

        self.dropout_attn = nn.Dropout(dropout_rate_attn)
        self.dropout_out = nn.Dropout(dropout_rate_attn)
        self.layer_norm = nn.LayerNorm(self.d_model, eps = layer_norm_eps)

    def forward(self, x, valid_lens=None):
        residual = x
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

        scores = torch.matmul(Q / self.d_k ** 0.5, K.transpose(2, 3))
        attn_weights = masked_softmax(scores, valid_lens)
        attn_weights = self.dropout_attn(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)

        output = self.layer_norm(self.dropout_out(output) + residual)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_dimension, dropout_rate_pwff, layer_norm_eps):
        super().__init__()

        self.W_1 = nn.Linear(hidden_size, ffn_dimension)
        self.W_2 = nn.Linear(ffn_dimension, hidden_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps = layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate_pwff)

    def forward(self, x):
        residual = x
        x = self.W_2(self.relu(self.W_1(x)))
        
        return self.layer_norm(self.dropout(x) + residual)
    
class SingleLayerEncoder(nn.Module):
    def __init__(self, hidden_size, n_heads, max_seq_len, self_attn_size, ffn_dimension, layer_norm_eps, dropout_rate_attn, dropout_rate_pwff):
        super().__init__()

        self.attn = MultiHeadAttention(hidden_size, n_heads, max_seq_len, self_attn_size, dropout_rate_attn, layer_norm_eps)
        self.ffn = PositionWiseFeedForward(hidden_size, ffn_dimension, dropout_rate_pwff, layer_norm_eps)

    def forward(self, x, valid_lens=None):
        output = self.attn(x, valid_lens)
        output = self.ffn(output)

        return output

class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_rate, layer_norm_eps, pad_id):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, token_ids):
        token_embeddings = self.token_embeddings(token_ids)
        embeddings = self.layer_norm(self.dropout(token_embeddings))

        return embeddings

class Encoder(nn.Module):
    def __init__(self, encoder_stack_depth, hidden_size, n_heads, max_seq_len, self_attn_size, ffn_dimension, layer_norm_eps, dropout_rate_attn, dropout_rate_pwff, vocab_size, pad_id):
        super().__init__()

        self.pad_idx = pad_id
        self.embedding_layer = Embeddings(vocab_size, hidden_size, dropout_rate_attn, layer_norm_eps, pad_id)
        self.year_proj = nn.Linear(1, hidden_size)
        self.month_proj = nn.Linear(1, hidden_size)
        self.age_proj = nn.Linear(1, hidden_size)
        
        self.blks = nn.Sequential()
        for i in range(encoder_stack_depth):
            self.blks.add_module(f"i{i}", SingleLayerEncoder(hidden_size, n_heads, max_seq_len, self_attn_size, ffn_dimension, layer_norm_eps, dropout_rate_attn, dropout_rate_pwff))

    def forward(self, token_ids, year, month, age, valid_lens=None):
        x = self.embedding_layer(token_ids)
        year_emb = self.year_proj(year)
        month_emb = self.month_proj(month)
        age_emb = self.age_proj(age)

        cont_emb = torch.stack([year_emb, month_emb, age_emb], dim=1)
        x = torch.cat([cont_emb, x], dim=1)
        for blk in self.blks:
            x = blk(x, valid_lens)

        return x
    
class AntibioticClassifier(nn.Module):
    def __init__(self, hidden_size, n_outputs, layer_norm_eps) -> None:
        super().__init__()

        self.mlm = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                                 nn.LayerNorm(hidden_size, eps=layer_norm_eps), nn.Linear(hidden_size, n_outputs))

    def forward(self, x):
        x = self.mlm(x)

        return x
    
class ABPred(nn.Module):
    def __init__(self, hidden_size, n_antibiotics=37, n_outputs=2, layer_norm_eps=0.00001, **kwargs):
        super(ABPred, self).__init__(**kwargs)

        self.nn_stack = nn.ModuleList([AntibioticClassifier(hidden_size, n_outputs, layer_norm_eps) for _ in range(n_antibiotics)])

    def forward(self, X, positions_y):
        outs = [layer(X) for layer in self.nn_stack]
        ab_Y_hat = torch.cat(outs, dim=1)
            
        Y_hat = ab_Y_hat[positions_y]

        return Y_hat