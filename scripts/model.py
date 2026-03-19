#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, n_heads, encoder_stack_depth, ffn_dimension, max_seq_len, vocab_size, 
                 pad_id, dropout_rate_attn, layer_norm_eps, model_name="bert-base-uncased"):
        """
        Initialize transformer encoder.
        
        Args:
            embedding_size: Embedding size
            n_heads: Number of attention heads
            encoder_stack_depth: Number of transformer layers
            ffn_dimension: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            pad_id: Padding token ID
            dropout_rate_attn: Attention dropout rate
            layer_norm_eps: Layer norm epsilon
            model_name: HuggingFace model name (optional - can use custom config instead)
        """
        super().__init__()
        
        self.pad_idx = pad_id
        self.embedding_size = embedding_size
        
        n_continuous = 3

        # Create custom config for transformer
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len + n_continuous,
            hidden_size=embedding_size,
            num_hidden_layers=encoder_stack_depth,
            num_attention_heads=n_heads,
            intermediate_size=ffn_dimension,
            hidden_dropout_prob=dropout_rate_attn,
            attention_probs_dropout_prob=dropout_rate_attn,
            initializer_range=0.02,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_id,
            position_embeddings_type="none"
        )

        # Load transformer model with custom config
        self.transformer = AutoModel.from_config(config)

        # Token embedding layer (same as transformer's)
        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)

        # Continuous feature projections
        self.year_proj = nn.Linear(1, embedding_size)
        self.month_proj = nn.Linear(1, embedding_size)
        self.age_proj = nn.Linear(1, embedding_size)
        
    def forward(self, token_ids, year, month, age, attention_mask):
        """
        Forward pass through the encoder.
        
        Args:
            token_ids: Token indices [batch_size, seq_len]
            year: Year values [batch_size, 1]
            month: Month values [batch_size, 1]
            age: Age values [batch_size, 1]
            attention_mask: Mask for <pad> tokens [batch_size, seq_len + 3] (1 for real tokens/features, 0 for padding)
            
        Returns:
            Encoded representations [batch_size, seq_len + 3, embedding_size]
        """
        
        # Embed tokens
        token_embeds = self.token_embedding(token_ids)

        # Project continuous features
        year_emb = self.year_proj(year)
        month_emb = self.month_proj(month)
        age_emb = self.age_proj(age)
        continuous_embeds = torch.stack([year_emb, month_emb, age_emb], dim=1)

        # Concatenate continuous features and token embeddings
        input_embeds = torch.cat([continuous_embeds, token_embeds], dim=1)

        # Pass through transformer
        transformer_output = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )
        x = transformer_output.last_hidden_state
        return x


class AntibioticClassifier(nn.Module):
    """
    Classification head for a single antibiotic.
    """
    def __init__(self, embedding_size, n_outputs, layer_norm_eps):
        """
        Initialize antibiotic classifier.
        
        Args:
            embedding_size: Input embedding size
            n_outputs: Number of output classes
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_size, eps=layer_norm_eps),
            nn.Linear(embedding_size, n_outputs)
        )
    
    def forward(self, x):
        """
        Forward pass through classifier.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_size]
            
        Returns:
            Classification logits [batch_size, n_outputs]
        """
        return self.classification_head(x)


class AMRPredictionModel(nn.Module):
    """
    Antibiotic prediction model using HuggingFace transformers encoder.
    Maintains same interface as original ABPred but with transformer-based backend.
    """
    def __init__(self, embedding_size, n_antibiotics, antibiotics, layer_norm_eps, n_heads, encoder_stack_depth, 
                 ffn_dimension, max_seq_len, vocab_size, pad_id, dropout_rate_attn,
                 n_outputs_per_antibiotic=None, **kwargs):
        """
        Initialize ABPred model.
        
        Args:
            embedding_size: Embedding size
            n_antibiotics: Number of antibiotics to predict (default 37)
            n_outputs: Number of output classes per antibiotic (default 2)
            layer_norm_eps: Layer normalization epsilon
            n_heads: Number of attention heads
            encoder_stack_depth: Number of transformer layers
            ffn_dimension: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            pad_id: Padding token ID
            dropout_rate_attn: Attention dropout rate
            dropout_rate_pwff: Deprecated - kept for compatibility, not used (HuggingFace uses unified dropout)
            n_outputs_per_antibiotic: List/dict specifying output classes per antibiotic.
                                     If None, all use n_outputs (default behavior).
                                     If list: [2, 3, 2, ...] for each antibiotic
                                     If dict: {'AMC': 2, 'AMI': 3, ...}
        """
        super().__init__(**kwargs)
        
        # Initialize transformer encoder
        self.encoder = TransformerEncoder(
            embedding_size=embedding_size,
            n_heads=n_heads,
            encoder_stack_depth=encoder_stack_depth,
            ffn_dimension=ffn_dimension,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            pad_id=pad_id,
            dropout_rate_attn=dropout_rate_attn,
            layer_norm_eps=layer_norm_eps
        )

        self.n_antibiotics = n_antibiotics
        self.n_outputs_list = [n_outputs_per_antibiotic[ab] for ab in antibiotics]
        
        # Create classifiers with variable output sizes
        self.classifiers = nn.ModuleList([
            AntibioticClassifier(embedding_size, n_out, layer_norm_eps)
            for n_out in self.n_outputs_list
        ])
    
    def forward(self, token_ids, year, month, age, attention_mask):
        encoded = self.encoder(token_ids, year, month, age, attention_mask)
        cls_representation = encoded[:, 3, :] 

        logits_list = []

        for classifier in self.classifiers:
            logits = classifier(cls_representation)
            logits_list.append(logits)

        return logits_list
    