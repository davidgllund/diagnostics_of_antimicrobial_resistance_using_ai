#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, n_heads, encoder_stack_depth, ffn_dimension, max_n_tokens, vocab_size, 
                 pad_id, dropout_rate_attn, layer_norm_eps, model_name="bert-base-uncased"):
        super().__init__()
        
        self.pad_idx = pad_id
        self.embedding_size = embedding_size

        n_continuous = 3

        # Create custom config for transformer
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            max_position_embeddings=max_n_tokens + n_continuous,
            hidden_size=embedding_size,
            num_hidden_layers=encoder_stack_depth,
            num_attention_heads=n_heads,
            intermediate_size=ffn_dimension,
            hidden_dropout_prob=dropout_rate_attn,
            attention_probs_dropout_prob=dropout_rate_attn,
            initializer_range=0.02,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_id
        )

        # Load transformer model with custom config
        self.transformer = AutoModel.from_config(config)

        # No positional encoding
        with torch.no_grad():
            self.transformer.embeddings.position_embeddings.weight.zero_()
        self.transformer.embeddings.position_embeddings.weight.requires_grad_(False)
        
        # Token embedding layer (same as transformer's)
        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.transformer.embeddings.word_embeddings = self.token_embedding
        
        # Continuous feature projections
        self.year_proj = nn.Linear(1, embedding_size)
        self.month_proj = nn.Linear(1, embedding_size)
        self.age_proj = nn.Linear(1, embedding_size)
        
    def forward(self, token_ids, year, month, age, attention_mask):
        # Embed tokens
        token_embeds = self.token_embedding(token_ids)

        # Project continuous features
        year_emb = self.year_proj(year)
        month_emb = self.month_proj(month)
        age_emb = self.age_proj(age)
        continuous_embeds = torch.stack([year_emb, month_emb, age_emb], dim=1)

        # Concatenate continuous features and token embeddings
        input_embeds = torch.cat([continuous_embeds, token_embeds], dim=1)

        # Continuous features are never padding, so always attend to them.
        continuous_mask = attention_mask.new_ones(attention_mask.size(0), continuous_embeds.size(1))
        attention_mask = torch.cat([continuous_mask, attention_mask], dim=1)

        x = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        return x
    

class AntibioticClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_outputs, layer_norm_eps, antibiotic):
        super().__init__()

        self.antibiotic = antibiotic

        n_sampling_time_data = 2
        input_size = embedding_size + n_sampling_time_data 

        self.classification_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            nn.Linear(hidden_size, n_outputs)
        )
        
    def forward(self, x):
        return self.classification_head(x)


class AMRPredictionModel(nn.Module):
    def __init__(self, embedding_size, n_antibiotics, antibiotics, layer_norm_eps, n_heads, encoder_stack_depth, 
                 ffn_dimension, max_n_tokens, vocab_size, pad_id, dropout_rate_attn, hidden_size,
                 n_outputs_per_antibiotic=None, **kwargs):
        
        super().__init__(**kwargs)
        
        self.encoder = TransformerEncoder(
            embedding_size=embedding_size,
            n_heads=n_heads,
            encoder_stack_depth=encoder_stack_depth,
            ffn_dimension=ffn_dimension,
            max_n_tokens=max_n_tokens,
            vocab_size=vocab_size,
            pad_id=pad_id,
            dropout_rate_attn=dropout_rate_attn,
            layer_norm_eps=layer_norm_eps
        )

        self.n_antibiotics = n_antibiotics
        self.n_outputs_list = [n_outputs_per_antibiotic[ab] for ab in antibiotics]
        
        self.classifiers = nn.ModuleList([
            AntibioticClassifier(embedding_size, hidden_size, n_out, layer_norm_eps, ab)
            for n_out, ab in zip(self.n_outputs_list, antibiotics)
        ])
    
    def forward(self, token_ids, year, month, age, attention_mask):
        encoded = self.encoder(token_ids, year, month, age, attention_mask)
        cls_representation = encoded[:, 3, :] 

        sampling_time_data = torch.cat([
            (year - 2015) / 10.0, 
            month / 12.0 
        ], dim=1)

        logits_list = []

        for classifier in self.classifiers:

            cls_input = torch.cat([cls_representation, sampling_time_data], dim=1)
            logits = classifier(cls_input)

            logits_list.append(logits)

        return logits_list