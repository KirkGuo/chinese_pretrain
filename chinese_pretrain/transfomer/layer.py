import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transfomer.sublayer import PositionWiseFeedForwardLayer, MultiHeadAttentionLayer


class PositionalEmbeddingLayer(nn.Module):

    def __init__(self, dim_model, max_len=2000):
        super(PositionalEmbeddingLayer, self).__init__()

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = self.pe[:x.size(0), :]
        return out


class TokenEmbeddingLayer(nn.Module):

    def __init__(self, n_token, dim_model):
        super(TokenEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Embedding(n_token, dim_model)

    def forward(self, x):
        out = self.embed(x)
        out = out * np.sqrt(self.dim_model)
        return out


class StrainEmbeddingLayer(nn.Module):
    def __init__(self, n_strain, dim_model):
        super(StrainEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Embedding(n_strain, dim_model)

    def forward(self, x):
        out = self.embed(x)
        out = out * np.sqrt(self.dim_model)
        return out


class FeatureEmbeddingLayer(nn.Module):

    def __init__(self, dim_feature, dim_model):
        super(FeatureEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Linear(dim_feature, dim_model)

    def forward(self, x):
        out = self.embed(x)
        out = out * np.sqrt(self.dim_model)
        return out


class ResidualConnectionLayer(nn.Module):
    def __init__(self, dim_model, prob_dropout):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x, sub):
        out = x + self.dropout(sub)
        out = self.norm(out)
        return out


class EncoderLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, h, prob_dropout):
        super(EncoderLayer, self).__init__()

        self.self_att = MultiHeadAttentionLayer(dim_model, h, prob_dropout)
        self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff, prob_dropout)
        self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)

    def forward(self, x, mask):
        out = self.rc1(x, self.self_att(x, x, x, mask))
        out = self.rc2(out, self.ff(out))
        return out


class DecoderLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, h, prob_dropout):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttentionLayer(dim_model, h, prob_dropout)
        self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.mem_att = MultiHeadAttentionLayer(dim_model, h, prob_dropout)
        self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff, prob_dropout)
        self.rc3 = ResidualConnectionLayer(dim_model, prob_dropout)

    def forward(self, x, memory, mask_x, mask_mem):
        out = self.rc1(x, self.self_att(x, x, x, mask_x))
        out = self.rc2(out, self.mem_att(out, memory, memory, mask_mem))
        out = self.rc3(out, self.ff(out))
        return out


