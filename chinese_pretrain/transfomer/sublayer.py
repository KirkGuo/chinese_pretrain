import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.activation import MultiheadAttention

import utils


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim_model, h, prob_dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = MultiheadAttention(dim_model, h, prob_dropout)

    def forward(self, query, key, value, mask):
        # query, key value are arranged as seq * batch_size * dim_model
        # mask should be batch_size*num_heads * query_length * key/value_length
        return self.attention(query, key, value, attn_mask=mask)[0]


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    pad = 0
    x_token = F.relu(torch.randn(64, 10))
    y_token = F.relu(torch.rand(64, 14))
    x = torch.rand(10, 64, 512)
    mem = torch.randn(14, 64, 512)
    attn_mask = utils.attention_mask(x_token, y_token)
    attn_mask = torch.cat([attn_mask.contiguous() for _ in range(4)])
    attn_layer = MultiHeadAttentionLayer(512, 4, 0.1)
    output = attn_layer(x, mem, mem, attn_mask)

