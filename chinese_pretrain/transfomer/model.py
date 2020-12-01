from copy import deepcopy

import torch
import torch.nn as nn

from transfomer.layer import TokenEmbeddingLayer, FeatureEmbeddingLayer, PositionalEmbeddingLayer, StrainEmbeddingLayer
from transfomer.layer import EncoderLayer, DecoderLayer


class TokenEmbedding(nn.Module):

    def __init__(self, n_vocab, dim_model, n_strain=3):
        super(TokenEmbedding, self).__init__()
        self.token = TokenEmbeddingLayer(n_vocab, dim_model)
        self.strain = StrainEmbeddingLayer(n_strain, dim_model)

    def forward(self, x, strains=None):
        out = self.token(x)
        out = out + self.strain(strains) if strains is not None else out
        return out


class ImageEmbedding(nn.Module):

    def __init__(self, dim_image, dim_model):
        super(ImageEmbedding, self).__init__()
        self.img = FeatureEmbeddingLayer(dim_image, dim_model)

    def forward(self, x):
        out = self.img(x)
        return out


class FeatureFusion(nn.Module):

    def __init__(self, dim_model):
        super(FeatureFusion, self).__init__()
        self.linear = nn.Linear(dim_model + dim_model, dim_model)

    def forward(self, text, img):
        out = torch.cat([text, img], dim=-1)
        out = self.linear(out)
        return out


class GateSelectionLayer(nn.Module):

    def __init__(self, dim_model):
        super(GateSelectionLayer, self).__init__()
        self.reset = nn.Linear(dim_model*2, dim_model)
        self.update = nn.Linear(dim_model*2, dim_model)
        self.proposal = nn.Linear(dim_model*2, dim_model)

    def forward(self, x_1, x_2, *args):
        reset = torch.sigmoid(self.reset(torch.cat([x_1, x_2], -1)))
        update = torch.sigmoid(self.update(torch.cat([x_1, x_2], -1)))
        proposal = torch.tanh(self.proposal(torch.cat([reset * x_1, x_2], -1)))
        out = (1 - update) * x_1 + update * proposal
        return out


class InputEmbedding(nn.Module):

    def __init__(self, n_vocab, dim_img, dim_model, prob_dropout, n_strain=3):
        super(InputEmbedding, self).__init__()
        self.token_emb = TokenEmbedding(n_vocab, dim_model, n_strain)
        self.img_emb = ImageEmbedding(dim_img, dim_model)
        self.fusion = GateSelectionLayer(dim_model)
        self.pos_emb = PositionalEmbeddingLayer(dim_model)
        self.seg_emb = nn.Embedding(2, dim_model)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, tokens, strains, seg, img):

        out_text = self.token_emb(tokens, strains)
        out_img = self.img_emb(img)
        out = self.fusion(out_text, out_img)
        out += self.seg_emb(seg)
        out += self.pos_emb(out)

        return self.dropout(out)


class Encoder(nn.Module):

    def __init__(self, dim_model, dim_ff, h, prob_dropout, n):
        super(Encoder, self).__init__()
        encoder_layer = EncoderLayer(dim_model, dim_ff, h, prob_dropout)
        self.stack = nn.ModuleList([deepcopy(encoder_layer) for _ in range(n)])

    def forward(self, x, mask=None):
        out = x.transpose(0, 1).contiguous()
        for layer in self.stack:
            out = layer(out, mask)
        out = out.transpose(0, 1).contiguous()
        return out


class Decoder(nn.Module):
    def __init__(self, dim_model, dim_ff, h, prob_dropout, n):
        super(Decoder, self).__init__()
        decoder_layer = DecoderLayer(dim_model, dim_ff, h, prob_dropout)
        self.stack = nn.ModuleList([deepcopy(decoder_layer) for _ in range(n)])

    def forward(self, x, mem, mask_x, mask_mem):
        out = x.transpose(0, 1).contiguous()
        for layer in self.stack:
            out = layer(out, mem, mask_x, mask_mem)
        out = out.transpose(0, 1).contiguous()
        return x


class Generation(nn.Module):
    
    def __init__(self, dim_model, n_vocab, prob_dropout):
        super(Generation, self).__init__()
        self.fc1 = nn.Linear(dim_model, n_vocab)
        self.dropout = nn.Dropout(prob_dropout)
        self.fc2 = nn.Linear(n_vocab, n_vocab)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        out = torch.log_softmax(out, dim=-1)
        return out


class PretrainModel(nn.Module):
    
    def __init__(self, n_vocab, dim_img, dim_model, dim_ff, h, prob_dropout, n_enc, n_strain=3):
        super(PretrainModel, self).__init__()
        
        self.emb = InputEmbedding(n_vocab, dim_img, dim_model, prob_dropout, n_strain)
        self.encoder = Encoder(dim_model, dim_ff, h, prob_dropout, n_enc)

    def forward(self, source):
        (text, strains, seg), img = source

        out = self.emb(text, strains, seg, img)
        out = self.encoder(out)

        return out
