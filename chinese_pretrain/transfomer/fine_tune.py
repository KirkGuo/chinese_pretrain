import torch
import torch.nn as nn

from transfomer.model import PretrainModel
from transfomer.model import TokenEmbedding, ImageEmbedding
from transfomer.model import FeatureFusion, Decoder, Generation


class PoemWriteModel(nn.Module):

    def __init__(self, n_vocab, dim_img, dim_model, dim_ff, h, prob_dropout, n_enc, n_dec, n_strain=3, pos_train=False):
        super(PoemWriteModel, self).__init__()

        self.encoder = PretrainModel(n_vocab, dim_img, dim_model, dim_ff, h, prob_dropout, n_enc)

        self.token_emb = TokenEmbedding(n_vocab, dim_model, prob_dropout, n_strain, pos_train)
        self.image_emb = ImageEmbedding(dim_img, dim_model, prob_dropout, pos_train)

        self.fusion = FeatureFusion()

        self.decoder = Decoder(dim_model, dim_ff, h, prob_dropout, n_dec)

        self.generator = Generation(dim_model, n_vocab, prob_dropout)

    def forward(self, source, target):

        (text_target, strains_target), img_target = target

        mem = self.encoder(source)

        out_text = self.token_emb(text_target, strains_target)
        out_img = self.image_emb(img_target)
        out = self.fusion(out_text, out_img)

        out = self.decoder(out, mem)

        out = self.generator(out)

        return out
