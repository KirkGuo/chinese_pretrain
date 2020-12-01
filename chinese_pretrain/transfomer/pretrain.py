import torch
import torch.nn as nn

from transfomer.model import PretrainModel


class Classifier(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.fc2 = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        out = self.fc(x)
        out = self.fc2(torch.relu(out))
        out = torch.log_softmax(out, dim=-1)
        return out


class PretrainEmbeddingModel(nn.Module):

    def __init__(self, n_vocab, dim_img, dim_model, dim_ff, h, prob_dropout, n_enc, n_strain=2):
        super(PretrainEmbeddingModel, self).__init__()
        self.encoder = PretrainModel(n_vocab, dim_img, dim_model, dim_ff, h, prob_dropout, n_enc, n_strain)
        self.mask_lm = Classifier(dim_model, n_vocab)
        self.nsp = Classifier(dim_model, 2)

    def forward(self, source):
        out = self.encoder(source)
        out_mask_lm = self.mask_lm(out)
        out_nsp = self.nsp(out[:, 0, :])
        return out_mask_lm, out_nsp
