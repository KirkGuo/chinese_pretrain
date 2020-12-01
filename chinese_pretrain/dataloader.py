import os
import collections
import json
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vocab

import numpy as np
import pandas as pd
from tqdm import tqdm

from task.pretrain_masked_ml import create_masked_lm_predictions


class Tokenizer(object):

    def __init__(self, max_len=33, init_token='s', eos_token='/s', pad_token='pad', sep_token='sep'):
        super(Tokenizer, self).__init__()
        self.max_len = max_len
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token

    def tokenize(self, sentences):
        # out = [' '.join([word for word in sentence]) for sentence in sentences]
        out = [
            ' '.join([word for word in sentence]+[self.pad_token for _ in range(7-len(sentence))])
            for sentence in sentences
        ]
        out = f' {self.sep_token} '.join(out)

        tokens = [self.init_token]
        tokens += out.split()
        tokens += [self.eos_token]
        tokens = tokens + [self.pad_token for _ in range(self.max_len)]

        return tokens[:self.max_len]

    def type_tokenize(self, tokens):
        length = len(tokens)
        i = length - 1
        while tokens[i] != self.sep_token:
            i -= 1
        type_tokens = [0 for _ in range(i+1)] + [1 for _ in range(length-i-1)]
        return type_tokens


class ImageDataset(Dataset):
    def __init__(self, img_path):
        super(ImageDataset, self).__init__()
        self.special_tokens = {'s', 'sep', 'pad', '/s', 'unk'}
        self.dataset = self.load_dataset(img_path)

    def __getitem__(self, item):
        feature_tensor = []
        for each in item:
            if each in self.dataset:
                feature = self.dataset[each]
            else:
                feature = np.random.rand(self.dataset['一'].shape[0])
            word_feature = torch.from_numpy(feature).unsqueeze(0)
            feature_tensor.append(word_feature)
        out_tensor = torch.cat(feature_tensor, dim=0)
        return out_tensor

    def load_dataset(self, path):
        dataset = pickle.load(open(path, 'rb'))
        return dataset


class PretrainDataset(Dataset):
    def __init__(self, meta_path, img_path,
                 init_token='s', sep_token='sep',
                 eos_token='/s', pad_token='pad',
                 unk_token='unk', mask_token='[MASK]'
                 ):

        self.init_token = init_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.image_data = ImageDataset(img_path)
        self.meta_data = pd.read_csv(meta_path)
        self.vocab = self._construct_vocab()

    def __getitem__(self, item):

        token, strain, types, nsp_label = self._parse_item(item)
        token_masked, masked_pos, masked_label = create_masked_lm_predictions(token, 0.15, self.vocab.itos[6:])

        tensor_types = torch.tensor(types).long()
        tensor_strain = self._strain_to_tensor(strain)
        tensor_token_masked = self._token_to_tensor(token_masked)
        tensor_masked_label = self._token_to_tensor(masked_label)
        tensor_masked_pos = torch.tensor(masked_pos).long()
        tensor_img = self.image_data[token].float()

        source = (tensor_token_masked, tensor_strain, tensor_types), tensor_img
        label = tensor_masked_pos, tensor_masked_label, nsp_label

        return source, label

    def __len__(self):
        return len(self.meta_data)

    def _parse_item(self, idx):
        _, poem_id, title, author, sent_1, sent_2, sent_3, sent_4, strain_1, strain_2, strain_3, strain_4, _, nsp_label, *_ = self.meta_data.iloc[idx]
        tokens = Tokenizer().tokenize([sent_1, sent_2, sent_3, sent_4])
        strains = Tokenizer().tokenize([strain_1, strain_2, strain_3, strain_4])
        types = Tokenizer().type_tokenize(tokens)
        return tokens, strains, types, nsp_label

    def _token_to_tensor(self, tokens):
        out = torch.tensor([self.vocab.stoi[each] for each in tokens]).long()
        return out

    def _strain_to_tensor(self, strains):
        look_up = collections.defaultdict(self.__unk_token)
        look_up['平'] = 1
        look_up['仄'] = 2

        out = torch.tensor([look_up[each] for each in strains]).long()
        return out

    def __unk_token(self):
        return 0

    def _construct_vocab(self):
        sentences = self.meta_data['sent1'].tolist()
        sentences += self.meta_data['sent2'].tolist()
        sentences += self.meta_data['sent3'].tolist()
        sentences += self.meta_data['sent4'].tolist()

        word_counter = collections.Counter()
        for sentence in sentences:
            word_counter.update(list(sentence))
        vocab = Vocab(
            word_counter, min_freq=1,
            specials=[self.unk_token, self.init_token, self.eos_token, self.sep_token, self.pad_token, self.mask_token]
        )
        return vocab


class FineTuneDataset:
    pass


if __name__ == '__main__':
    tokenizer = Tokenizer(33)
    test_input = ["床前明月光", "床前明月光", "床前明月光", "床前明月光"]
    test_output = tokenizer.tokenize(test_input)

    test_set = PretrainDataset('data/pretrain_train.csv', 'data/dic_64.pt')
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    for i, each in tqdm(enumerate(test_loader)):
        break
