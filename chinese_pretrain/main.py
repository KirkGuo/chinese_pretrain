import os
import warnings
from datetime import datetime
import pickle

import torch
from torch.utils.data import DataLoader

import numpy as np

import config
from dataloader import PretrainDataset
from transfomer.pretrain import PretrainEmbeddingModel
from loss_scheduler import Scheduler
from runner import Runner


def main():
    args = config.parse_config()

    # set seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.warm_up == -1:
        args.warm_up = int(args.epoch/2)

    # dataset
    print('loading training dataset') if args.verbose else False
    train_dataset = PretrainDataset(args.train_dataset, args.image_feature)

    print('loading validation dataset') if args.verbose else False
    val_dataset = PretrainDataset(args.test_dataset, args.image_feature)

    # model
    print('loading model') if args.verbose else False

    if args.stage == 'pretrain':
        target_model = PretrainEmbeddingModel
    elif args.model == 'fine_tune':
        target_model = None
    else:
        raise ValueError(f'Unknown model : {args.stage}')

    model = target_model(len(train_dataset.vocab), args.dim_img, args.dim_model, args.dim_ff, args.head,
                         args.dropout, args.n_enc, 3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters : {total_params}") if args.verbose else False
    print(f"trainable parameters : {trainable_params}") if args.verbose else False

    if args.gpu:
        torch.cuda.set_device(0)
        model.cuda(0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2)

    # scheduler
    print('loading scheduler') if args.verbose else False
    scheduler = Scheduler(model, args)

    # epoch runner
    print('loading epoch runner') if args.verbose else False
    trainer = Runner(model, train_loader, val_loader, scheduler, args)

    trainer.run()


if __name__ == '__main__':
    main()
