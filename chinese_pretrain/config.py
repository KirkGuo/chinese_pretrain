import argparse


def parse_config():
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--image_feature", type=str)

    # experiment config
    parser.add_argument("--stage", choices=['pretrain', 'fine_tune'])
    parser.add_argument("--modality", choices=['t', 'ts', 'ti', 'tis'])

    # model config
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dim_img", type=int)
    parser.add_argument("--dim_model", type=int)
    parser.add_argument("--dim_ff", type=int)
    parser.add_argument("--head", type=int)
    parser.add_argument("--n_enc", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--min_freq", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--n_dec", type=int)

    # dataloader config
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch", type=int)

    # optimizer config
    parser.add_argument("--lr", type=float)
    parser.add_argument("--adam_weight_decay", type=float)
    parser.add_argument("--adam_beta1", type=float)
    parser.add_argument("--adam_beta2", type=float)
    parser.add_argument("--warm_up", type=int)
    parser.add_argument("--smoothing", type=float)

    # hardware config
    parser.add_argument("--gpu", type=bool)

    # log config
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--log", type=bool)
    parser.add_argument("--log_freq", type=int)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--save_model", type=bool)

    # set default args
    parser.set_defaults(train_dataset='data/pretrain_train.csv')
    parser.set_defaults(test_dataset='data/pretrain_val.csv')
    parser.set_defaults(image_feature='data/dic_64.pt')

    parser.set_defaults(stage='pretrain')
    parser.set_defaults(modality='ti')

    parser.set_defaults(seed=1011)
    parser.set_defaults(dim_img=64)
    parser.set_defaults(dim_model=64)
    parser.set_defaults(dim_ff=64)
    parser.set_defaults(head=2)
    parser.set_defaults(n_enc=12)
    parser.set_defaults(dropout=0.4)
    parser.set_defaults(min_freq=1)
    parser.set_defaults(seq_len=33)
    parser.set_defaults(n_dec=1)

    parser.set_defaults(batch_size=128)
    parser.set_defaults(epoch=25)

    parser.set_defaults(lr=1e-4)
    parser.set_defaults(adam_weight_decay=0)
    parser.set_defaults(adam_beta1=0.9)
    parser.set_defaults(adam_beta2=0.98)
    parser.set_defaults(warm_up=15)
    parser.set_defaults(smoothing=0.7)

    parser.set_defaults(gpu=True)

    parser.set_defaults(verbose=True)
    parser.set_defaults(log=True)
    parser.set_defaults(log_freq=1)
    parser.set_defaults(log_path='log')
    parser.set_defaults(save_model=False)

    return parser.parse_args()
