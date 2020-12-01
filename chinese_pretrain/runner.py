import os
from time import gmtime, strftime
import torch
import pickle
import json

# from torch.utils.tensorboard import SummaryWriter

import numpy as np


class Runner:

    def __init__(self, model, train_loader, valid_loader,  scheduler, args, log_interval = 20):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = args.epoch
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.curr_step = None
        self.args = args
        self.scores_record = {'epoch': [], 'train_loss': [], 'val_loss': [], 'mask_lm_acc': [], 'nsp_acc': []}

        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        args.log_path = os.path.join(args.log_path, timestamp)

        # make log dir
        os.makedirs(args.log_path)
        pickle.dump(args, open(os.path.join(args.log_path, 'config.pt'), 'wb'))
        print(f'log path : {args.log_path}') if args.verbose else False

    def run(self):
        best_val_loss = float('inf')
        best_mask_lm_acc = -float('inf')
        best_nsp_acc = -float('inf')

        for epoch in range(self.epochs):
            train_loss = self.__train_step(epoch)
            val_loss, val_acc_mask_lm, val_acc_nsp = self.__eval_step(epoch)

            self.scores_record['epoch'].append(epoch)
            self.scores_record['train_loss'].append(train_loss)
            self.scores_record['val_loss'].append(val_loss)
            self.scores_record['mask_lm_acc'].append(val_acc_mask_lm)
            self.scores_record['nsp_acc'].append(val_acc_nsp)

            best_val_loss = min(best_val_loss, val_loss)
            best_mask_lm_acc = max(best_mask_lm_acc, val_acc_mask_lm)
            best_nsp_acc = max(best_nsp_acc, val_acc_nsp)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     self.save_state(epoch)

        print(f'Summary :\nbest_val_loss : {best_val_loss}\nbest_mask_lm_acc : {best_mask_lm_acc}\nbest_nsp_acc : {best_nsp_acc}')

        self.scores_record['best_record'] = [best_val_loss, best_mask_lm_acc, best_nsp_acc]

        json.dump(self.scores_record, open(os.path.join(self.args.log_path, 'scores.json'), 'w'))

    def __train_step(self, epoch):
        self.model.train()
        train_loss = 0.0
        for i, (source, target) in enumerate(self.train_loader):
            source, target = self.__allocate_data(source), self.__allocate_data(target)
            out_mask_lm, out_nsp = self.model(source)

            loss, _, _ = self.scheduler(out_mask_lm, out_nsp, target)
            self.scheduler.step(epoch)
            train_loss += loss.item()
        train_loss = train_loss / len(self.train_loader)
        print('='*50)
        print(f'Epoch [{epoch}/{self.epochs}]: Train batch loss:{train_loss:.6f}')
        return train_loss

    def __eval_step(self, epoch):
        self.model.eval()
        valid_loss = 0.0
        mask_lm_acc = 0.0
        nsp_acc = 0.0
        with torch.no_grad():
            for i, (source, target) in enumerate(self.valid_loader):
                source, target = self.__allocate_data(source), self.__allocate_data(target)
                out_mask_lm, out_nsp = self.model(source)
                loss, curr_mask_lm_acc, curr_nsp_acc = self.scheduler(out_mask_lm, out_nsp, target)
                valid_loss += loss.item()
                mask_lm_acc += curr_mask_lm_acc.item()
                nsp_acc += curr_nsp_acc.item()

        valid_loss = valid_loss / len(self.valid_loader)
        mask_lm_acc = mask_lm_acc / len(self.valid_loader)
        nsp_acc = nsp_acc / len(self.valid_loader)
        print('=' * 50)
        print(f'Epoch [{epoch}/{self.epochs}]: Validation batch loss:{valid_loss:.6f}')
        print(f'Epoch [{epoch}/{self.epochs}]: Validation batch mask_lm_acc:{mask_lm_acc:.6f}')
        print(f'Epoch [{epoch}/{self.epochs}]: Validation batch nsp_acc:{nsp_acc:.6f}')
        return valid_loss, mask_lm_acc, nsp_acc

    def __predict(self):
        pass

    def __evaluate_predict(self):
        pass

    def __allocate_data(self, x):
        if self.args.gpu:
            if type(x) == torch.Tensor:
                return x.to(0)
            else:
                return [self.__allocate_data(each) for each in x]
        return x

    def save_state(self, epoch):
        save_path = os.path.join(self.args.log_path, 'best_mdl.pt')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.scheduler.optimizer.state_dict(),
            'epoch': epoch,
            'config': self.args,
             }, save_path)

        print(f'Save model to checkpoint to: {save_path}')



