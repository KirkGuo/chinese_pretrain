import torch
import torch.nn as nn
from torch.optim import AdamW, SGD


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, yhat, y):
        return self.loss(yhat, y)


class Scheduler:
    def __init__(self, model, args):
        super(Scheduler, self).__init__()
        self.loss = Loss()
        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )
        self.warm_up = args.warm_up
        self.curr_step = 0
        self.init_lr = args.lr
        self.curr_loss = None

    def __call__(self, out_mask_lm, out_nsp, target):

        mask_pos, mask_label, nsp_label = target
        mask_pos = mask_pos.unsqueeze(-1).expand(mask_pos.size(0), mask_pos.size(1), out_mask_lm.size(-1))
        out_mask_lm = torch.gather(out_mask_lm, 1, mask_pos)
        nsp_label = nsp_label.long()

        # calculate loss
        loss_nsp = self.loss(out_nsp, nsp_label)
        loss_mask_lm = self.loss(out_mask_lm.transpose(1, 2), mask_label)

        self.curr_loss = loss_mask_lm + loss_nsp

        # calculate acc
        pred_mask_lm = out_mask_lm[:, :, :].max(dim=-1)[1]
        pred_nsp_lm = out_nsp[:, :].max(dim=-1)[1]
        mask_lm_acc = pred_mask_lm.eq(mask_label).sum() / len(pred_mask_lm.view(-1))
        nsp_acc = pred_nsp_lm.eq(nsp_label).sum() / len(pred_nsp_lm.view(-1))

        return self.curr_loss.data, mask_lm_acc, nsp_acc

    def step(self, epoch):
        self.curr_loss.backward()
        self._update(epoch)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _update(self, epoch):
        self.curr_step = epoch
        lr = self.init_lr * self._lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _lr_scale(self):

        # if self.curr_step < self.warm_up:
        #      return 1
        # else:
        #     return 2 ** -((self.curr_step - self.warm_up) // 35)

        return 1