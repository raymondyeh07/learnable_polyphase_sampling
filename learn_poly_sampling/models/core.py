import logging

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
import pytorch_lightning as pl
import numpy as np

from .warmup_scheduler import WarmupScheduler


class AbstractBaseClassifierModel(pl.LightningModule):
    """Abstract class classifiers.
    Reusable code for clasifier models.
    To make new classifiers, just implement the initializer and the forward method
    """

    def __init__(self,
                 optimizer=None, optimizer_kwargs={},
                 scheduler=None, scheduler_kwargs={},
                 param_scheduler=None, param_scheduler_kwargs={},
                 warmup_epochs=0, eval_mode='class_accuracy',
                 shift_seed=7,shift_max=None,
                 shift_samples=None,shift_patch_size=None
    ):
        super().__init__()
        self.optimizer_fn = optimizer if optimizer is not None else torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_fn = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.param_scheduler_fn = param_scheduler
        self.param_scheduler_kwargs = param_scheduler_kwargs
        self.warmup_epochs = warmup_epochs

        # Evaluation settings
        self.eval_mode = eval_mode
        self.shift_seed = shift_seed
        self.shift_max = shift_max
        self.shift_samples = shift_samples
        self.shift_patch_size = shift_patch_size
        logging.info(f'Evaluation mode: {self.eval_mode}')

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss',loss,prog_bar=True,sync_dist=True)
        self.log('val_acc',acc,prog_bar=True,sync_dist=True)
        return loss

    def on_test_start(self) -> None:
        """Called when the test begins."""
        np.random.seed(self.shift_seed)

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        if self.eval_mode=='class_accuracy':
            preds = torch.argmax(logits,dim=1)
            acc = accuracy(preds,y)
            self.log('test_loss',loss,prog_bar=True,sync_dist=True)
            self.log('test_acc',acc,prog_bar=True,sync_dist=True)
        elif self.eval_mode=='shift_consistency':
            self.eval_consistency_step(batch,batch_idx,'test','shift')
            self.eval_consistency_step(batch,batch_idx,'test','circular')
        return loss

    def eval_consistency_step(self, batch, batch_idx,
                              val_mode='test', mode='shift'):
        assert mode in ['shift', 'circular']
        x,_ = batch

        # Read dataset from 'inc_conv1_support'
        if self.inc_conv1_support: dataset = 'imagenet'
        else: dataset = 'cifar10'

        if dataset=='imagenet':
            outputs = []
            if mode == 'shift':
                # Pass shifted inputs
                offsets = [np.random.randint(self.shift_max,size=2) for j in range(0,self.shift_samples)]
                for j in range(0,self.shift_samples):
                    outputs.append(self(x[:,:,offsets[j][0]:offsets[j][0]+self.shift_patch_size,offsets[j][1]:offsets[j][1]+self.shift_patch_size]))
                # Compute consistency
                cur_agree = self.agreement(outputs,self.shift_samples).type(torch.FloatTensor).to(outputs[0].device)
            elif mode == 'circular':
                # Pass rolled inputs
                # -max to max for comparison purposes
                offsets = [np.random.randint(-self.shift_max,self.shift_max,size=2) for j in range(0,self.shift_samples)]
                for j in range(0,self.shift_samples):
                    outputs.append(self(torch.roll(x,shifts=(offsets[j][0],offsets[j][1]),dims=(2,3))))
                # Compute consistency
                cur_agree = self.agreement(outputs,self.shift_samples).type(torch.FloatTensor).to(outputs[0].device)
        elif dataset=='cifar10':
            max_shift = 3
            random_shift1 = torch.randint(-max_shift, max_shift, (2,))
            random_shift2 = torch.randint(-max_shift, max_shift, (2,))
            if mode == 'shift':
                pad_lengths = (max_shift, max_shift, max_shift, max_shift)
                i1_l, i1_r = random_shift1[0] + max_shift, random_shift1[0]-max_shift
                j1_l, j1_r = random_shift1[1] + max_shift, random_shift1[1]-max_shift
                i2_l, i2_r = random_shift2[0] + max_shift, random_shift2[0]-max_shift
                j2_l, j2_r = random_shift2[1] + max_shift, random_shift2[1]-max_shift
                shifted_x1 = F.pad(x, pad_lengths)[:, :, i1_l:i1_r, j1_l:j1_r ]
                shifted_x2 = F.pad(x, pad_lengths)[:, :, i2_l:i2_r, j2_l:j2_r ]
            elif mode == 'circular':
                shifted_x1 = torch.roll(x, shifts = (random_shift1[0], random_shift1[1]), dims = (2, 3))
                shifted_x2 = torch.roll(x, shifts = (random_shift2[0], random_shift2[1]), dims = (2, 3))

            # Compute consistency
            shifted_preds1 = torch.argmax(self(shifted_x1),1)
            shifted_preds2 = torch.argmax(self(shifted_x2),1)
            cur_agree = accuracy(shifted_preds1,shifted_preds2)
        self.log('%s_%s_consistency' % (val_mode, mode), cur_agree, prog_bar=True)
        return

    # (Core) Compute consistency
    def agreement(self,outputs,robust_num):
        preds = torch.stack([output.argmax(dim=1,keepdim=False) for output in outputs], dim=0)
        similarity = torch.sum((preds == preds[0:1,:]).int(), dim=0)
        agree = 100*torch.mean((similarity == robust_num).float())
        return agree

    def optimizer_step( self,epoch,batch_idx,
                        optimizer,optimizer_idx,optimizer_closure,
                        on_tpu,using_native_amp,using_lbfgs):
        """Called when the train epoch begins."""
        # Lr warmup
        if self.current_epoch < self.warmup_epochs:
            it_curr = self.trainer.num_training_batches*self.current_epoch+1+batch_idx
            it_max = self.trainer.num_training_batches*self.warmup_epochs
            lr_scale = float(it_curr) / it_max
            for pg in self.trainer.optimizers[0].param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        # Update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    def configure_optimizers(self):
        logging.info(f'Configuring optimizer: {self.optimizer_fn} with {self.optimizer_kwargs}')

        # Filter net/pool parameters
        net_pars, pool_pars = [], []
        for n,p in self.named_parameters():
            if "component_selection" in n and p.requires_grad_:
                pool_pars.append(p)
            elif p.requires_grad_:
                net_pars.append(p)

        # Set opt
        # Start lr at 0 if warmup #moved to WarmUpScheduler
        #_lr = self.learning_rate
        #print('*************************************************************_lr',_lr)
        _lr= 0 if self.warmup_epochs!=0 else self.learning_rate
        #optimizer = self.optimizer_fn([{'params': net_pars,
        #                                'lr': _lr},
        #                               {'params': pool_pars,
        #                                'weight_decay': 0,
        #                                'lr': _lr}],
        #                              lr=_lr,
        #                              **self.optimizer_kwargs)
        optimizer = self.optimizer_fn([{'params': net_pars},
                                       {'params': pool_pars,'weight_decay': 0}],
                                      lr=_lr,
                                      **self.optimizer_kwargs)

        if self.scheduler_fn is None:
            return optimizer

        logging.info(f'Configuring lr scheduler: {self.scheduler_fn} with {self.scheduler_kwargs}')
        if isinstance(self.scheduler_fn, list):
            schedulers = [sch_fn(optimizer, **sch_kwargs) for sch_fn, sch_kwargs in zip(self.scheduler_fn, self.scheduler_kwargs)]
        else:
            schedulers = [
                {'scheduler': self.scheduler_fn(optimizer, **self.scheduler_kwargs),
                 'frequency': 1,
                 'name': 'main_lr_scheduler'
                 }]

        #if self.warmup_epochs > 0:
        #    schedulers.append({'scheduler': WarmupScheduler(optimizer,
        #                                                    warmup_steps=self.warmup_epochs),
        #                       'frequency': 1,
        #                       'name': 'warmup_scaled_lr'})

        return [optimizer], schedulers

    def configure_callbacks(self):
        if self.param_scheduler_fn is not None:
            logging.info(f'Configuring tau scheduler: {self.param_scheduler_fn} with {self.param_scheduler_kwargs}')
            return [self.param_scheduler_fn('gumbel_tau', **self.param_scheduler_kwargs)]
        return super().configure_callbacks()
