import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
# from metrics.iou import iou_ignore as iou

from .thirdparty.utils.loss import FocalLoss
from .thirdparty import network
from .thirdparty import utils
from . import lps_segmentation
from .warmup_scheduler import WarmupScheduler
from .metrics.ddac_metrics import SegmentationMetrics
from .metrics.massc import mASSC_circular, mASSC

import numpy as np


logger = logging.getLogger('SegmentationModule')


class BaseSegmentation(pl.LightningModule):
    def __init__(self, loss_type='focal',
                 ignore_index=255,
                 num_classes=21,
                 learning_rate=0.001,
                 num_shifts=5,
                 max_shift=32,
                 optimizer=None, optimizer_kwargs={},
                 scheduler=None, scheduler_kwargs={},
                 param_scheduler=None, param_scheduler_kwargs={},
                 warmup_epochs=0,
                 scheduler_interval='epoch',
                 backbone_lr_scale=0.1,
                 pool_par_lr_scale=0.1,
                 ) -> None:
        super().__init__()

        self.loss_type = loss_type

        self.optimizer_fn = optimizer if optimizer is not None else torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_fn = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.param_scheduler_fn = param_scheduler
        self.param_scheduler_kwargs = param_scheduler_kwargs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.backbone_lr_scale = backbone_lr_scale
        self.pool_par_lr_scale = pool_par_lr_scale
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        self.scheduler_interval = scheduler_interval

        self.val_acc = Accuracy(num_classes=self.num_classes)
        self.val_segmetrics = SegmentationMetrics(num_classes=self.num_classes)

        self.test_acc = Accuracy(num_classes=self.num_classes)
        self.test_segmetrics = SegmentationMetrics(num_classes=self.num_classes)

        self.loss = self.build_loss()

        self._num_shifts = num_shifts
        self._max_shift = max_shift

    def build_loss(self):
        if self.loss_type == 'focal':
            return FocalLoss(ignore_index=self.ignore_index, size_average=True)
        elif self.loss_type == 'crossentropy':
            return nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y = y.type(torch.LongTensor).to(y_hat.device)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y = y.type(torch.LongTensor).to(y_hat.device)
        loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        y_mask = y != self.ignore_index

        preds = torch.masked_select(preds, y_mask)
        y = torch.masked_select(y, y_mask)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)
        self.log('val_acc', self.val_acc(preds, y),
                 on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)

        self.val_segmetrics(y, preds)

        return loss

    def validation_epoch_end(self, outputs):
        seg_metrics = self.val_segmetrics.compute()
        for k, v in seg_metrics.items():
            if k == 'Class_IoU':
                for vk, vv in v.items():
                    self.log(f'val_iou_{vk}', vv, on_step=False, on_epoch=True, logger=True)
            else:
                self.log(f'val_seg_{k}', v, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        self.val_segmetrics.reset()

    def _shifts(self, x):
        num_shifts = self._num_shifts
        max_shift = self._max_shift
        x_padded = torch.nn.functional.pad(x,
                                           pad=(max_shift, max_shift, max_shift, max_shift),
                                           mode='constant',
                                           value=0)

        shifts = []
        for _ in range(num_shifts):
            #shifted
            #circular shift
            rolls = tuple(np.random.randint(-max_shift, max_shift+1, size=2))
            x_roll = torch.roll(x, shifts=rolls, dims=(-2, -1))
            y_hat_roll = self.forward(x_roll)

            #linear shift with zero padding
            _h, _w = x.shape[-2:]
            x_h0 = rolls[0] + max_shift
            x_h1 = x_h0 + _h
            x_w0 = rolls[1] + max_shift
            x_w1 = x_w0 + _w
            x_shift = x_padded[:, :, x_h0:x_h1, x_w0:x_w1]
            y_hat_shift = self.forward(x_shift)

            shifts.append((y_hat_roll, y_hat_shift, rolls))

        return shifts

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y = y.type(torch.LongTensor).to(y_hat.device)

        shifts = self._shifts(x)

        loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        y_mask = y != self.ignore_index
        preds = torch.masked_select(preds, y_mask)
        y = torch.masked_select(y, y_mask)

        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_acc', self.test_acc(preds, y),
                 on_step=False, on_epoch=True, logger=True)

        self.test_segmetrics(y, preds)
        # TODO: add consistency check
        for idx, (y_hat_roll, y_hat_shift, rolls) in enumerate(shifts):
            self.log(f'test_massc_circular_{idx}',
                     mASSC_circular(y_hat, y_hat_roll, rolls),
                     on_step=False, on_epoch=True, logger=True)
            self.log(f'test_massc_linear_{idx}',
                     mASSC(y_hat, y_hat_shift, rolls),
                     on_step=False, on_epoch=True, logger=True)

        return loss

    def test_epoch_end(self, outputs):
        seg_metrics = self.test_segmetrics.compute()
        for k, v in seg_metrics.items():
            if k == 'Class_IoU':
                for vk, vv in v.items():
                    self.log(f'test_iou_{vk}', vv, on_step=False, on_epoch=True, logger=True)
            else:
                self.log(f'test_seg_{k}', v, on_step=False, on_epoch=True, logger=True)

        self.test_segmetrics.reset()

        logs = self.trainer.callback_metrics

        massc_circular = [logs[f'test_massc_circular_{idx}'].detach().cpu().numpy()
                          for idx in range(self._num_shifts)]
        massc_linear = [logs[f'test_massc_linear_{idx}'].detach().cpu().numpy()
                        for idx in range(self._num_shifts)]

        mean_massc_circular = np.mean(massc_circular)
        mean_massc_linear = np.mean(massc_linear)

        var_massc_circular = np.var(massc_circular)
        var_massc_linear = np.var(massc_linear)

        self.log('test_massc_circular_mean', mean_massc_circular,
                 on_step=False, on_epoch=True, logger=True)
        self.log('test_massc_circular_var', var_massc_circular,
                 on_step=False, on_epoch=True, logger=True)
        self.log('test_massc_linear_mean', mean_massc_linear,
                 on_step=False, on_epoch=True, logger=True)
        self.log('test_massc_linear_var', var_massc_linear,
                 on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        logger.info(f'Configuring optimizer: {self.optimizer_fn} with {self.optimizer_kwargs}')

        # Filter net/pool parameters
        net_pars, pool_pars = [], []
        for n, p in self.model.backbone.named_parameters():
            if "component_selection" in n and p.requires_grad_:
                pool_pars.append(p)
            elif p.requires_grad_:
                net_pars.append(p)



        optimizer = self.optimizer_fn([# {'params': net_pars},
                                       # {'params': pool_pars, 'weight_decay': 0},
                                       {'params': pool_pars,
                                        'lr': self.pool_par_lr_scale * self.learning_rate,
                                        'weight_decay': 0,
                                        'name': 'backbone_pool_pars'
                                       },
                                       {'params': net_pars,
                                        'lr': self.backbone_lr_scale * self.learning_rate,
                                        'name': 'backbone_net_pars'
                                       },
                                       {'params': self.model.classifier.parameters(),
                                        'lr': self.learning_rate,
                                        'name': 'classifier_pars'
                                       }],
                                      lr=self.learning_rate,
                                      **self.optimizer_kwargs)

        if self.scheduler_fn is None:
            return optimizer

        logger.info(
            'Configuring lr scheduler: %s with %s',
            self.scheduler_fn, self.scheduler_kwargs
        )
        if isinstance(self.scheduler_fn, list):
            schedulers = [
                sch_fn(optimizer, **sch_kwargs)
                for sch_fn, sch_kwargs in zip(self.scheduler_fn, self.scheduler_kwargs)
            ]
        else:
            schedulers = [
                {'scheduler': self.scheduler_fn(optimizer, **self.scheduler_kwargs),
                 'interval': self.scheduler_interval,
                 'frequency': 1,
                 'name': 'main_lr_scheduler'
                 }]

        if self.warmup_epochs > 0:
            schedulers.append({'scheduler': WarmupScheduler(optimizer,
                                                            warmup_steps=self.warmup_epochs),
                               'interval': self.scheduler_interval,
                               'frequency': 1,
                               'name': 'warmup_scaled_lr'})

        return [optimizer], schedulers

    def configure_callbacks(self):
        if self.param_scheduler_fn is not None:
            logger.info(
                'Configuring tau scheduler: %s, with %s',
                self.param_scheduler_fn, self.param_scheduler_kwargs)
            return [self.param_scheduler_fn('gumbel_tau', **self.param_scheduler_kwargs)]
        return super().configure_callbacks()


DDAC_MODEL_MAP = {
    'deeplabv3_resnet18': network.deeplabv3_resnet18,
    'deeplabv3plus_resnet18': network.deeplabv3plus_resnet18,
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
    'deeplabv3plus_gpasa_resnet101': network.deeplabv3plus_gpasa_resnet101,
    'deeplabv3plus_lpf_resnet101': network.deeplabv3plus_lpf_resnet101,
    # 'deeplabv3plus_pasa_resnet101': network.deeplabv3plus_pasa_resnet101,
    # 'deeplabv3plus_pasadebug_resnet101': network.deeplabv3plus_pasadebug_resnet101,
    'deeplabv3_resnet_lps': lps_segmentation.lps_resnet_segmentation,
    'deeplabv3plus_resnet_lps': lps_segmentation.lps_resnet_segmentation,
    'deeplabv3plus_resnet_lps_unpool': lps_segmentation.lps_resnet_segmentation_unpool,
}


class DDACSegmentation(BaseSegmentation):  # pylint: disable=too-many-ancestors
    """Lightning wrapper for DDAC models."""
    def __init__(self,
                 model_name,
                 separable_conv=False,
                 output_stride=16,
                 filter_size=3,
                 groups=8,
                 backbone=None,
                 unpool_layer=None,
                 classifier_padding_mode='zeros',
                 **kwargs):  # pylint: disable=too-many-arguments
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['backbone'])
        self.model_name = model_name
        self.separable_conv = separable_conv
        self.output_stride = output_stride
        self.filter_size = 1 if filter_size == None else filter_size
        self.groups = groups
        self.unpool_layer = unpool_layer
        self.classifier_padding_mode = classifier_padding_mode

        if 'gpasa' in self.model_name or 'lpf' in self.model_name or 'pasa' in self.model_name:
            model = DDAC_MODEL_MAP[self.model_name](num_classes=self.num_classes,
                                                    output_stride=self.output_stride,
                                                    filter_size=self.filter_size,
                                                    pasa_group=self.groups)
        elif 'resnet_lps' in self.model_name:
            if 'unpool' in self.model_name:
                # Pass unpool layer and padding mode
                model = DDAC_MODEL_MAP[self.model_name](name=self.model_name,
                                                        backbone_model=backbone,
                                                        unpool_layer=self.unpool_layer,
                                                        classifier_padding_mode=self.classifier_padding_mode,
                                                        num_classes=self.num_classes,
                                                        output_stride=self.output_stride,
                                                        )
            else:
                model = DDAC_MODEL_MAP[self.model_name](name=self.model_name,
                                                        backbone_model=backbone,
                                                        num_classes=self.num_classes,
                                                        output_stride=self.output_stride,
                                                        )
        else:
            model = DDAC_MODEL_MAP[self.model_name](num_classes=self.num_classes,
                                                    output_stride=self.output_stride)

        if self.separable_conv and 'plus' in self.model_name:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        self.model = model

    def forward(self, x):
        return self.model(x)
