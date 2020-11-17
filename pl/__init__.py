import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.classification import Accuracy
from pytorch_lightning import Trainer, seed_everything
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
from .opt_and_loss import get_optimizer, get_scheduler, get_loss_func


class Lt(LightningModule):

    def __init__(self, net, loader_dict: dict, opt_conf, loss_func, cfg, nni=False):
        super().__init__()
        self.network = net  # nn.Module
        self.hparams = cfg  # 통째로 설정 집어넣음
        self.acc = Accuracy()
        self.dataloader = loader_dict
        self.opt_conf = opt_conf
        self.loss = get_loss_func(loss_func)
        self.lr = self.opt_conf.lr
        self.final_result = 5e8
        self.nni = nni

    def forward(self, batch, network):
        x, y = batch
        y_hat = network(x)
        loss = self.loss(y_hat, y)
        accu = self.acc(y_hat, y)
        return y_hat, loss, accu  # 여기 층수 맞추기... 다시 한번 딥러닝 로스 function 공부

    # def configure_optimizers(self):
    #     optimizer = get_optimizer(self.opt_conf.optimizer)(self.parameters(), self.lr)
    #     if self.opt_conf.lr_scheduler == 'OneCycleLR': self.opt_conf.hpara2 = int(
    #         self.hparams.data.total_length / self.hparams.data.batch_size) + 1
    #     scheduler = get_scheduler(self.opt_conf.lr_scheduler, optimizer, self.lr, self.opt_conf.hpara,
    #                               self.opt_conf.hpara2)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = get_optimizer(self.opt_conf.optimizer)(self.parameters(), self.lr)
        if self.opt_conf.scheduler == 'OneCycleLR': self.opt_conf.hpara2 = int(
            self.hparams.data.total_length / self.hparams.data.batch_size) + 1
        scheduler = get_scheduler(self.opt_conf.scheduler, optimizer, self.lr, self.opt_conf.hpara,
                                  self.opt_conf.hpara2)
        configure_optimizers = {'optimizer': optimizer,
                                'lr_scheduler': scheduler,
                                'monitor': 'val_loss'}
        return configure_optimizers

    def train_dataloader(self):
        return self.dataloader["train"]

    def val_dataloader(self):
        return self.dataloader["valid"]

    def test_dataloader(self):
        return self.dataloader["test"]

    def training_step(self, batch, batch_idx):
        pred, loss, acc = self.forward(batch, self.network)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        pred, loss, acc = self.forward(batch, self.network)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        if self.nni:
            import nni
            nni.report_intermediate_result(float(avg_loss))
        if self.final_result > float(avg_loss):
            self.final_result = float(avg_loss)
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        pred, loss, acc = self.forward(batch, self.network)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'log': logs}

