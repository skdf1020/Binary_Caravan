import torch
from torch import nn
from torch.nn import functional as F


def get_optimizer(x):
    opt_dict = {
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop,
        'SGD': torch.optim.SGD,
        'AdamW': torch.optim.AdamW
    }
    return opt_dict[x]


def get_scheduler(x, optimizer, lr, hpara, hpara2):
    scheduler_dict = {
        'Step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1),
        'Exponential': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1),
        'Cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1),
        'Reduce': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                             verbose=False, threshold=0.0001, threshold_mode='rel',
                                                             cooldown=0, min_lr=0, eps=1e-08),
        'Cyclic': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.25 * lr, max_lr=4 * lr, step_size_up=2000,
                                                    step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None,
                                                    scale_mode='cycle', cycle_momentum=True, base_momentum=0.8,
                                                    max_momentum=0.9, last_epoch=-1),
        'One': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4 * lr, total_steps=None, epochs=hpara,
                                                   steps_per_epoch=hpara2, pct_start=0.3, anneal_strategy='cos',
                                                   cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                                                   div_factor=25.0, final_div_factor=10000.0, last_epoch=-1),
        'CosineWarm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0,
                                                                           last_epoch=-1)
    }
    return scheduler_dict[x]


def get_loss_func(x):
    loss_dict = {
        'MCE': nn.CrossEntropyLoss(),
        'MSE': nn.MSELoss()
    }
    return loss_dict[x]
