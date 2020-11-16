import torch
from torch import nn
from torch.nn import functional as F
from .effi_model import EfficientNet, VALID_MODELS
from .NST_effi_model import EfficientNet as NST_EfficientNet, VALID_MODELS

from .NSTNet import NST_Net, NST_Net_1d
from .effi_utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

def get_network(x):
    network_dict={
        'NST_Net' : NST_Net,
        'NST_Net_1d' :NST_Net_1d,
        'EffiNet' : get_EffiNet,
        'NST_EffiNet' : get_NST_EffiNet
    }
    return network_dict[x]

def get_EffiNet(ver=0, gradcam=False):
    tmp = EfficientNet.from_name('efficientnet-b{}'.format(ver))
    tmp.gradcam=gradcam
    return tmp

def get_NST_EffiNet(ver=0, gradcam=False):
    tmp = NST_EfficientNet.from_name('efficientnet-b{}'.format(ver))
    tmp.gradcam = gradcam
    return tmp
