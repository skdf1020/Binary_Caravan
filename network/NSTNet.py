import torch
from torch import nn
from torch.nn import functional as F


class NST_Net(nn.Module):
    '''conv1 layer(1,3,599,70,0,1), fc(387,2), dropout(0.5) '''

    def __init__(self, ver=1):
        super().__init__()
        conv1 = nn.Conv2d(1, 3, 300, stride=70, padding=0, dilation=1)
        bn = nn.BatchNorm2d(3)
        pool1 = nn.MaxPool2d(2)

        fc1 = nn.Linear(396, 2)
        dropout = nn.Dropout(p=0.5)

        if ver == 1:
            self.conv_module = nn.Sequential(
                conv1,
                bn,
                nn.ReLU(),
                pool1
            )

            self.fc_module = nn.Sequential(
                fc1,
                dropout
            )
        if ver == 1_1:
            conv1 = nn.Conv2d(2, 3, 300, stride=70, padding=0, dilation=1)
            self.conv_module = nn.Sequential(
                conv1,
                bn,
                nn.ReLU(),
                pool1
            )

            self.fc_module = nn.Sequential(
                fc1,
                dropout
            )

        conv1 = nn.Conv2d(1, 3, 300, stride=70, padding=0, dilation=2)  #
        fc1 = nn.Linear(198, 2)
        dropout = nn.Dropout(p=0.5)

        if ver == 2:
            self.conv_module = nn.Sequential(
                conv1,
                bn,
                nn.ReLU()
            )
            self.fc_module = nn.Sequential(
                fc1,
                dropout
            )

        conv1 = nn.Conv2d(1, 3, 40, stride=3, padding=0, dilation=1)
        conv2 = nn.Conv2d(3, 5, 30, stride=3, padding=0, dilation=1)
        conv3 = nn.Conv2d(5, 3, 10, stride=1, padding=0, dilation=1)
        conv4 = nn.Conv2d(5, 3, 3, stride=1, padding=0, dilation=1)
        conv5 = nn.Conv2d(3, 3, 3, stride=1, padding=0, dilation=1)
        conv6 = nn.Conv2d(5, 4, 3, stride=1, padding=0, dilation=1)
        conv7 = nn.Conv2d(4, 3, 3, stride=1, padding=0, dilation=1)
        fc1 = nn.Linear(378, 2)
        dropout = nn.Dropout(p=0.5)
        maxpool = nn.MaxPool2d(2)
        bn1 = nn.BatchNorm2d(3)
        bn2 = nn.BatchNorm2d(5)
        bn3 = nn.BatchNorm2d(3)
        bn4 = nn.BatchNorm2d(6)
        pool = nn.MaxPool2d(2)

        if ver == 3:
            self.conv_module = nn.Sequential(
                conv1, bn1, nn.ReLU(), pool,
                conv2, bn2, nn.ReLU(), pool,
                conv3, bn3, nn.ReLU(), pool
            )

            self.fc_module = nn.Sequential(
                fc1,
                dropout
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_module(x)
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return out


class NST_Net_1d(nn.Module):
    '''conv1 layer(1,3,599,70,0,1), fc(387,2), dropout(0.5) '''

    def __init__(self, ver=1, gradcam=False):
        super().__init__()
        self.gradient = None
        self.gradcam = gradcam
        conv1 = nn.Conv1d(1, 3, 300, stride=70, padding=0, dilation=1)
        bn = nn.BatchNorm1d(3)
        self.pool1 = nn.MaxPool1d(2)

        fc1 = nn.Linear(198, 2)
        dropout = nn.Dropout(p=0.5)

        if ver == 1:
            self.conv_module = nn.Sequential(
                conv1,
                bn,
                nn.ReLU()

            )

            self.fc_module = nn.Sequential(
                fc1,
                dropout
            )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_module(x)
        if self.gradcam == True:
            h = x.register_hook(self.activations_hook)
        out = self.pool1(x)
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return out

    def get_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.conv_module(x)

    def activations_hook(self, grad):
        self.gradients = grad