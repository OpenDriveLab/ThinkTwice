from collections import deque
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import force_fp32, auto_fp16

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        self.fp16_enabled = False


    @force_fp32()
    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds.float(), labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

def set_dropout_zero(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1 or classname.find('Dropout2d') != -1:
            m.p = 0
        elif classname.find('DropPath') != -1:
            m.drop_prob = 0



def init_weights(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('Linear') != -1:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(m.weight)
        elif classname.find('Embedding') != -1:
            nn.init.trunc_normal_(m.weight, mean=0, std=0.02)

        

class SEModule(nn.Module):
    def __init__(self, channels, act):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = act()
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, act):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * 2, kernel_size=3, padding=1,)
        self.bn1 = nn.BatchNorm2d(inplanes * 2)
        self.act1 = act()
        self.conv2 = nn.Conv2d(inplanes*2, inplanes, kernel_size=3, padding=1,)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.act2 = act()
        self.se = SEModule(inplanes, act)
        self.act3 = act()
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x += shortcut
        x = self.act3(x)
        return x

