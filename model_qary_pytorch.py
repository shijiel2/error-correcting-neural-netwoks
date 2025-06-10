# PyTorch implementation of error-correcting neural network (ECNN)
# NOTE: this file provides a partial translation of TensorFlow code in model_qary.py.
# It is intended as a starting point only and does not replicate training logic.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class Linear(nn.Module):
    """Linear decoding layer using a fixed code matrix."""
    def __init__(self, cm: torch.Tensor):
        super().__init__()
        # store transposed code matrix as weight
        w = torch.tensor(cm, dtype=torch.float32).t()
        self.register_buffer('w', w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mat1 = torch.matmul(torch.sigmoid(x), self.w)
        l1 = torch.clamp(mat1, min=0)
        return l1


def decoder(inputs: torch.Tensor, opt: str = 'dense', drop_prob: float = 0.0,
            cm: Optional[torch.Tensor] = None) -> nn.Module:
    layers: List[nn.Module] = []
    if drop_prob > 0:
        layers.append(nn.Dropout(drop_prob))
    if opt == 'dense':
        layers.append(nn.Linear(inputs, 10))
    elif opt == 'dense_L1':
        lin = nn.Linear(inputs, 10)
        # L1 regularisation to be applied via optimizer
        layers.append(lin)
    elif opt == 'linear':
        layers.append(Linear(cm))
    return nn.Sequential(*layers)


def shared(q: int) -> nn.Linear:
    return nn.Linear(q, q)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetV1(nn.Module):
    def __init__(self, depth: int, num_classes: int = 10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2')
        n = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        strides = [stride] + [1]*(blocks-1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(ResNetBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ECNNModel(nn.Module):
    """Ensemble of ResNet feature extractors with shared dense layers."""
    def __init__(self, num_models: int, q: int, depth: int = 20):
        super().__init__()
        self.backbone = ResNetV1(depth, num_classes=64)
        self.shared_dense = nn.Linear(64, q)
        self.classifiers = nn.ModuleList([
            nn.Linear(q, q) for _ in range(num_models)
        ])
        self.num_models = num_models
        self.q = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        shared_out = self.shared_dense(feat)
        outs = []
        for clf in self.classifiers:
            outs.append(clf(shared_out))
        return torch.cat(outs, dim=1)
