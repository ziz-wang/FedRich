import torch.nn as nn
from models.flattern import FlattenLayer

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


def make_layers(cfg, batch_norm=True, num_class=100):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=(3, 3), padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU()]
        input_channel = l
    layers += [FlattenLayer()]
    layers += [nn.Linear(512, 4096)]
    layers += [nn.ReLU()]
    layers += [nn.Dropout()]
    layers += [nn.Linear(4096, 4096)]
    layers += [nn.ReLU()]
    layers += [nn.Dropout()]
    layers += [nn.Linear(4096, num_class)]

    return nn.Sequential(*layers)


def vgg11():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    return VGG(make_layers(cfg['D'], batch_norm=True, num_class=10))


def vgg19():
    return VGG(make_layers(cfg['E'], batch_norm=True))
