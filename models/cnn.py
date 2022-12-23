import torch.nn as nn
from models.flattern import FlattenLayer


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = self._make_layers()

    def _make_layers(self):
        layers = []
        layers += [nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(0, 0), stride=(1, 1), bias=True),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        layers += [nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(0, 0), stride=(1, 1), bias=True),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        layers += [FlattenLayer()]
        layers += [nn.Linear(1024, 512),
                   nn.ReLU(),
                   nn.Linear(512, 10)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


def cnn():
    return CNN()
