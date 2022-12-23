import torch
from configs import config_args


class Accumulator:
    """
    Toy example:
        let n = 3:
            initialize: self.data = [0.0, 0.0, 0.0]
            add(test_l, test_acc, length):  self.data = [0.0 + test_l, 0.0 + test_acc, 0.0 + length]
            self.data[0] / self.data[2]: average loss
            self.data[1] / self.data[2]: average accuracy
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *res):
        self.data = [a + float(b) for a, b in zip(self.data, res)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """
    The number of "exact prediction" in a batch.
    """
    cmp = (y_hat.argmax(dim=1).type(y.dtype) == y).sum().float()
    return cmp


def evaluate_accuracy(model, data_iter, loss=torch.nn.CrossEntropyLoss()):
    """
    Calculate testing accuracy and testing loss in a testing dataset.
    """
    model.eval()
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(config_args.device)
        y = y.to(config_args.device)
        y_pred = model(X)
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    model.train()
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_accuracy_hfl(client_model, mediator_model, data_iter, loss=torch.nn.CrossEntropyLoss()):
    client_model.eval()
    mediator_model.eval()
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(config_args.device)
        y = y.to(config_args.device)
        y_pred = mediator_model(client_model(X))
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    client_model.train()
    mediator_model.train()
    return metric[0] / metric[2], metric[1] / metric[2]


