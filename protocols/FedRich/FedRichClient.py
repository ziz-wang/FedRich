from component.client import Client
from configs import config_args
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
import copy


class FedRichClient(Client):
    def __init__(self, client_id, local_dataloader, local_data_size):
        super(FedRichClient, self).__init__(client_id, local_dataloader, local_data_size)

    def set_mediator(self, mediator):
        self.mediator = mediator

    def client_forward(self):
        self.local_model = copy.deepcopy(self.server.client_models[self.mediator.mediator_id])
        self.local_model.load_state_dict(self.server.client_models[self.mediator.mediator_id].state_dict())
        for batch, (X, y) in enumerate(self.local_dataloader):
            if batch % self.batches == self.participating_round % self.batches:
                X.requires_grad_(True)
                if config_args.redundant_forward:
                    self.local_model.apply(self._fix_bn)
                self.output = checkpoint(self.local_model, X.to(config_args.device))
                if config_args.feature_memo:
                    print('Output size: {}, Memory usage of features: {:4f}M'.format(
                        self.output.size(), self.output.flatten().size()[0] * 4 / 1024 / 1024))
                self.to_mediator = (self.output.clone().detach(), y)

    def _fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def client_backward(self):
        lr = config_args.lr * config_args.lr_decay ** (self.server.current_round - 1)
        local_optimizer = optim.SGD(self.local_model.parameters(),
                                    lr=lr,
                                    momentum=config_args.momentum,
                                    weight_decay=config_args.weight_decay)
        local_optimizer.zero_grad()
        self.output.backward(self.from_mediator)
        local_optimizer.step()

        # update bn
        if config_args.redundant_forward:
            self.local_model.train()
            for batch, (X, y) in enumerate(self.local_dataloader):
                if batch % self.batches == self.participating_round % self.batches:
                    self.local_model(X.to(config_args.device))
        self.participating_round += 1
