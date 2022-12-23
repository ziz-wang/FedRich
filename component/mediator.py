import torch
import torch.optim as optim
from configs import config_args


class Mediator:
    def __init__(self, mediator_id, clients):
        self.mediator_id = mediator_id
        self.server = None
        self.clients = clients
        self.participants = None
        self.data_size = 0
        self.mediator_model = None
        self.loss = torch.nn.CrossEntropyLoss()

    def select_participants(self, *params):
        pass

    def mediator_train(self):
        self.load_mediator_model()
        lr = config_args.lr * config_args.lr_decay ** (self.server.current_round - 1)
        mediator_optimizer = optim.SGD(self.mediator_model.parameters(),
                                       lr=lr,
                                       momentum=config_args.momentum,
                                       weight_decay=config_args.weight_decay)
        features = None
        labels = None
        client_step_sizes = []

        for client_id in self.participants:
            X, y = self.server.clients[client_id].to_mediator
            client_step_sizes.append(len(X))
            if (features, labels) == (None, None):
                features, labels = X, y.to(config_args.device)
            else:
                features, labels = torch.cat((features, X)), torch.cat((labels, y.to(config_args.device)))
        features.requires_grad_(True)
        features.retain_grad()

        for _ in range(config_args.mediator_epochs):
            if features.grad is not None:
                features.grad.zero_()
            mediator_optimizer.zero_grad()
            train_l = self.loss(self.mediator_model(features), labels)
            train_l.backward()
            mediator_optimizer.step()

        self._send_to_client(features.grad.data, client_step_sizes)

    def load_mediator_model(self):
        pass

    def _send_to_client(self, output_grad, client_step_sizes):
        start = 0
        end = 0
        for i, client_id in enumerate(self.participants):
            step_size = client_step_sizes[i]
            end += step_size
            self.server.clients[client_id].from_mediator = output_grad[start: end]
            start = end
