from configs import config_args
from component.server import Server
import torch
import torch.nn as nn
import copy


class FedRichServer(Server):
    def __init__(self, clients, mediators, test_loader):
        super(FedRichServer, self).__init__(clients, test_loader)
        self.mediators = mediators
        self.client_models = None
        self.mediator_models = None
        self.client_model_param_nums = None

        # determine a target ld for each edge
        total_examples = 0
        if config_args.dataset in ['cifar10', 'cifar100']:
            total_examples = 50000
        elif config_args.dataset in ['mnist', 'fmnist']:
            total_examples = 60000
        target_samples = int(config_args.participating_ratio * total_examples / config_args.mediator_num)
        if config_args.dataset == 'cifar100':
            self.target_ld = torch.tensor([round(target_samples / 100)] * 100)
        else:
            self.target_ld = torch.tensor([round(target_samples / 10)] * 10)

        # split the full model
        self._split_model()

    def _split_model(self, split=False, verbose=True):
        split_points = [int(split_point) for split_point in config_args.split_points.split(', ')]
        split_points.sort()
        assert len(split_points) == config_args.mediator_num

        children = list(self.model.children())
        split_models = []
        for split_point in split_points:
            split_models.append((children[0][:split_point].to(config_args.device),
                                 children[0][split_point:].to(config_args.device)))

        client_models = []
        self.client_models = []
        self.mediator_models = []
        for model in split_models:
            client_models.append(model[0])
            self.mediator_models.append(model[1])
        if config_args.keep_bn:
            if config_args.redundant_forward:
                for layer_index, client_model in enumerate(client_models):
                    layers = []
                    for layer in client_model.children():
                        if isinstance(layer, nn.BatchNorm2d):
                            layer.momentum = 1
                        layers += [layer]
                    client_models[layer_index] = nn.Sequential(*layers)
            self.client_models = client_models
        else:
            for client_model in client_models:
                for layer_index, layer in enumerate(client_model.children()):
                    if isinstance(layer, nn.BatchNorm2d):
                        client_model[layer_index] = nn.Sequential()
                self.client_models.append(client_model)
            if not split:
                layers = []
                for layer in self.client_models[0]:
                    layers.append(layer)
                for layer in self.mediator_models[0]:
                    layers.append(layer)
                self.model = nn.Sequential(nn.Sequential(*layers))

        if verbose:
            for client_model, mediator_model in zip(self.client_models, self.mediator_models):
                print(f'\n{client_model}\n'
                      f'{client_model.state_dict().keys()}\n'
                      f'{mediator_model.state_dict().keys()}\n')
            print(f'\n{self.model.state_dict().keys()}\n'
                  f'split_points: {config_args.split_points}\n')

    def select_participants(self):
        self.current_round += 1
        self.total_size = 0
        for k, mediator in enumerate(self.mediators):
            mediator.select_participants(self.target_ld)
            if k == 0:
                self.participants = copy.deepcopy(mediator.participants)
            else:
                self.participants.extend(mediator.participants)
            self.total_size += mediator.data_size
        print(
            f'Participants in round {self.current_round}: {[client_id for client_id in self.participants]},\n'
            f'Total size in round {self.current_round}: {self.total_size}')

    def aggregate(self, global_lr=1):
        current_model = self.model.state_dict()
        aggregated_model = self.model.state_dict()
        for k, client_id in enumerate(self.participants):
            weight = self.clients[client_id].participating_example_size / self.total_size
            for name, param in self.clients[client_id].local_model.state_dict().items():
                aggregated_model[f'0.{name}'] = aggregated_model[f'0.{name}'] - \
                                                    global_lr * weight * (current_model[f'0.{name}'] - param.data)
            self.clients[client_id].local_model = None
            self.clients[client_id].to_mediator = None
            self.clients[client_id].from_mediator = None
            torch.cuda.empty_cache()

        for k, mediator in enumerate(self.mediators):
            weight = mediator.data_size / self.total_size
            for name, param in mediator.mediator_model.state_dict().items():
                aggregated_model[f'0.{name}'] = aggregated_model[f'0.{name}'] - \
                                                    global_lr * weight * (current_model[f'0.{name}'] - param.data)
        self.model.load_state_dict(aggregated_model)
        self._split_model(split=True, verbose=False)
