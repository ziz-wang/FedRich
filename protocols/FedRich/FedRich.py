import random
import numpy as np
from protocols.Trainer import Trainer
from configs import config_args
from data.data_utils import get_data_loaders
from protocols.FedRich.FedRichServer import FedRichServer
from protocols.FedRich.FedRichMediator import FedRichMediator
from protocols.FedRich.FedRichClient import FedRichClient


class FedRichTrainer(Trainer):
    def __init__(self):
        super(FedRichTrainer, self).__init__()
        # create server, edges and clients
        local_dataloaders, local_data_sizes, test_dataloader = get_data_loaders()
        # clients
        all_clients = [FedRichClient(_id, local_dataloader, local_data_size)
                       for _id, (local_dataloader, local_data_size)
                       in enumerate(zip(local_dataloaders, local_data_sizes))]

        # edges
        mediator_num = config_args.mediator_num
        indices = list(range(len(all_clients)))
        random.shuffle(indices)
        num_for_mediator = None
        # the number of devices in each mediator follow a exponential/dirichlet distribution
        if config_args.device_distribution == 'exponential':
            beta = config_args.beta
            ratios = []
            for i in range(mediator_num):
                if i + 1 == mediator_num:
                    ratios.append(round(1 - sum(ratios[:i]), 2))
                else:
                    ratios.append(round(beta ** (i + 1), 2))
            ratios = np.array(ratios)
            num_for_mediator = (ratios * len(all_clients)).astype(int)
        elif config_args.device_distribution == 'dirichlet':
            np.random.seed(config_args.seed)
            num_for_mediator = len(all_clients) * np.random.dirichlet(
                np.array(config_args.mediator_num * [config_args.alpha]))
            num_for_mediator = [round(n) for n in num_for_mediator]
            num_for_mediator[-1] = len(all_clients) - sum(num_for_mediator[:-1])
        print(f'Number of client on each edge: {num_for_mediator}')
        all_mediators = []
        for i in range(config_args.mediator_num):
            client_for_mediator = [all_clients[k] for k in indices[:num_for_mediator[i]]]
            indices = indices[num_for_mediator[i]:]
            all_mediators.append(FedRichMediator(i, client_for_mediator))

        # server
        self.server = FedRichServer(all_clients, all_mediators, test_dataloader)
        for client in all_clients:
            client.server = self.server
        for edge in all_mediators:
            edge.server = self.server

    def train_and_aggregate(self):
        # training phase
        for client_id in self.server.participants:
            self.server.clients[client_id].client_forward()
        for mediator in self.server.mediators:
            mediator.mediator_train()
        for client_id in self.server.participants:
            self.server.clients[client_id].client_backward()

        # aggregation phase
        self.server.aggregate()
