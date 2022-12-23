import torch
from configs import config_args


class Client:
    def __init__(self, client_id, local_dataloader, local_data_size):
        self.client_id = client_id
        self.local_dataloader = local_dataloader
        self.local_data_size = local_data_size
        self.server = None
        if config_args.protocol in ['FedRich']:
            self.local_model = None
            # likelihood distribution
            if config_args.dataset == 'cifar100':
                self.ld = torch.zeros(100)
            else:
                self.ld = torch.zeros(10)
            for _, y in self.local_dataloader:
                for label in y:
                    self.ld[label] += 1

            # Total bathes on each client
            self.batches = len(self.local_dataloader)

            # Participating round of each client
            self.participating_round = 0
            self.participating_example_size = round(self.local_data_size * config_args.sampling_ratio)

            # The upper edge of each client
            self.mediator = None

            # Forward propagation of local model
            self.output = None

            # What client sends to mediator
            self.to_mediator = None

            # What client receives from mediator
            self.from_mediator = None
