import torch
import os
import re
from configs import config_args
from models.create_model import create_model
from component.evaluation import evaluate_accuracy


class Server:
    def __init__(self, clients, test_loader):
        if config_args.resumed:
            self.model = torch.load(os.path.join('../saved_models', config_args.resumed_name),
                                    map_location=config_args.device)
        else:
            self.model = create_model(config_args.dataset)
        self.clients = clients
        self.participants = None
        self.test_loader = test_loader
        self.total_size = 0
        self.current_round = 0
        if config_args.resumed:
            self.current_round = int(re.findall(r'\d+\d*', config_args.resumed_name.split('/')[1])[0])
            test_acc, test_l = self.validate()
            print(f"Accuracy on testset: {test_acc: .4f}, "
                  f"Loss on testset: {test_l: .4f}\n")

    # should be overridden in some subclass
    def select_participants(self):
        pass

    # should be overridden in some subclass
    def aggregate(self, global_lr=1):
        pass

    def validate(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.model, self.test_loader)
        return test_acc, test_l
