from configs import config_args
from component.mediator import Mediator
import torch
import random
import copy


class FedRichMediator(Mediator):
    def __init__(self, mediator_id, clients):
        super(FedRichMediator, self).__init__(mediator_id, clients)
        for client in self.clients:
            client.set_mediator(self)
        self.ld = None
        for k, client in enumerate(self.clients):
            client.set_mediator(self)
            if k == 0:
                self.ld = copy.deepcopy(client.ld)
            else:
                self.ld += client.ld

    def select_participants(self, target_ld):
        participant_num = round(config_args.participating_ratio * len(self.server.clients) / config_args.mediator_num)
        assert participant_num <= len(self.clients)
        self.data_size = 0
        to_be_selected = None
        self.participants = None
        if config_args.heuristic_search:
            current_ld = None
            while participant_num:
                if not self.participants:
                    random_select_participant_num = max(1, int(participant_num * config_args.random_select_ratio))
                    print(f'Mediator {self.mediator_id} randomly selects {random_select_participant_num} participants.')
                    selected_client = random.sample(self.clients, random_select_participant_num)
                    self.participants = selected_client
                    to_be_selected = list(set(self.clients) - set(self.participants))
                    for i, client in enumerate(selected_client):
                        if i == 0:
                            current_ld = copy.deepcopy(client.ld)
                        else:
                            current_ld += client.ld
                    participant_num -= random_select_participant_num
                else:
                    min_loss = float('inf')
                    candidate = None
                    for client in to_be_selected:
                        tmp_ld = current_ld + client.ld
                        loss = self._cal_loss(tmp_ld, target_ld)
                        if loss < min_loss:
                            min_loss = loss
                            candidate = client
                    self.participants.append(candidate)
                    to_be_selected.remove(candidate)
                    current_ld += candidate.ld
                    participant_num -= 1
            self.participants = [client.client_id for client in self.participants]
        else:
            to_be_selected = [client.client_id for client in self.clients]
            self.participants = random.sample(to_be_selected, participant_num)
        for client_id in self.participants:
            self.data_size += self.server.clients[client_id].participating_example_size

    def _cal_loss(self, current_ld, target_ld):
        class_loss = 10000 * torch.exp(len(current_ld) - torch.count_nonzero(
            (current_ld - torch.div(target_ld, 2, rounding_mode='floor')) > 0))
        num_loss = torch.norm((current_ld - target_ld).type(torch.float), p=1)
        return class_loss + num_loss

    def load_mediator_model(self):
        self.mediator_model = copy.deepcopy(self.server.mediator_models[self.mediator_id])
        self.mediator_model.load_state_dict(self.server.mediator_models[self.mediator_id].state_dict())
