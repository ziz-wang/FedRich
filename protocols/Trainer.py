import time
import os
import torch
from configs import config_args


class Trainer:
    def __init__(self):
        print(f'We are running federated training using {config_args.protocol} on {config_args.dataset}... \n')
        self.server = None
        self.results = {'loss': [], 'accuracy': []}

    def begin_train(self):
        assert self.server is not None
        # save path
        localtime = time.localtime(time.time())
        path = f"{config_args.protocol}_{localtime[1]:02}{localtime[2]:02}{localtime[3]:02}{localtime[4]:02}"
        saved_model_path = os.path.join('../saved_models', path)
        saved_results_path = os.path.join('../results', config_args.protocol)
        res_path = os.path.join(saved_results_path, f"{config_args.dataset}_{path}.pt")
        arg_path = os.path.join(saved_results_path, f"{config_args.dataset}_{path}.txt")
        if not os.path.exists(saved_results_path):
            os.makedirs(saved_results_path)

        with open(arg_path, 'w') as f:
            for eachArg, value in config_args.__dict__.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')

        # federated training process
        start_round = self.server.current_round
        print(f"Total rounds: {config_args.rounds - start_round}")
        for _ in range(start_round, config_args.rounds):
            start_time = time.time()
            # participants selection phase
            self.server.select_participants()

            # training and aggregating phase
            self.train_and_aggregate()

            # evaluation phase
            test_acc, test_loss = self.server.validate()
            self.results['accuracy'].append(test_acc)
            self.results['loss'].append(test_loss)
            print(f"[Round: {self.server.current_round: 04}], "
                  f"Accuracy on testset: {test_acc: .4f}, "
                  f"Loss on testset: {test_loss: .4f}, "
                  f"Time spent: {time.time() - start_time: .4f} seconds, "
                  f"Estimated time required to complete the training: "
                  f"{(time.time() - start_time) * (config_args.rounds - self.server.current_round) / 3600}"
                  f" hours.\n")

            # save the model every args.record_step rounds
            if self.server.current_round % config_args.record_step == 0:
                if not os.path.exists(saved_model_path):
                    os.makedirs(saved_model_path)
                torch.save(self.server.model,
                           os.path.join(saved_model_path,
                                        f"{config_args.protocol}"
                                        f"_round_{self.server.current_round}.pth"))
        # save the final results
        torch.save(self.results, res_path)


    def train_and_aggregate(self):
        pass
