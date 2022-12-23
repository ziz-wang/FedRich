from configs import config_args
from protocols.FedRich.FedRich import FedRichTrainer


trainer = None
if config_args.protocol == 'FedRich':
    trainer = FedRichTrainer()
trainer.begin_train()
