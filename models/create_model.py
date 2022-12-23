from models.cnn import cnn
from models.vgg import vgg16, vgg19
from configs import config_args
from torch import nn
import numpy as np
import torch


# @ profile
def create_model(dataset):
    model = None
    if dataset in ['mnist', 'fmnist']:
        model = nn.Sequential(list(cnn().children())[0])
    elif dataset == 'cifar10':
        model = nn.Sequential(list(vgg16().children())[0])
    elif dataset == 'cifar100':
        model = nn.Sequential(list(vgg19().children())[0])
    return model.to(config_args.device)


if __name__ == '__main__':
    model = create_model(config_args.dataset).cpu()
    # children = list(model.children())
    # model = children[0][:8]
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # type_size = 4
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1024 / 1024))
    from thop import profile
    input = torch.randn(500, 3, 32, 32)
    flops, params = profile(model, inputs=(input,))
    print(flops / 1e9, params / 1e6)  # flops单位G，para单位M
