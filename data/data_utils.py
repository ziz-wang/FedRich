import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import scipy.stats
from torch.utils.data.dataset import Dataset
from configs import config_args
from scipy.stats import wasserstein_distance

# -------------------------------------------------------------------------------------------------------
# DATASETS
# -------------------------------------------------------------------------------------------------------
DATA_PATH = None
if config_args.machine == 'local':
    DATA_PATH = 'data'
elif config_args.machine == 'remote':
    DATA_PATH = '~/dataset/PyTorch'
np.random.seed(config_args.seed)


def get_mnist():
    """Return MNIST train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True)
    data_test = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True)

    x_train, y_train = data_train.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_fmnist():
    """Return FMNIST train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, download=False)
    data_test = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=False)

    x_train, y_train = data_train.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_cifar10():
    """Return CIFAR10 train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=False)
    data_test = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=False)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_cifar100():
    """Return CIFAR100 train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.CIFAR100(root=DATA_PATH, train=True, download=False)
    data_test = torchvision.datasets.CIFAR100(root=DATA_PATH, train=False, download=False)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))


# -------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
# -------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients=10, classes_per_client=10,
                     shuffle=True, verbose=True, balancedness=None):
    """
    Splits (data, labels) evenly among 'n_clients' s.t. every client holds 'classes_per_client' different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    """
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    """
    data_per_client: data size of each client.
    data_per_client_per_class: data size of each class of each client.
    """
    if balancedness >= 1.0:
        """
        Toy example:
            let n_data = 100, n_clients = 10, class_per_client = 2:
                data_per_client = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
                data_per_client_per_class = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        """
        data_per_client = [n_data // n_clients] * n_clients
        data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
    else:
        fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
        fracs /= np.sum(fracs)
        fracs = 0.1 / n_clients + (1 - 0.1) * fracs
        data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]
        data_per_client = data_per_client[::-1]
        data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()

    """
    sort for labels
        data_idcs:
            0: [label_index, ..., label_index]
            ...
            9: [label_index, ..., label_index] 
    """
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split():
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split()

    return clients_split


# -------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
# -------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.tensor(inputs)
        self.labels = torch.tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(name, verbose=True):
    transforms_train = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
    }
    transforms_eval = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return transforms_train[name], transforms_eval[name]


def get_data_loaders(verbose=True):
    x_train, y_train, x_test, y_test = globals()['get_' + config_args.dataset]()
    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)
    transforms_train, transforms_eval = get_default_data_transforms(config_args.dataset, verbose=False)

    split = split_image_data(x_train, y_train, n_clients=config_args.n_clients,
                             classes_per_client=config_args.classes_per_client,
                             balancedness=config_args.balancedness,
                             verbose=verbose)

    client_loaders = [torch.utils.data.DataLoader(
        CustomImageDataset(x, y, transforms_train),
        batch_size=round(x.shape[0] * config_args.sampling_ratio), shuffle=True) for x, y in split]

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval),
                                              batch_size=config_args.batch_size, shuffle=False)

    data_sizes = [x.shape[0] for x, y in split]

    return client_loaders, data_sizes, test_loader


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = globals()['get_' + config_args.dataset]()
    splits = []
    split_image_data(x_train, y_train, n_clients=config_args.n_clients,
                     classes_per_client=config_args.classes_per_client,
                     balancedness=config_args.balancedness,
                     verbose=True)
    splits = np.array(splits)
    global_dist = splits.sum(axis=0) / len(splits)
    heterogeneity = 0
    emds = []
    for split in splits:
        d = wasserstein_distance(split, global_dist)
        emds.append(d)
    d_max = max(emds)
    if d_max == 0:
        for split in splits:
            heterogeneity += JS_divergence(split, global_dist)
    else:
        for split, emd in zip(splits, emds):
            heterogeneity += (JS_divergence(split, global_dist) + emd / d_max)
    heterogeneity /= len(splits)
    print('Heterogeneity: ', heterogeneity)
