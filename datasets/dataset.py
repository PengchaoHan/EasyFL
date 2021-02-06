# Part of this code is adapted from https://github.com/jz9888/federated-learning
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100


def get_data_info(dataset, model_name):
    if dataset == 'MNIST':
        if model_name == 'ModelCNNMnist':
            img_size = 28
        elif model_name == 'LeNet5':
            img_size = 32
        else:
            raise Exception('Unknown model name for MNIST.')
        channels = 1
        num_classes = 10
    elif dataset == 'SVHN':
        img_size = 32
        channels = 3
        num_classes = 10
    elif dataset == 'cifar10':
        img_size = 32
        channels = 3
        num_classes = 10
    elif dataset == 'cifar100':
        img_size = 32
        channels = 3
        num_classes = 100
    elif dataset == 'FEMNIST':
        img_size = 28
        channels = 1
        num_classes = 62
    elif dataset == 'celeba':
        img_size = 84
        channels = 3
        num_classes = 2
    else:
        raise Exception('Unknown dataset name.')
    return img_size, channels, num_classes


def load_data(dataset, data_path, model_name):
    img_size, _, _ = get_data_info(dataset, model_name)
    if dataset == 'MNIST':
        data_train = MNIST(data_path,
                           transform=transforms.Compose([
                               transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]),
                           download=True)  # True for the first time
        data_test = MNIST(data_path,
                          train=False,
                          transform=transforms.Compose([
                              transforms.Resize((img_size, img_size)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    elif dataset == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        data_train = SVHN(data_path+'/SVHN', split='train', download=True, transform=transform_train)
        data_test = SVHN(data_path+'/SVHN', split='test', download=True, transform=transform_test)

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR10(data_path,
                             transform=transform_train,
                             download=True)  # True for the first time
        data_test = CIFAR10(data_path,
                            train=False,
                            transform=transform_test)

    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR100(data_path,
                              transform=transform_train,
                              download=True)  # True for the first time
        data_test = CIFAR100(data_path,
                             train=False,
                             transform=transform_test)

    elif dataset == 'FEMNIST':
        from datasets.femnist import FEMNIST

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        data_train = FEMNIST(data_path+'/femnist/', transform=transform, train=True)
        data_test = FEMNIST(data_path+'/femnist/', transform=transform, train=False)

    elif dataset == 'celeba':
        from datasets.celeba import CelebA

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            # transforms.RandomCrop((84, 84)),
            transforms.CenterCrop((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        data_train = CelebA(data_path + '/celeba/', transform=transform, train=True, read_all_data_to_mem=False)
        data_test = CelebA(data_path + '/celeba/', transform=transform,  train=False, read_all_data_to_mem=False)

    else:
        raise Exception('Unknown dataset name.')

    return data_train, data_test
