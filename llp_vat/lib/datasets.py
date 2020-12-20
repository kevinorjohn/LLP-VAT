import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN


class ToOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, y: int) -> torch.Tensor:
        one_hot = F.one_hot(torch.tensor(y), num_classes=self.num_classes)
        return one_hot.float()


def cifar10(root):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
    num_classes = 10

    transform = {
        "train":
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]),
        "test":
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**channel_stats)])
    }
    train = CIFAR10(root,
                    train=True,
                    transform=transform["train"],
                    target_transform=ToOneHot(num_classes),
                    download=True)
    test = CIFAR10(root,
                   train=False,
                   transform=transform["test"],
                   target_transform=ToOneHot(num_classes),
                   download=True)
    return {'train': train, 'test': test, 'num_classes': num_classes}


def cifar100(root):
    channel_stats = dict(mean=[0.5071, 0.4865, 0.4409],
                         std=[0.2673, 0.2564, 0.2762])
    num_classes = 100

    transform = {
        "train":
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]),
        "test":
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**channel_stats)])
    }
    train = CIFAR100(root,
                     train=True,
                     transform=transform["train"],
                     target_transform=ToOneHot(num_classes),
                     download=True)
    test = CIFAR100(root,
                    train=False,
                    transform=transform["test"],
                    target_transform=ToOneHot(num_classes),
                    download=True)
    return {'train': train, 'test': test, 'num_classes': num_classes}


def svhn(root):
    channel_stats = dict(mean=[0.4377, 0.4438, 0.4728],
                         std=[0.1980, 0.2010, 0.1970])
    num_classes = 10

    transform = {
        "train":
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]),
        "test":
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**channel_stats)])
    }
    train = SVHN(root,
                 split='train',
                 transform=transform["train"],
                 target_transform=ToOneHot(num_classes),
                 download=True)
    test = SVHN(root,
                split='test',
                transform=transform["test"],
                target_transform=ToOneHot(num_classes),
                download=True)
    return {'train': train, 'test': test, 'num_classes': num_classes}


def load_dataset(root, dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        dataset = cifar10(root)
    elif dataset_name == "cifar100":
        dataset = cifar100(root)
    elif dataset_name == "svhn":
        dataset = svhn(root)
    else:
        raise NameError("dataset {} is not supported".format(dataset_name))
    return dataset
