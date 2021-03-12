import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def get_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    trainset = datasets.MNIST("~/MNIST/train", download=False,
                              train=True, transform=transform)
    valset = datasets.MNIST("~/MNIST/val", download=False,
                            train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainloader, valloader

def get_svhn(batch_size, batch_nb=938):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5),
                                    transforms.Resize(28),
                                    ])

    dataset = datasets.SVHN("~/data/", download=False, transform=transform)

    train_size = batch_size*batch_nb
    val_size = len(dataset) - train_size

    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader
