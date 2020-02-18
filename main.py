# Multiprocessing is used to handle multiple core operations at the same time
import multiprocessing

# Torch and Torchvision are used the machine learning Framework
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

data = mnist_data()

data_loader = torch.utils.