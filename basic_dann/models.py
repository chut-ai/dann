import torch.nn as nn
from dann.transfer import get_loader
from dann.basic_dann.function import ReverseLayerF

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("conv1", nn.Conv2d(1, 32, kernel_size=5))
        self.net.add_module("bn1", nn.BatchNorm2d(32))
        self.net.add_module("pool1", nn.MaxPool2d(2))
        self.net.add_module("relu1", nn.ReLU(True))
        self.net.add_module("conv2", nn.Conv2d(32, 64, kernel_size=5))
        self.net.add_module("bn2", nn.BatchNorm2d(64))
        self.net.add_module("pool2", nn.MaxPool2d(2))
        self.net.add_module("relu2", nn.ReLU(True))
        self.net.add_module("drop1", nn.Dropout2d())

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 64*4*4)
        return x

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("fc1", nn.Linear(64*4*4, 500))
        self.net.add_module("bn1", nn.BatchNorm1d(500))
        self.net.add_module("relu1", nn.ReLU(True))
        self.net.add_module("drop1", nn.Dropout2d())
        self.net.add_module("fc2", nn.Linear(500, 100))
        self.net.add_module("bn2", nn.BatchNorm1d(100))
        self.net.add_module("relu2", nn.ReLU(True))
        self.net.add_module("fc3", nn.Linear(100, 10))
        self.net.add_module("softmax", nn.Softmax(dim=1))

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("fc1", nn.Linear(64*4*4, 100))
        self.net.add_module("bn1", nn.BatchNorm1d(100))
        self.net.add_module("relu1", nn.ReLU(True))
        self.net.add_module("fc2", nn.Linear(100, 2))
        self.net.add_module("softmax", nn.Softmax(dim=1))

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = self.net(x)
        return x
