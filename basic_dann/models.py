import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualConv, self).__init__()

        self.shortcut = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + residual)
        return x


class Encoder(nn.Module):
    """
        Feature Extractor
    """

    def __init__(self, in_channel=1, hidden_dims=512):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            ResidualConv(in_channel, 64),
            ResidualConv(64, 128),
            ResidualConv(128, 256),
            ResidualConv(256, 256),
            ResidualConv(256, hidden_dims),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):

        h = self.net(x).squeeze()  # (N, hidden_dims)
        return h


class Classifier(nn.Module):
    """
        Classifier
    """

    def __init__(self, input_size=512, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """

    def __init__(self, input_size=512, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
