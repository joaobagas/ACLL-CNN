import torch
from torch import nn
from torch.autograd.grad_mode import F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, (3, 3), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 3, (3, 3), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )

        # Instantiate fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(6912, 2304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2304, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 6912)
        x = self.fc(x)

        return x
