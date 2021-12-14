import torch
from torch import nn
from torch.autograd.grad_mode import F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate convolutional layers
        self.h_conv = nn.Sequential(
            nn.Conv2d(3, 6, (3, 3), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(6, 12, (3, 3), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4)
        )

        self.l_conv = nn.Sequential(
            nn.Conv2d(3, 6, (3, 3), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4, 4),
            nn.Conv2d(6, 12, (3, 3), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4, 4)
        )

        # Instantiate fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1452, 363),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(363, 121),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(121, 3)
        )

    def forward(self, x):
        hx = self.h_conv(x)
        lx = self.l_conv(x)
        hx = hx.view(-1, 1452)
        lx = lx.view(-1, 1452)
        hx = self.fc(hx)
        lx = self.fc(lx)
        x = hx + lx

        return x
