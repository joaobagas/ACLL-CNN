from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 12, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.AvgPool2d(2, 2)
        )

        # Instantiate fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(57600, 3600),
            nn.ReLU(inplace=True),
            nn.Linear(3600, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 57600)
        x = self.fc(x)

        return x
