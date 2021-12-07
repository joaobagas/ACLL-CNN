from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate convolutional layers
        self.conv = nn.Sequential(
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
            nn.Linear(363, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1452)
        x = self.fc(x)

        return x
