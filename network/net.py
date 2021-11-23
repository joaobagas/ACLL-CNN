from torch import nn


class Net(nn.Module):
    # 921 600 -> 614 404 -> 614 404 -> 4

    def __init__(self):
        super(Net, self).__init__()

        # Instantiate the ReLU non-linearity
        self.relu = nn.ReLU()

        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Instantiate a average pooling layer
        self.pool = nn.AvgPool2d(2, 2)  # Half the dimension size

        # Instantiate a fully connected layer
        neurons_left = int(921600 / 4)
        self.fc = nn.Linear(neurons_left, 4)

    def forward(self, x):
        # Apply conv followed by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followed by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Apply the fully connected layer
        x = self.fc(x)

        # Return the result
        return x
    