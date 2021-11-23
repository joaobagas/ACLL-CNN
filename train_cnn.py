import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms

from network.dataset_creator import load_labels
from network.net import Net

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
path = ""

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


for i, data in enumerate(dataloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    wanted_values = load_labels(path)

    # Forward + Backward + Optimize
    outputs = net.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("Finished training!")