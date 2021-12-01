import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms

from network.dataset_creator import load_labels
from network.loss_function import compute_score
from network.module import Net

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=3e-4)
path = "/home/bernardo/Desktop/Images/No animals/Dataset1"

transform = transforms.ToTensor()
dataset = datasets.ImageFolder(path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


for i, data in enumerate(dataloader, 0):
    inputs, labels_not_used = data
    labels = load_labels(path)

    # Forward + Backward + Optimize
    outputs = net.forward(inputs)
    loss = criterion(outputs, torch.Tensor(labels[i]).unsqueeze(0))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


print("Finished training!")

correct, total = 0, 0
predictions = []
net.eval()

for i, data in enumerate(dataloader, 0):
    inputs, labels = data
    labels = load_labels(path)
    labels = torch.Tensor(labels)

    outputs = net.forward(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    print("pred: " + str(predictions[i].tolist()[0]))
    print("labl: " + str(labels.tolist()[i]))
    correct += (predicted == labels).sum().item()

print("The testing is done! " + str(correct) + "/" + str(total))


