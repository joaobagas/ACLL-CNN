import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms

from network import model_loader
from network.dataset_creator import load_labels
from network.module import Net

net, current_epoch, loss = model_loader.load()

transform = transforms.ToTensor()
test_path = "datasets/acll/test"
test_set = datasets.ImageFolder(test_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

correct, total, count = 0, 0, 0
predictions = []
net.eval()

for i, data in enumerate(test_loader, 0):
    inputs, _ = data
    labels = load_labels(test_path)
    labels = torch.Tensor(labels)

    outputs = net.forward(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    print(count)
    x1, y1, z1, w1 = predictions[i].tolist()[0]
    x2, y2, z2, w2 = labels.tolist()[count]
    if (x1 > 0) == (x2 > 0):
        correct += 1
    if (y1 > 0) == (y2 > 0):
        correct += 1
    if (z1 > 0) == (z2 > 0):
        correct += 1
    if (w1 > 0) == (w2 > 0):
        correct += 1
    count += 1

print("The testing is done! " + str(correct) + "/" + str(200))