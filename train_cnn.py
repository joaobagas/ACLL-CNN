import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms

from network import model_loader
from network.dataset_creator import load_labels
from network.module import Net

BATCH_SIZE = 5
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50

# net = Net()
# current_epoch = 0
net, current_epoch, loss = model_loader.load()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

transform = transforms.ToTensor()
train_path = "datasets/acll/train"
train_set = datasets.ImageFolder(train_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=False)

net.train()

print("Training...")

for e in range(NUM_EPOCHS):
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, _ = data
        labels = load_labels(train_path)

        # Forward + Backward + Optimize
        try:
            outputs = net.forward(inputs)
            loss = criterion(outputs, torch.Tensor(labels[count:count + BATCH_SIZE]))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            count += BATCH_SIZE
        except:
            print("There was an error!")
    current_epoch += 1
    print("Finished epoch {}!".format(current_epoch))

print("Finished training, saving...")
model_loader.save(net, optimizer, current_epoch, loss)
print("Saved!")



