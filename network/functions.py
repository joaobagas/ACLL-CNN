import torch
from torch import nn, optim
from torchvision import datasets, transforms

from network import model_loader
from network.dataset_creator import load_labels
from network.module import Net


def train(load_model, batch_size, num_epochs, learning_rate, min_loss, train_path):
    if load_model:
        net, current_epoch, loss = model_loader.load()
    else:
        net, current_epoch = Net(), 0
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001) # 3e-4

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((200, 200))])
    train_set = datasets.ImageFolder(train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    net.train()

    print("Training...")

    for e in range(num_epochs):
        count = 0
        losses = 0
        for i, data in enumerate(train_loader, 0):
            inputs, _ = data
            labels = load_labels(train_path)

            # Forward + Backward + Optimize
            try:
                outputs = net.forward(inputs)
                loss = criterion(outputs, torch.Tensor(labels[count:count + batch_size]))
                losses += loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                count += batch_size
            except:
                print("There was an error!")
        current_epoch += 1
        avg_loss = losses / count
        if avg_loss < min_loss:
            model_loader.save(net, optimizer, current_epoch, loss)
        print("Finished epoch {} Loss = {}!".format(current_epoch, avg_loss))

    print("Finished training, saving...")
    model_loader.save(net, optimizer, current_epoch, loss)
    print("Saved!")

def over_fit_single_batch(load_model, batch_size, num_epochs, test_every_epoch, train_path):
    if load_model is True:
        net, current_epoch, loss = model_loader.load()
    else:
        net, current_epoch = Net(), 0
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0.001) # 3e-4

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((200, 200))])
    train_set = datasets.ImageFolder(train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    net.train()

    print("Training...")

    data, _ = next(iter(train_loader))

    for e in range(num_epochs):
        inputs = data
        labels = load_labels(train_path)

        # Forward + Backward + Optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, torch.Tensor(labels[0:batch_size]))
        try:
            outputs = net.forward(inputs)
            loss = criterion(outputs, torch.Tensor(labels[0:batch_size]))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        except:
            print("There was an error!")
        current_epoch += 1
        print("Finished epoch {}! Loss:{}!".format(current_epoch, loss))

    print("Finished training, saving...")
    model_loader.save(net, optimizer, current_epoch, loss)
    print("Saved!")

def test(net, test_path):
    if net is None:
        net, current_epoch, loss = model_loader.load()
    criterion = nn.MSELoss()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((200, 200))])
    test_set = datasets.ImageFolder(test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    correct, total, count = 0, 0, 0
    predictions = []
    net.eval()
    losses = 0
    br, co, sa = 0, 0, 0

    print("Testing...")
    for i, data in enumerate(test_loader, 0):
        inputs, _ = data
        labels = load_labels(test_path)
        labels = torch.Tensor(labels)

        outputs = net.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(outputs)
        total += labels.size(0)
        losses += criterion(outputs[0], torch.Tensor(labels[count]))
        x1, y1, z1 = predictions[i].tolist()[0]
        x2, y2, z2 = labels.tolist()[count]
        if (x1 > 0) == (x2 > 0):
            br += 1
        if (y1 > 0) == (y2 > 0):
            co += 1
        if (z1 > 0) == (z2 > 0):
            sa += 1
        count += 1

    br = int((br / count) * 100)
    co = int((co / count) * 100)
    sa = int((sa / count) * 100)
    loss = losses/count
    print("Finished testing:\n - Brightness = {}%\n - Contrast = {}%\n - Saturation = {}%\n - Loss = {}".format(br, co, sa, loss))
