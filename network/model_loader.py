import torch
from torch import optim

from network.module import Net


def save(net, optimizer, epoch, loss):
    PATH = "checkpoint/model{}.pt".format(epoch)

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)


def load():
    model = Net()
    PATH = "checkpoint/model1000.pt"

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, epoch, loss