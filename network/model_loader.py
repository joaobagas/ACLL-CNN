import torch

from network.module import Net


def save():
    EPOCH = 5
    PATH = "model.pt"
    LOSS = 0.4

    torch.save({
        'epoch': EPOCH,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
    }, PATH)


def load():
    model = Net()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']