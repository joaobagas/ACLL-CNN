from numpy import sqrt, log
from torch.nn import NLLLoss


def compute_score(array1, array2):
    x1, y1, z1, w1 = array1.detach().numpy()[0]
    x2, y2, z2, w2 = array2
    if 0.2 <= x1 <= 5 and 0.2 <= y1 <= 5 and 0.2 <= z1 <= 5 and 0.2 <= w1 <= 5:
        x = log(x1) - log(x2)
        y = log(y1) - log(y2)
        z = log(z1) - log(z2)
        w = log(w1) - log(w2)
        score = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(w, 2)) / 2.8
    else:
        score = 1
    return score

def get_loss(array1, array2):
    criterion = NLLLoss()
    x1, y1, z1, w1 = array1.detach().numpy()[0]
    x2, y2, z2, w2 = array2
    loss = criterion(x1, x2)
    loss += criterion(y1, y2)
    loss += criterion(z1, z2)
    loss += criterion(w1, w2)
    return loss
