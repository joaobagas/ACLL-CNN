from numpy import sqrt, log


def compute_score(array1, array2):
    print(array1.detach().numpy()[0])
    print(array2)
    x1, y1, z1, w1 = array1.detach().numpy()[0]
    x2, y2, z2, w2 = array2
    if 0.2 <= x1 <= 5 and 0.2 <= y1 <= 5 and 0.2 <= z1 <= 5 and 0.2 <= w1 <= 5:
        x = log(x1) - log(x2)
        y = log(y1) - log(y2)
        z = log(z1) - log(z2)
        w = log(w1) - log(w2)
        score = (2.8 - sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(w, 2))) / 2.8
    else:
        score = 0
    return score
