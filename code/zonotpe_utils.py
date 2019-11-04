from torch import cat, eye, Tensor, zeros, transpose
from math import floor


def hypercube1d(x: Tensor, eps):
    # x must be a 1D tensor
    return cat((x.expand(1, x.shape[0]), eps * eye(x.shape[0])))


def hypercube2d(x: Tensor, eps):
    # Lift an image x into hypercube with norm eps
    # x is a (1 x 1 x n x n) dimensional tensor
    n = x.shape[-1]
    z = zeros(1, n ** 2, n, n)
    j = 0
    for i in range(n ** 2):
        # What a shit show this is
        z[0, i, floor(j), i % n] = eps
        j += 1 / n
    return cat((x, z), 1)
