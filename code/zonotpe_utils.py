import torch
from torch import cat, Tensor, zeros
from math import floor


def hypercube1d(x: Tensor, eps):
    # x must be a (n) tensor
    # output is (n + 1 x n) tensor
    # E.x [x0, x1, x2] -> [[x0, x1, x2], [eps, 0, 0], [0, eps, 0], [0, 0, eps]]
    n = x.shape[0]
    eye = torch.eye(n)
    return torch.stack([shift_zonotope(x[i], eps, eye[i]) for i in range(n)]).t()
    # non adaptable implementation
    # cat((x,torch.eye(x.shape[1])))


def shift_zonotope(x, eps, row):
    """
    Fixes the epsilon problem in the way proposed by Tom
    :param x:
    :param eps:
    :param row:
    :return:
    """
    if x + eps > 1:
        x, eps = (1 + x - eps) / 2, (1 - x + eps) / 2
    elif x - eps < 0:
        x = (x + eps) / 2
        eps = x

    return torch.cat((x.unsqueeze(0), eps * row))


# Sadly this is not used anymore due to the normalization strategy
def hypercube2d(x: Tensor, eps):
    # Lift an image x into hypercube with norm eps
    # x is a (n x n) dimensional tensor
    # E.x.  x =  [ [[x0  x1]
    #               [x2  x3]] ]
    # output:    [ [ [[x0 eps 0 0 0], [x0 0 eps 0 0]]
    #                [[x0 0 0 eps 0], [x0 0 0 0 eps]] ] ]
    # output.shape is a (1 x n x n x n + 1)

    n = x.shape[-1]
    eye = torch.eye(n * n)
    return torch.stack(
        [torch.stack([shift_zonotope(x[j, i], eps, eye[i + j * n]) for i in range(n)]) for j in range(n)]).view(
        1, n, n, n ** 2 + 1)


def box_upper(x):
    return x[0] + torch.sum(torch.abs(x[1:]))


if __name__ == '__main__':
    from transformers import box
    import networks
    print(hypercube2d(torch.tensor([.9] * 4).view(2, 2), .2))
