from torch import nn, cat, eye


class Hypercube(nn.Module):
    def __init__(self, eps):
        super(Hypercube).__init__()
        self.eps = eps

    def forward(self, x):
        return cat((x, self.eps * eye(x.shape[1])))
