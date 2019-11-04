import torch
from torch import tensor, matmul


def box(x):
    """
    Compute box bounds for a Zonotope dimension x
    :param x: a torch tensor [x0, x1, x2, ...] representing the Zonotope dimension
    for all i, -1 <= eps_i <= 1, x = (x0 + eps_1 * x1 + eps_2 * x2 + ...)
    :return: (l, u) where l and u are box bounds
    """
    radius = torch.sum(torch.abs(x[1:]))
    return x[0] - radius, x[0] + radius


class ReLU(torch.nn.Module):
    def __init__(self, lambdas):
        super(ReLU, self).__init__()
        self.lambdas = lambdas

    def forward(self, layer):
        boxes = [box(n) for n in layer]
        _, epsilon_id = layer.shape
        for i, (l, u), lmb in zip(range(layer.shape[0]), boxes, self.lambdas):
            if u <= 0:
                # FixMe
                layer[i] *= 0
            elif l < 0:
                layer = torch.nn.ConstantPad2d((0, 1), 0)(layer)
                layer[i] *= lmb

                if lmb >= u / (u - 1):
                    layer[i][epsilon_id] = -l * lmb / 2
                else:
                    layer[i][epsilon_id] = u * (1 - lmb)

                layer[i][0] += layer[i][epsilon_id]
                epsilon_id += 1

        return layer


def bias(layer, bias):
    """
    Adds a bias vector to the center of a zonotope without affecting the noise terms
    :param layer:
    :param bias:
    :return:
    """
    # A different and probably worse approach is to pad right and matrix add
    # return layer + torch.nn.ConstantPad2d((0, layer.shape[1] - 1),0)(b)
    for n, b in zip(layer, bias):
        n[0] += b[0]
    return layer


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np


    def optimize(attempts, steps_per_attempt, weights, biases, learning_rate=0.005):
        loss_history = np.empty((attempts, steps_per_attempt))
        num_layers = weights.shape[0]
        # Initialize weights and biases
        x0, y0, eta = 1 / 4, 1 / 5, .1
        for attempt in range(attempts):
            # Initialize lambdas
            lambdas = torch.rand(num_layers, 2, requires_grad=True)
            # Gradient descent loop
            for step in range(steps_per_attempt):
                # First layer of network
                layer = tensor([[x0, eta, 0], [y0, 0, eta]], requires_grad=True)
                # apply network layers
                for w, b, l in zip(weights, biases, lambdas):
                    layer = matmul(w, layer)
                    for n, b_ in zip(layer, b):
                        n[0] += b_[0]
                    layer = ReLU(l)(layer)

                # Compute loss and update gradients
                loss = box(layer[1] - layer[0])[1]
                loss_history[attempt][step] = loss
                loss.backward()
                if lambdas.grad is not None:
                    lambdas = torch.clamp(lambdas - learning_rate * lambdas.grad, 0,
                                          1).clone().detach_().requires_grad_(True)
        return loss_history


    num_layers = 20
    weights = torch.randn(num_layers, 2, 2)
    biases = torch.randn(num_layers, 2, 1) / 4
    with torch.autograd.set_detect_anomaly(True):
        plt.plot(np.transpose(optimize(5, 10, weights, biases, learning_rate=.05)))
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.show()
