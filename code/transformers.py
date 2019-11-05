import torch
from torch import tensor
from torch.nn.parameter import Parameter


def box(x):
    """
    Compute box bounds for a Zonotope dimension x
    :param x: a torch tensor [x0, x1, x2, ...] representing the Zonotope dimension
    for all i, -1 <= eps_i <= 1, x = (x0 + eps_1 * x1 + eps_2 * x2 + ...)
    :return: (l, u) where l and u are box bounds
    """
    # x = x.clone()
    radius = torch.sum(torch.abs(x[1:]))
    return x[0] - radius, x[0] + radius


class ReLU(torch.nn.Module):
    def __init__(self, in_features):
        super(ReLU, self).__init__()
        # Maybe random lambdas is not the best to do
        self.lambdas = Parameter(torch.rand(in_features))

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        boxes = [box(i) for i in x]
        new_errors = sum([int(l < 0 < u) for l, u in boxes])
        _, epsilon_id = x.shape
        x = torch.nn.ZeroPad2d((0, new_errors))(x)
        new_layer = []

        for xi, (l, u), lmb in zip(x, boxes, self.lambdas):
            xi_new, epsilon_id = relu(xi, lmb, l, u, epsilon_id)
            new_layer.append(xi_new)

        return torch.transpose(torch.stack(new_layer), 0, 1)


def relu(x, lmb, l, u, epsilon_id):
    if u <= 0:
        x = x * 0
    elif l < 0:
        x = x * lmb

        if lmb >= u / (u - 1):
            x[epsilon_id] = -l * lmb / 2
        else:
            x[epsilon_id] = u * (1 - lmb)

        x[0] = x[0] + x[epsilon_id]
        epsilon_id += 1
    return x, epsilon_id


class Affine(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Affine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # These tensors are automatically allocated with garbage values and will be loaded from a state_dict
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.bias = Parameter(torch.Tensor(out_features), requires_grad=False)

    def forward(self, x):
        # In PyTorch documentation nn.Linear is defined as x.W^(T) + b
        x = x.mm(self.weight.t())
        x[0] = x[0] + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class Normalization(torch.nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(0.1307).to(device)
        self.sigma = torch.tensor(0.3081).to(device)

    def forward(self, x):
        # PyTorch doesnt like in place operations on variables with gradients
        # (i.e use x = x + 1 vs x += 1)
        x[0] = x[0] - self.mean
        return x / self.sigma


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # Hardcoded for MNIST so we can catch any bugs is this breaks
        return torch.reshape(x, (x.shape[1], 784))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np


    def optimize(attempts, steps_per_attempt, weights, biases, learning_rate=0.005):
        loss_history = np.empty((attempts, steps_per_attempt))
        num_layers = weights.shape[0]
        # Initialize weights and biases
        x0, y0, eta = 1 / 8, 1 / 8, .1
        for attempt in range(attempts):
            # Initialize lambdas
            lambdas = torch.rand(num_layers, 2, requires_grad=True)
            # Gradient descent loop
            for step in range(steps_per_attempt):
                # First layer of network
                layer = torch.transpose(tensor([[x0, eta, 0], [y0, 0, eta]]), 0, 1).requires_grad_()
                # apply network layers
                for w, b, l in zip(weights, biases, lambdas):
                    layer = torch.matmul(w, layer)
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
