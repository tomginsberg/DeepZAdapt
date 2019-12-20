import torch
from torch import tensor
from torch.nn.parameter import Parameter
from zonotpe_utils import hypercube1d
from torch.functional import F


def box(x):
    """
    Compute box bounds for a Zonotope dimension x
    :param x: a torch tensor [x0, x1, x2, ...] representing the Zonotope dimension
    for all i, -1 <= eps_i <= 1, x = (x0 + eps_1 * x1 + eps_2 * x2 + ...)
    :return: (l, u) where l and u are box bounds
    """
    radius = torch.sum(torch.abs(x[1:]))
    return x[0] - radius, x[0] + radius


class ReLU2D(torch.nn.Module):
    def __init__(self, in_features, learnable=False):
        super(ReLU2D, self).__init__()
        self.in_features = in_features
        self.relu = ReLU(in_features, learnable=learnable)

    def forward(self, x):
        _, channels, n, _, = x.shape
        # x = self.relu(x.view(channels * n * n, errs))
        # return x.view(channels, n, n, x.shape[-1])
        x = self.relu(x.flatten(1))
        return x.view(x.shape[0], channels, n, n)


class ReLU(torch.nn.Module):
    def __init__(self, in_features, learnable=False):
        super(ReLU, self).__init__()
        self.lambdas = None
        self.learnable = learnable

    def forward(self, x):
        x = x.t()
        boxes = [box(l) for l in x]
        new_errors = sum([int(l < 0 < u) for l, u in boxes])
        _, epsilon_id = x.shape
        x = torch.nn.ZeroPad2d((0, new_errors))(x)
        new_layer = []
        if self.lambdas is None:
            # It appears that setting lambda to 0 for the edge case (abs(l)>>u) does not influence optimization much
            # See logs Min Area Box Learnable Lambdas
            # noinspection PyArgumentList
            self.lambdas = torch.nn.ParameterList(
                [Parameter((u / (u - l) + torch.randn(1) * .3 if l < 0 < u else torch.rand(1)).squeeze().clamp(0, 1),
                           requires_grad=self.learnable) for (l, u) in boxes])
        # noinspection PyTypeChecker
        for xi, (l, u), lambda_ in zip(x, boxes, self.lambdas):
            xi_new, epsilon_id = relu(xi, lambda_, l, u, epsilon_id)
            new_layer.append(xi_new)
        return torch.stack(new_layer).t()


def relu(x: torch.Tensor, lambda_: torch.nn.Parameter, l: torch.Tensor, u: torch.Tensor, epsilon_id: int):
    if u <= 0:
        return torch.zeros_like(x), epsilon_id
    elif l < 0:
        x = x.mul(lambda_)

        if lambda_ >= u / (u - l):
            val = -l * lambda_ / 2
        else:
            val = u * (1 - lambda_) / 2

        x[epsilon_id] += val
        x[0] += val
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
        return x.mm(self.weight.t()).index_add(0, tensor([0]), self.bias.unsqueeze(0))

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
        return x.index_add(0, tensor([0]), tensor([[-self.mean] * x.shape[1]])) / self.sigma


class PrepareInput1D(torch.nn.Module):

    def __init__(self, device, eps):
        super(PrepareInput1D, self).__init__()
        self.mean = torch.tensor(0.1307).to(device)
        self.sigma = torch.tensor(0.3081).to(device)
        self.eps = torch.tensor(eps).to(device)

    def forward(self, x):
        n = x.shape[-1]
        return hypercube1d(x.flatten(), self.eps).index_add(0, tensor([0]),
                                                            tensor([[-self.mean] * (n * n)])) / self.sigma


class Normalization2D(torch.nn.Module):

    def __init__(self, device):
        super(Normalization2D, self).__init__()
        self.mean = torch.tensor(0.1307).to(device)
        self.sigma = torch.tensor(0.3081).to(device)

    def forward(self, x, input_size=28):
        return (x.index_add(0, tensor([0]),
                            tensor([[-self.mean] * x.shape[1]])) / self.sigma).t().view(1,
                                                                                        input_size, input_size,
                                                                                        x.shape[0])


class PrepareInput2D(torch.nn.Module):

    def __init__(self, device, eps):
        super(PrepareInput2D, self).__init__()
        self.mean = torch.tensor(0.1307).to(device)
        self.sigma = torch.tensor(0.3081).to(device)
        self.eps = torch.tensor(eps).to(device)

    def forward(self, x):
        n = x.shape[-1]
        return (hypercube1d(x.flatten(), self.eps).index_add(0, tensor([0]),
                                                             tensor(
                                                                 [[-self.mean] * (n * n)])) / self.sigma).view(
            (n * n) + 1, 1, n, n)


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.flatten(1)


class FastConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FastConv2D, self).__init__()
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)
        self.bias = Parameter(torch.Tensor(out_channels), requires_grad=False)
        self.stride = stride
        self.padding = padding

    def forward(self, img):
        img = F.conv2d(img, self.weight, None, stride=self.stride, padding=self.padding)
        img[0] += self.bias.view(-1, 1, 1)
        return img
