import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import transformers


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnectedVerifiable(nn.Module):
    def __init__(self, input_size, fc_layers):
        super(FullyConnectedVerifiable, self).__init__()
        # We put the identity here because I need to remove the Flatten layer
        # but keep the state dict entries consistent
        layers = [nn.Identity(), nn.Identity()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [transformers.Affine(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [transformers.ReLU(fc_size, learnable=True)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load_state_dict(self, state_dict, **kwargs):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class ConvVerifiable(nn.Module):

    def __init__(self, input_size, conv_layers, fc_layers, n_class=10):
        super(ConvVerifiable, self).__init__()

        self.n_class = n_class

        layers = [nn.Identity()]
        prev_channels = 1
        img_dim = input_size

        for i, (n_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            layers += [
                transformers.FastConv2D(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                transformers.ReLU2D(img_dim * img_dim * n_channels, learnable=(len(conv_layers) == 1 or i > 0)),
            ]
            prev_channels = n_channels
            img_dim = (img_dim + 2 * padding - kernel_size) // stride + 1
        layers += [transformers.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [transformers.Affine(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [transformers.ReLU(fc_size, learnable=True)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load_state_dict(self, state_dict, **kwargs):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class FullyConnected(nn.Module):
    # Todo: Add ReLU tracker
    def __init__(self, device, input_size, fc_layers):
        super(FullyConnected, self).__init__()
        self.architecture = 'fcn'
        self.fc_layers = fc_layers
        layers = [Normalization(device), torch.nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [torch.nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [torch.nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):
    # Todo: Add ReLU tracker
    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.architecture = 'cnn'
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UnitClipper:
    def __call__(self, module):
        if hasattr(module, 'lambdas'):
            lambdas = module.lambdas
            for lbda in lambdas:
                lbda.data.clamp_(0, 1)


def filter_state_dict(net, keyword='lambdas'):
    return {key: val for key, val in net.state_dict().items() if keyword in key}
