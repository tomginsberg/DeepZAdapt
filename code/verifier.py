import argparse
from sys import argv
import torch
from networks import FullyConnected, Conv, FullyConnectedVerifiable, ConvVerifiable, UnitClipper, filter_state_dict
from zonotpe_utils import hypercube1d, hypercube2d, box_upper
from time import time
from transformers import Normalization, Normalization2D, PrepareInput1D, PrepareInput2D
from glob import glob

start_time = time()
DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int):
    clipper = UnitClipper()
    if net.architecture == 'fcn':
        # Fully connected
        inputs = PrepareInput1D(DEVICE, eps)(inputs)
        verify_net = FullyConnectedVerifiable(INPUT_SIZE, net.fc_layers).train()
    else:
        # CNN
        inputs = PrepareInput2D(DEVICE, eps)(inputs)
        verify_net = ConvVerifiable(INPUT_SIZE, net.conv_layers, net.fc_layers).train()

    verify_net.load_state_dict(net.state_dict())

    # Run a forward pass once before instantiating optimizer to dynamically create learnable param.
    verify_net(inputs)

    optim = torch.optim.Adamax(verify_net.parameters(), lr=.5)

    while True:
        optim.zero_grad()

        outputs = verify_net(inputs).t()

        # Compute how much bigger every label is be from the true label in the worst case
        losses = torch.stack(
            [box_upper(outputs[i] - outputs[true_label]) for i in range(len(outputs)) if i != true_label])

        if (losses < 0).all():
            return True

        loss = torch.sum(torch.relu(losses))

        loss.backward()

        optim.step()
        verify_net.apply(clipper)


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')

    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()
    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)), )

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    try:
        # If there are no learnable params, calling backwards() will throw an exception, which means initial bounds are not verifiable and cannot be optimized further
        if analyze(net, inputs, eps, true_label):
            print('verified')
        else:
            print('not verified')
    except:
        print('not verified')


if __name__ == '__main__':
    main()
