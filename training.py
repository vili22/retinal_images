import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse


def train(num_epochs, learning_rate, batch_size, x_train, y_train, x_valid, y_valid, device='cpu'):

    net = None
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # In[ ]:

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for k in range(int(y_train.size / batch_size) + 1):
            start_ind = k * batch_size
            end_ind = (k + 1) * batch_size if (k + 1) * batch_size < y_train.size else y_train.size
            x = torch.tensor(x_train[start_ind:end_ind, :], device=device, dtype=torch.float)
            y = torch.tensor(y_train[start_ind:end_ind, :], device=device, dtype=torch.int32)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Print accuracy after every epoch
        accuracy = compute_accuracy(net, x_valid, y_valid)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))


def compute_accuracy(net, x_valid, y_valid, device='cpu'):
    net.eval()
    correct = 0
    total = 0
    test_batch_size = 1000
    with torch.no_grad():
        for k in range(0, int(y_valid.size / test_batch_size)):
            start_ind = k * 1000
            end_ind = (k + 1) * test_batch_size if (k + 1) * test_batch_size < y_valid.size else y_valid.size
            x = torch.tensor(x_valid[start_ind:end_ind, :], device=device, dtype=torch.float)
            y = torch.tensor(y_valid[start_ind:end_ind, :], device=device, dtype=torch.int32)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_valid).sum().item()
    return correct / total


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    args = parser.parse_args(args)
    print(args.num_epochs)


if __name__ == '__main__':
    main(sys.argv[1:])
