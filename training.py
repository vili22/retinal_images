from os.path import join
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse

from conv_models.convnet import ConvNet
from utils.loading import load_samples
from utils.regularization import EarlyStopping, compute_loss


def train(num_epochs, learning_rate, batch_size, x_train, y_train, x_valid, y_valid, device='cpu'):

    net = ConvNet(1, 32, 64, 1024)
    net.to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_errors = []  # Keep track of the training error
    val_errors = []
    save_filename = join('saved_models', 'convnet_' + str(32) + 'x' + str(64) + 'x' + str(1024) + '.pth')
    print(save_filename)

    for epoch in range(num_epochs):
        for k in range(int(y_train.size / batch_size) - 1):
            start_ind = k * batch_size
            end_ind = (k + 1) * batch_size if (k + 1) * batch_size < y_train.size else y_train.size
            x = torch.tensor(x_train[start_ind:end_ind, :], device=device, dtype=torch.float)
            y = torch.tensor(y_train[start_ind:end_ind, :], device=device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Print accuracy after every epoch
        accuracy = compute_accuracy(net, x_valid, y_valid)
        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * accuracy))
    torch.save(net.state_dict(), join('saved_models', 'convnet_' + str(32) + 'x' + str(64) + 'x' + str(1024) + '.pth'))

def compute_accuracy(net, x_valid, y_valid, device='cpu'):
    net.eval()
    correct = 0
    total = y_valid.size
    test_batch_size = 1000
    with torch.no_grad():
        for k in range(0, int(y_valid.size / test_batch_size) - 1):
            start_ind = k * 1000
            end_ind = (k + 1) * test_batch_size if (k + 1) * test_batch_size < y_valid.size else y_valid.size
            x = torch.tensor(x_valid[start_ind:end_ind, :], device=device, dtype=torch.float)
            y = torch.tensor(y_valid[start_ind:end_ind, :], device=device, dtype=torch.float)
            outputs = net(x)
            predicted = torch.round(outputs.data)
            correct += (predicted == y).sum().item()
    return correct / total


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    args = parser.parse_args(args)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_samples(2000)
    train(num_epochs, learning_rate, batch_size, x_train, y_train, x_valid, y_valid, device='cpu')


if __name__ == '__main__':
    main(sys.argv[1:])
