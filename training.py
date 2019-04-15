from os.path import join
import time
import numpy as np
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
from utils.regularization import EarlyStopping
from utils.loss_utils import compute_loss, compute_accuracy
from utils.parameter_search import random_search


def train(num_epochs, learning_rate, model_params, batch_size, x_train, y_train, x_valid, y_valid, x_test, y_test, device='cpu', save_model=False):

    net = ConvNet(1, model_params[0], model_params[1], model_params[2])
    net.to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    val_errors = []
    val_losses = []

    early_stop = EarlyStopping(tolerance=0.03, patience=4)

    for epoch in range(num_epochs):
        start_time = time.time()
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
        validation_accuracy = compute_accuracy(net, x_valid, y_valid)
        val_errors.append(validation_accuracy)
        validation_loss = compute_loss(net, x_valid, y_valid)
        val_losses.append(validation_loss)
        time_taken = (time.time() - start_time)
        print('Accuracy of the network on epoch %d %%' % epoch + ': %f %%' % (100 * validation_accuracy) +
              ' loss: %f ' % validation_loss + ' took %f' %time_taken + 'seconds')
        if early_stop.stop_criterion(val_losses):
            print('Stop after %d epochs' % epoch)
            break

    if save_model:
        save_filename = join('saved_models',
                             'convnet_' + model_params[0] + 'x' + model_params[1] + 'x' + model_params[2] + '.pth')
        torch.save(net.state_dict(), save_filename)

    test_accuracy = compute_accuracy(net, x_test, y_test)
    print('Final test accuracy: %f %%' % (100 * test_accuracy))
    return test_accuracy


def optimize_hyper_parameters(num_epochs, learning_rate, num_combinations, batch_size, x_train, y_train, x_valid, y_valid, x_test, y_test):

    range1 = [10, 35]
    range2 = [35, 70]
    range3 = [200, 1200]

    parameter_combinations = random_search(num_combinations, range1, range2, range3)
    hyper_parameters = []
    accuracies = []

    for n1, n2, n3 in parameter_combinations:
        accuracy = train(num_epochs, learning_rate, [n1, n2, n3], batch_size, x_train, y_train, x_valid, y_valid, x_test, y_test)
        hyper_parameters.append([n1, n2, n3])
        accuracies.append(accuracy)
        print('accuracy with parameters [' + str(n1) + ', ' + str(n2) + ', ' + str(n3) + '] ' + str(accuracy))

    hyper_parameters = np.array(hyper_parameters)
    accuracies = np.array(accuracies)
    np.savez(join('saved_models', 'hyperparameters1.npz'),
             hyperparameters=hyper_parameters,
             accuracies=accuracies)

    ix = accuracies.argsort()[-1::-1]

    print(accuracies[ix])
    print(hyper_parameters[ix, :])


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    parser.add_argument('--optimize_hyperparameters', default=False, type=bool)
    parser.add_argument('--save_model', default=False, type=bool)
    parser.add_argument('--num_combinations', default=10, type=int)
    args = parser.parse_args(args)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hyper_parameter_optimization = args.optimize_hyperparameters
    num_combinations = args.num_combinations
    save_model = args.save_model

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_samples(2000)

    if hyper_parameter_optimization:
        optimize_hyper_parameters(num_epochs, learning_rate, num_combinations, batch_size,
                                  x_train, y_train, x_valid, y_valid, x_test, y_test)
    else:
        train(num_epochs, learning_rate, [32, 64, 1024], batch_size, x_train, y_train, x_valid,
              y_valid, x_test, y_test, device='cpu', save_model=save_model)


if __name__ == '__main__':
    main(sys.argv[1:])
