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
    train_losses = []
    val_errors = []
    val_losses = []

    early_stop = EarlyStopping(tolerance=0.01, patience=2)

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0
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
            epoch_loss += np.asscalar(loss.cpu().data.numpy())

            loss.backward()
            optimizer.step()

        # Print accuracy after every epoch
        train_losses.append(epoch_loss)
        validation_accuracy = compute_accuracy(net, x_valid, y_valid)
        val_errors.append(validation_accuracy)
        validation_loss = compute_loss(net, x_valid, y_valid)
        val_losses.append(validation_loss)
        time_taken = (time.time() - start_time)
        print('Accuracy of the network on epoch %d %%' % epoch + ': %f %%' % (100 * validation_accuracy) +
              'validation loss: %f ' % validation_loss + 'train loss: %f ' % epoch_loss + ' took %f' %time_taken + 'seconds')
        if early_stop.stop_criterion(val_losses):
            print('Stop after %d epochs' % epoch)
            break

    test_accuracy = compute_accuracy(net, x_test, y_test)
    if save_model:
        save_filename = join('saved_models',
                             'convnet_' + str(model_params[0]) + 'x' + str(model_params[1]) + 'x' + str(model_params[2]) + '.pth')
        torch.save(net.state_dict(), save_filename)
        np.savez(join('saved_models', 'training_losses.npz'),
                 train_losses=train_losses,
                 val_losses=val_losses,
                 final_accuracy=test_accuracy)

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
    np.savez(join('saved_models', 'hyperparameters4.npz'),
             hyperparameters=hyper_parameters,
             accuracies=accuracies)

    ix = accuracies.argsort()[-1::-1]

    print(accuracies[ix])
    print(hyper_parameters[ix, :])


def train_random_forets_classifier(x_train, y_train, x_test, y_test, save_model=False):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pickle

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2] * x_train.shape[3]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2] * x_test.shape[3]))
    y_train = np.reshape(y_train, (y_train.size, ))
    y_test = np.reshape(y_test, (y_test.size, ))
    classifier = RandomForestClassifier(n_estimators=200, verbose=1, n_jobs=-1)
    classifier.fit(x_train, y_train)
    pred_test = classifier.predict(x_test)  # Predict labels of test data using the trained classifier
    rf_accuracy = accuracy_score(y_test, pred_test)
    print("Accuracy of random forest: {:.2f}".format(rf_accuracy))

    if save_model:
        pickle.dump(classifier, open('saved_models/random_forest.p', 'wb'))

    classifier = pickle.load(open('saved_models/random_forest.p', 'rb'))
    pred_test = classifier.predict(x_test)  # Predict labels of test data using the trained classifier
    rf_accuracy = accuracy_score(y_test, pred_test)
    print("Accuracy of random forest: {:.2f}".format(rf_accuracy))


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    parser.add_argument('--optimize_hyperparameters', default=False, type=bool)
    parser.add_argument('--save_model', default=False, type=bool)
    parser.add_argument('--num_combinations', default=10, type=int)
    parser.add_argument('--use_random_forest', default=False, type=bool)
    args = parser.parse_args(args)

    use_random_forest = args.use_random_forest
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hyper_parameter_optimization = False#args.optimize_hyperparameters
    num_combinations = args.num_combinations
    save_model = args.save_model

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_samples(2000)

    if use_random_forest:
        train_random_forets_classifier(x_train, y_train, x_test, y_test, save_model)
    elif hyper_parameter_optimization:
        optimize_hyper_parameters(num_epochs, learning_rate, num_combinations, batch_size,
                                  x_train, y_train, x_valid, y_valid, x_test, y_test)
    else:
        train(num_epochs, learning_rate, [33, 67, 355], batch_size, x_train, y_train, x_valid,
              y_valid, x_test, y_test, device='cpu', save_model=save_model)


if __name__ == '__main__':
    main(sys.argv[1:])
