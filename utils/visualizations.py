import matplotlib.pyplot as plt
import random
import numpy as np

from utils.loading import load_visualization_samples


def visualize_samples():

    x, y = load_visualization_samples(2000)
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]


    num_examples = 3
    random_positive_indices = random.sample(range(0, positive_indices.size), num_examples)
    random_negative_indices = random.sample(range(0, negative_indices.size), num_examples)

    for k in range(0, num_examples):
        plt.figure()
        plt.imshow(x[positive_indices[random_positive_indices[k]], 0, :, :], cmap='gray')
        plt.figure()
        plt.imshow(x[negative_indices[random_negative_indices[k]], 0, :, :], cmap='gray')

    plt.show()


def visualize_pseudoparameter_accuracies():

    hyperparameters = np.zeros((0, 3), dtype=int)
    accuracies = np.zeros((0, ), dtype=float)

    stored = np.load('../saved_models/hyperparameters1.npz')
    hyperparameters = np.row_stack((hyperparameters, stored['hyperparameters']))
    accuracies = np.append(accuracies, stored['accuracies'])

    stored = np.load('../saved_models/hyperparameters2.npz')
    hyperparameters = np.row_stack((hyperparameters, stored['hyperparameters']))
    accuracies = np.append(accuracies, stored['accuracies'])

    stored = np.load('../saved_models/hyperparameters3.npz')
    hyperparameters = np.row_stack((hyperparameters, stored['hyperparameters']))
    accuracies = np.append(accuracies, stored['accuracies'])

    stored = np.load('../saved_models/hyperparameters4.npz')
    hyperparameters = np.row_stack((hyperparameters, stored['hyperparameters']))
    accuracies = np.append(accuracies, stored['accuracies'])

    ix = accuracies.argsort()[-1::-1]

    print(accuracies[ix])
    print(hyperparameters[ix, :])




if __name__ == '__main__':
    #visualize_samples()
    visualize_pseudoparameter_accuracies()