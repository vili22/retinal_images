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


if __name__ == '__main__':
    visualize_samples()