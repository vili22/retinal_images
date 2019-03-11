import numpy as np
import random
from os.path import join


def load_samples():

    snapshots = np.load(join('data', 'training_samples', 'snapshots_2000.npy'))
    labels = np.load(join('data', 'training_samples', 'labels_2000.npy'))
    snapshots = np.reshape(snapshots, (snapshots.shape[0], snapshots.shape[1], snapshots.shape[2], 1))
    snapshots = snapshots.astype(np.float32)

    random_indices = random.sample(range(0, labels.size), labels.size)
    snapshots = snapshots[random_indices, :]
    labels = labels.astype(np.int32)
    labels = labels[random_indices]
    labels = np.reshape(labels, (labels.size, 1))

    num_training_samples = int(0.75 * labels.size)
    x_training = snapshots[0:num_training_samples, :]
    y_training = labels[0: num_training_samples, :]

    x_valid = snapshots[num_training_samples:, :]
    y_valid = labels[num_training_samples:, :]

    return (x_training, y_training), (x_valid, y_valid)