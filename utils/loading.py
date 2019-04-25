import numpy as np
import random
from os.path import join


def load_samples(num_samples):

    mode = 'training'
    snapshots = np.load(join('data', 'training_samples', 'snapshots_' + mode + '_' + str(num_samples) + '.npy'))
    labels = np.load(join('data', 'training_samples', 'labels_' + mode + '_' + str(num_samples) + '.npy'))
    snapshots = np.reshape(snapshots, (snapshots.shape[0], 1, snapshots.shape[1], snapshots.shape[2]))
    snapshots = snapshots.astype(np.float32)

    random_indices = random.sample(range(0, labels.size), labels.size)
    x_train = snapshots[random_indices, :]
    y_train = labels.astype(np.int32)
    y_train = y_train[random_indices]
    y_train = np.reshape(y_train, (y_train.size, 1))

    mode = 'validation'
    snapshots = np.load(join('data', 'training_samples', 'snapshots_' + mode + '_' + str(num_samples) + '.npy'))
    labels = np.load(join('data', 'training_samples', 'labels_' + mode + '_' + str(num_samples) + '.npy'))
    snapshots = np.reshape(snapshots, (snapshots.shape[0], 1, snapshots.shape[1], snapshots.shape[2]))
    snapshots = snapshots.astype(np.float32)

    random_indices = random.sample(range(0, labels.size), labels.size)
    x_valid = snapshots[random_indices, :]
    y_valid = labels.astype(np.int32)
    y_valid = y_valid[random_indices]
    y_valid = np.reshape(y_valid, (y_valid.size, 1))

    mode = 'test'
    snapshots = np.load(join('data', 'training_samples', 'snapshots_' + mode + '_' + str(num_samples) + '.npy'))
    labels = np.load(join('data', 'training_samples', 'labels_' + mode + '_' + str(num_samples) + '.npy'))
    snapshots = np.reshape(snapshots, (snapshots.shape[0], 1, snapshots.shape[1], snapshots.shape[2]))
    snapshots = snapshots.astype(np.float32)

    random_indices = random.sample(range(0, labels.size), labels.size)
    x_test = snapshots[random_indices, :]
    y_test = labels.astype(np.int32)
    y_test = y_test[random_indices]
    y_test = np.reshape(y_test, (y_test.size, 1))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_visualization_samples(num_samples):

    mode = 'training'
    snapshots = np.load(join('..', 'data', 'training_samples', 'snapshots_' + mode + '_' + str(num_samples) + '.npy'))
    labels = np.load(join('..', 'data', 'training_samples', 'labels_' + mode + '_' + str(num_samples) + '.npy'))
    snapshots = np.reshape(snapshots, (snapshots.shape[0], 1, snapshots.shape[1], snapshots.shape[2]))
    snapshots = snapshots.astype(np.float32)


    return (snapshots, labels)
