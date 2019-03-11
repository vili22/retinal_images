from os.path import join, isdir
from os import mkdir
from utils import image_utils
import numpy as np

path = join('data', 'DRIVE', 'training')

N_samples = 2000
[training_data, labels] = image_utils.read_training_data(path, N_samples)

print(training_data.shape)
print(labels.shape)

if not isdir(join('data', 'training_samples')):
    mkdir(join('data', 'training_samples'))

np.save(join('data', 'training_samples', 'snapshots_' + str(N_samples) + '.npy'), training_data)
np.save(join('data', 'training_samples', 'labels_' + str(N_samples) + '.npy'), labels)