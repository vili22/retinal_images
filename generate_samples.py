from os.path import join, isdir
from os import mkdir
from utils import image_utils
import numpy as np

mode ='test'
path = join('data', 'DRIVE', mode)

N_samples = 2000
[training_data, labels] = image_utils.read_training_data(path, N_samples, mode)

print(training_data.shape)
print(labels.shape)

if not isdir(join('data', 'training_samples')):
    mkdir(join('data', 'training_samples'))

np.save(join('data', 'training_samples', 'snapshots_' + mode + '_' + str(N_samples) + '.npy'), training_data)
np.save(join('data', 'training_samples', 'labels_' + mode + '_' + str(N_samples) + '.npy'), labels)