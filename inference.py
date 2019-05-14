from os import listdir, mkdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import pickle

import torch

from conv_models.convnet import  ConvNet
import utils.image_utils as image_utils


def evaluate_performance():

    image_path = join('data', 'DRIVE', 'test', 'images')
    mask_path = join('data', 'DRIVE', 'test', 'mask')
    result_path = join('data', 'results', 'rf')

    images = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    patch_size = 2000

    vessel_detector = VesselDetectorRf(patch_size)

    for image in images:

        id = image[0:2]
        image_file = join(image_path, image)
        mask_file = join(mask_path, id + '_test_mask.gif')
        image_data = mimg.imread(image_file)[:, :, 1]
        mask_data = mimg.imread(mask_file)
        predicted_data = inference(image_data, mask_data, vessel_detector)
        plt.imsave(join(result_path, image), predicted_data, cmap='binary')
        # plt.figure()
        # plt.imshow(image_data, cmap='gray')
        # plt.figure()
        # plt.imshow(predicted_data, cmap='binary', interpolation='nearest')
        # plt.figure()
        # plt.imshow(mask_data, cmap='binary', interpolation='nearest')
        # plt.show()


def inference(image_data, mask_data, vessel_detector):

    indices = np.where(mask_data == 255)
    labels = vessel_detector.detect_vessels(image_data, indices)
    predicted = np.zeros_like(image_data)
    predicted[indices] = labels[:, 0]
    return predicted


class VesselDetector:

    def __init__(self, patch_size, convmodel):
        self.patch_size = patch_size
        self.net = ConvNet(1, 33, 67, 355)
        self.net.load_state_dict(torch.load(convmodel, map_location=lambda storage, loc: storage))

    def detect_vessels(self, image_data, indices):

        self.net.eval()
        labels = np.zeros((indices[0].size, 1), dtype=np.int32)
        for k in range(int(np.ceil(indices[0].size / self.patch_size))):
            start_ind = k * self.patch_size
            end_ind = (k + 1) * self.patch_size
            if end_ind > indices[0].size:
                end_ind = indices[0].size

            patch_indices = (indices[0][start_ind:end_ind], indices[1][start_ind:end_ind])
            patch_data = image_utils.get_clipped_area(image_data, patch_indices)
            patch_data = np.reshape(patch_data, [patch_data.shape[0], 1, patch_data.shape[1], patch_data.shape[2]])
            x = torch.tensor(patch_data, device='cpu', dtype=torch.float)
            probabilities = self.net(x)
            predicted = torch.round(probabilities.data)
            labels[start_ind:end_ind] = predicted.data.numpy()
            print('Processed  %f %%' % (100 * k / int(np.ceil(indices[0].size / self.patch_size))))

        return labels


class VesselDetectorRf:

    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.classifier = pickle.load(open('saved_models/random_forest.p', 'rb'))

    def detect_vessels(self, image_data, indices):

        labels = np.zeros((indices[0].size, 1), dtype=np.int32)
        for k in range(int(np.ceil(indices[0].size / self.patch_size))):
            start_ind = k * self.patch_size
            end_ind = (k + 1) * self.patch_size
            if end_ind > indices[0].size:
                end_ind = indices[0].size

            patch_indices = (indices[0][start_ind:end_ind], indices[1][start_ind:end_ind])
            patch_data = image_utils.get_clipped_area(image_data, patch_indices)
            patch_data = np.reshape(patch_data, [patch_data.shape[0], patch_data.shape[1] * patch_data.shape[2]])
            predicted = self.classifier.predict(patch_data)
            labels[start_ind:end_ind, 0] = predicted
            print('Processed  %f %%' % (100 * k / int(np.ceil(indices[0].size / self.patch_size))))

        return labels



if __name__ == '__main__':
    evaluate_performance()

