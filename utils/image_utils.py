from os import listdir, mkdir
from os.path import isfile, isdir, join
from scipy import misc
import numpy as np
from scipy.misc import imsave


def read_image_file(filename):

        imagedata = misc.imread(filename, mode='RGB')
        return imagedata


def get_clipped_area(image_data, indices, dim=None):

        height = image_data.shape[0]
        width = image_data.shape[1]

        dest_size = 33
        center_index = 16
        if not dim:
            clip_size = int((dest_size - 1) / 2)
        else:
            clip_size = int((dim - 1) / 2)

        clips = np.zeros((indices[0].size, dest_size, dest_size))

        for k in range(0, indices[0].size):
            clipped_data = np.zeros((dest_size, dest_size))
            start_y = clip_size if indices[0][k] - clip_size >= 0 else indices[0][k]
            end_y = clip_size if indices[0][k] + clip_size < height else height - indices[0][k] - 1
            start_x = clip_size if indices[1][k] - clip_size >= 0 else indices[1][k]
            end_x = clip_size if indices[1][k] + clip_size < width else width - indices[1][k] - 1

            clipped_data[center_index - start_y:center_index + end_y + 1,
                         center_index - start_x:center_index + end_x + 1] =\
                image_data[indices[0][k] - start_y:indices[0][k] + end_y + 1, indices[1][k] - start_x:indices[1][k] + end_x + 1]
            clips[k, :, :] = clipped_data

        return clips


def read_training_data(path, num_training_pixels, mode='training'):

    annotation_path = join(path, '1st_manual')
    mask_path = join(path, 'pure_mask')
    training_image_path = join(path, 'images')
    annotations = [f for f in listdir(annotation_path) if isfile(join(annotation_path, f))]
    training_data = np.zeros((0, 33, 33))
    labels = np.zeros((0, ))

    for annotation in annotations:
        annotation_data = read_image_file(join(annotation_path, annotation))
        vein_pixels = np.where(annotation_data > 0)
        pixel_indices = np.random.randint(0, high=vein_pixels[0].size, size=num_training_pixels)
        vein_pixels = (vein_pixels[0][pixel_indices], vein_pixels[1][pixel_indices])

        id = annotation[0:2]
        mask_file = id + '_puremask.gif'
        mask_data = read_image_file(join(mask_path, mask_file))
        mask_pixels = np.where(mask_data > 0)
        pixel_indices = np.random.randint(0, high=mask_pixels[0].size, size=num_training_pixels)
        mask_pixels = (mask_pixels[0][pixel_indices], mask_pixels[1][pixel_indices])

        image_file = id + '_' + mode + '.tif'
        image_data = read_image_file(join(training_image_path, image_file))[:, :, 1]
        clipped_data = get_clipped_area(image_data, vein_pixels, 0)
        training_data = np.concatenate((training_data, clipped_data), axis=0)
        labels = np.concatenate((labels, np.ones((num_training_pixels, ))), axis=0)
        clipped_data = get_clipped_area(image_data, mask_pixels, 0)
        training_data = np.concatenate((training_data, clipped_data), axis=0)
        labels = np.concatenate((labels, np.zeros((num_training_pixels,))), axis=0)

    return [training_data, labels]


def generate_pure_mask_images(path, mode='training'):

    annotation_path = join(path, '1st_manual')
    training_mask_path = join(path, 'mask')
    pure_mask_path = join(path, 'pure_mask')
    if not isdir(pure_mask_path):
        mkdir(pure_mask_path)

    annotations = [f for f in listdir(annotation_path) if isfile(join(annotation_path, f))]

    for annotation in annotations:
        annotation_data = read_image_file(join(annotation_path, annotation))
        vein_pixels = np.where(annotation_data > 0)

        image_id = annotation[0:2]

        mask_file = image_id + '_' + mode + '_mask.gif'
        mask_data = read_image_file(join(training_mask_path, mask_file))
        mask_pixels = np.where(mask_data > 0)
        mask_pixels = remove_vein_pixels_from_mask(mask_pixels, vein_pixels)
        pure_mask_data = np.zeros((mask_data.shape), dtype=np.uint8)
        pure_mask_data[mask_pixels[0], mask_pixels[1]] = 255
        imsave(join(pure_mask_path, image_id + '_puremask.gif'), pure_mask_data)


def remove_vein_pixels_from_mask(mask_pixels, vein_pixels):
    bare_mask = [[], []]

    vein_iter = 0
    vein_iter_prev = 0
    for k in range(0, mask_pixels[0].size):

        if k % 1000 == 0:
            print(str(float(k) / float(mask_pixels[0].size)))
        notfound = True
        if k > 0 and mask_pixels[0][k] == mask_pixels[0][k - 1]:
            vein_iter = vein_iter_prev
        else:
            vein_iter_prev = vein_iter

        while vein_iter < vein_pixels[0].size and vein_pixels[0][vein_iter] <= mask_pixels[0][k]:
            if mask_pixels[0][k] == vein_pixels[0][vein_iter] and mask_pixels[1][k] == vein_pixels[1][vein_iter]:
                notfound = False

            vein_iter = vein_iter + 1

        if notfound:
            bare_mask[0].append(mask_pixels[0][k])
            bare_mask[1].append(mask_pixels[1][k])

    return bare_mask