from os import listdir, mkdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate_statistics():

    image_path = join('..', 'data', 'DRIVE', 'test', 'images')
    gt_path = join('..', 'data', 'DRIVE', 'test', '1st_manual')
    mask_path = join('..', 'data', 'DRIVE', 'test', 'mask')
    result_path = join('..', 'data', 'results', 'rf')

    images = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    patch_size = 2000

    all_correct_labels = np.zeros((0, 1))
    all_predicted_labels = np.zeros((0, 1))

    for image in images:

        id = image[0:2]
        image_file = join(image_path, image)
        gt_file = join(gt_path, id + '_manual1.gif')
        mask_file = join(mask_path, id + '_test_mask.gif')
        result_file = join(result_path, image)
        image_data = plt.imread(image_file)
        gt_data = plt.imread(gt_file)
        mask_data = plt.imread(mask_file)
        predicted_data = 255 - plt.imread(result_file)[:, :, 0]        # plt.figure()
        indices = np.where(mask_data == 255)
        correct_labels = np.reshape(gt_data[indices] / 255, [indices[0].size, 1])
        predicted_labels = np.reshape(predicted_data[indices] / 255, [indices[0].size, 1])
        all_correct_labels = np.row_stack((all_correct_labels, correct_labels))
        all_predicted_labels = np.row_stack((all_predicted_labels, predicted_labels))
        accuracy = accuracy_score(correct_labels, predicted_labels)
        print("Accuracy for image" + image + ": {:.2f}".format(accuracy))
        # plt.imshow(image_data)
        # plt.figure()
        # plt.imshow(gt_data, cmap='binary', interpolation='nearest')
        # plt.figure()
        # plt.imshow(mask_data, cmap='binary', interpolation='nearest')
        # plt.figure()
        # plt.imshow(predicted_data, cmap='binary', interpolation='nearest')
        # plt.show()

    accuracy = accuracy_score(correct_labels, predicted_labels)
    print("Total accuracy: {:.2f}".format(accuracy))
    c_matrix = confusion_matrix(all_correct_labels, all_predicted_labels)
    fig, ax = plt.subplots(1, figsize=(4, 4))
    ax.set_title("Confusion matrix")
    sns.heatmap(c_matrix, cmap='Blues', annot=True, fmt='g', cbar=False)
    ax.set_xlabel('predictions')
    ax.set_ylabel('true labels')
    plt.show()



if __name__ == '__main__':
    evaluate_statistics()