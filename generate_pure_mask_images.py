from os.path import join
from utils import image_utils

path = join('data', 'DRIVE', 'training')
image_utils.generate_pure_mask_images(path)