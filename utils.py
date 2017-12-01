import scipy.misc
import math
import numpy as np

# conv out size
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# load data from datasets
def get_image(batch_file, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(batch_file, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(batch_file).astype(np.float)
        
def save_images(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img
