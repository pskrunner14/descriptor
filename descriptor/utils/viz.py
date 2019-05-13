""" Visualization Utility Module for Image Captioning CRNN model. """
import os

import imageio as io
import matplotlib.pyplot as plt

from .data import get_captions, image_center_crop

def show_example(idx, json_file='captions_train2014.json', root_dir='data/train2014'):
    """ Shows a Training example image along with
    all captions corresponding to the image.

    Args:
        json_file (string): Path to the json file with annotations.
        root_dir (string): Directory with all the images.
    """
    filenames = os.listdir(root_dir)
    captions = get_captions(
        f'data/captions_train-val2014/annotations/{json_file}',
        filenames
    )
    img, captions = io.imread(
        f'{root_dir}/{filenames[idx]}'), captions[idx]
    plt.imshow(image_center_crop(img))
    plt.title("\n".join(captions))
    plt.show()
