""" Utility Module for Image Captioning CRNN model. """
import os
import json
import collections

import torch
import torchvision as vision

import imageio as io
import matplotlib.pyplot as plt


def get_captions(json_file_path, filenames):
    """ Get captions for given filenames.

    Args:
        json_file_path (string): Path to the json file containing annotations/captions.
        filenames (list): List with all the filenames.
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
    # dict(image_idx: image_file_name)
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    # defualtdict(new_key: [])
    filenames_to_captions = collections.defaultdict(list)
    # add captions corresponding to image under image_id in dict
    for caption in data['annotations']:
        filenames_to_captions[id_to_filename[caption['image_id']]].append(
            caption['caption'])
    filenames_to_captions = dict(filenames_to_captions)
    # create a list of list of captions so we can access by idx
    return list(map(lambda x: filenames_to_captions[x], filenames))


class ImageDataset(torch.utils.data.Dataset):
    """ Image dataset.

    Args:
        json_file (string): Path to the json file with annotations.
        root_dir (string): Directory with all the images.
    """

    def __init__(self, json_file='captions_train2014.json', root_dir='data/train2014'):
        self.filenames = os.listdir(root_dir)
        self.captions = get_captions(
            f'data/captions_train-val2014/annotations/{json_file}',
            self.filenames
        )
        self.root_dir = root_dir
        self.transform = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.Resize((299, 299)),
            vision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = io.imread(f'{self.root_dir}/{img_name}')
        image = image_center_crop(image)
        image = self.transform(image)
        captions = self.captions[idx]
        return image, captions


def image_center_crop(img):
    """ Center crop images.

    Args:
        img (numpy.ndarray): Image array to crop.
    """
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top: h - pad_bottom, pad_left: w - pad_right, :]


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
