""" Trains the Image Captioning CRNN model
on embeddings of images and sequences of captions.
"""
import torch
from torch import nn
from torch import optim

from descriptor.models.cnn_encoder import encode
from descriptor.models.rnn_decoder import RNNDecoder

from descriptor.utils.data import ImageDataset

def train():
    """
    """
    train_dataset = ImageDataset()
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                    shuffle=True, num_workers=8)
    print(f'Training set size: {len(train_dataset)}')

    for i, batch in enumerate(train_data_loader):
        print(i, batch['image'].size())#, batch['captions'].size())
        if i > 10:
            break
