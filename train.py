""" Trains the Image Captioning CRNN model
on embeddings of images and sequences of captions.
"""
import torch
from torch import nn
from torch import optim

from cnn_encoder import encode
from rnn_decoder import RNNDecoder
from util import ImageDataset

def train():
    """
    """
    train_dataset = ImageDataset()
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    print(f'Training set size: {len(train_dataset)}')

    for i, batch in enumerate(train_data_loader):
        print(i, batch['image'].size())#, batch['captions'].size())
        if i > 10:
            break

def main():
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == "__main__":
    main()
