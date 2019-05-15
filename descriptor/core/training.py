""" Trains the Image Captioning CRNN model
on embeddings of images and sequences of captions.
"""
import torch
from torch import nn
from torch import optim

from descriptor.models.cnn_encoder import encode
from descriptor.models.rnn_decoder import RNNDecoder

from descriptor.utils.data import Image2CaptionDataset, load_vocab

def train():
    """
    """
    vocab = load_vocab()
    idx2word = vocab.itos
    word2idx = vocab.stoi
    vectors = vocab.vectors

    train_dataset = Image2CaptionDataset(word2idx=word2idx)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                    shuffle=True, num_workers=8)
    print(f'Training set size: {len(train_dataset)}')

    # val_dataset = Image2CaptionDataset(word2idx=word2idx)
    # val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, num_workers=8)
    # print(f'Validation set size: {len(val_dataset)}')

    for i, batch in enumerate(train_data_loader):
        print(i, batch['image'].size(), batch['captions'].size())
        if i > 10:
            break
