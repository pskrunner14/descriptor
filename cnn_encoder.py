""" Generates embeddings of images using pre-trained
InceptionV3 model to condition the Decoder RNN hidden states on.
"""
# pylint:disable=E1101
import torch

from inception import inception_v3

IMG_SIZE = 299


def get_cnn_encoder():
    return inception_v3(pretrained=True).cuda()


def encode(cnn_encoder, images):
    """ Generate embeddings/encodings from images.
    """
    images = torch.randn(5, 3, 299, 299).to(device='cuda')
    output = cnn_encoder(images)
    print(output.size())
