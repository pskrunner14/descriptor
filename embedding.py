""" Generates embeddings of images using pre-trained
InceptionV3 model to condition the Decoder RNN hidden states on.
"""
# pylint:disable=E1101
import torch
import torchvision as vision

from inception import inception_v3

IMG_SIZE = 299


def create_embeddings():
    """ Create embeddings from images.
    """
    cnn_encoder = inception_v3(pretrained=True).cuda()

    transform = vision.transforms.Compose([
        vision.transforms.Resize((299, 299)),
        vision.transforms.ToTensor()
    ])

    images = torch.randn(5, 3, 299, 299).to(device='cuda')
    output = cnn_encoder(images)
    print(output.size())
