""" CNN Encoder model for Image Captioning CRNN model. """

# pylint:disable=E1101
import torch

from inception import inception_v3

IMG_SIZE = 299


def get_cnn_encoder():
    """ Returns the Pre-trained CNN Encoder model. """
    return inception_v3(pretrained=True).cuda()


def encode(cnn_encoder, images, debug=False):
    """ Generates embeddings of images using pre-trained
    InceptionV3 model to condition the Decoder RNN hidden states on.

    Args:
        cnn_encoder (torch.nn.Module): Pre-trained CNN Encoder model.
        images (torch.Tensor): Tensor containing images to encoded.
    """
    if debug:
        images = torch.randn(5, 3, 299, 299).to(device='cuda')
    else:
        images = images.to(device='cuda')
    return cnn_encoder(images)

if __name__ == "__main__":
    print(encode(get_cnn_encoder(), None, debug=True).size())
