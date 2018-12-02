import torch
import torch.nn as nn

from inception import inception_v3

IMG_SIZE = 299


def main():
    CNNEncoder = inception_v3(pretrained=True).cuda()

    images = torch.randn(5, 3, 299, 299).to(device='cuda')
    output = CNNEncoder(images)
    print(output.size())


if __name__ == '__main__':
    main()
