""" Data Utility Module for Image Captioning CRNN model. """
import os
import json
import collections

import numpy as np
import imageio as io

import torch
import torchvision as vision
import torchtext as text

SPECIAL_TOKENS = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']

cuda0 = torch.device('cuda:0')

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
        filenames_to_captions[id_to_filename[caption['image_id']]].append(caption['caption'])
    filenames_to_captions = dict(filenames_to_captions)
    # create a list of list of captions so we can access by idx
    return list(map(lambda x: filenames_to_captions[x], filenames))

def load_vocab(name='6B', dim=300):
    """
    """
    vocab = text.vocab.GloVe(name=name, dim=dim)
    print(f'Loaded {len(vocab.itos)} {vocab.dim}-dimensional word vectors!')
    vocab.itos = SPECIAL_TOKENS + vocab.itos

    del vocab.stoi
    vocab.stoi = {}
    for i, word in enumerate(vocab.itos):
        vocab.stoi[word] = i

    print(f'Adding special tokens to the vocab: {SPECIAL_TOKENS}')
    special_token_tensors = torch.zeros(len(SPECIAL_TOKENS), vocab.dim)
    vocab.vectors = torch.cat(tensors=(special_token_tensors, vocab.vectors))

    print(vocab.itos[:5])
    print(vocab.vectors.size())
    return vocab

def seq_to_tensor(sequence, word2idx, max_len=20):
    """ Casts a list of sequences into rnn-digestable padded tensor """
    seq_idx = torch.Tensor([word2idx['<SOS>']] + [word2idx[token] if token in word2idx else word2idx['<UNK>'] for token in sequence.lower().split(' ')])
    seq_idx = seq_idx[: max_len] if len(seq_idx) < max_len else seq_idx[: max_len - 1]
    seq_idx = torch.cat(tensors=(seq_idx, torch.Tensor([word2idx['<EOS>']])))
    seq_idx = torch.cat(tensors=(seq_idx, torch.Tensor([word2idx['<PAD>']] * (max_len - len(seq_idx)))))
    return seq_idx


class Image2CaptionDataset(torch.utils.data.Dataset):
    """ Image dataset.

    Args:
        json_file (string): Path to the json file with annotations.
        root_dir (string): Directory with all the images.
    """

    def __init__(self, word2idx, max_len=20,
                json_file='captions_train2014.json',
                root_dir='data/train2014'):
        self.word2idx = word2idx
        self.max_len = max_len
        self.images = os.listdir(root_dir)
        self.captions = get_captions(
            f'data/captions_train-val2014/annotations/{json_file}',
            self.images
        )
        self.root_dir = root_dir
        self.transform = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.Resize((299, 299)),
            vision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = io.imread(f'{self.root_dir}/{img_name}')

        image = image_center_crop(image)
        image = self.transform(image)

        ridx = np.random.randint(5)
        caption = self.captions[idx][ridx]

        return {
            'image': image,
            'caption': seq_to_tensor(caption, self.word2idx, max_len=self.max_len)
            }

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
