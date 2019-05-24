""" Data Utility Module for Image Captioning CRNN model. """
import os
import json
import collections
import multiprocessing as mp

import numpy as np
import cv2

from tqdm import tqdm

import torch
import torchvision as vision
import torchtext as text

from descriptor.models.cnn_encoder import get_cnn_encoder, encode
from keras.applications.inception_v3 import preprocess_input

SPECIAL_TOKENS = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']

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
    """Loads a pretrained GloVe word embeddings model.

    Args:
    -----
        name (str): name of the GloVe model.
        dim (int): dimension of the word vector.

    Returns:
    --------
        torchtext.vocab.GloVe: the pretrained GloVe word embeddings model.
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
    """Casts a text sequence into rnn-digestable padded tensor.

    Args:
    -----
        sequence (str): the input text sequence.
        word2idx (dict): the mapping from word to index.
        max_len (int): maximum rnn-digestable length of the sequence.

    Returns:
    --------
        torch.Tensor: the output tensor of token ids.
    """
    seq_idx = torch.LongTensor([word2idx['<SOS>']] + [word2idx[token] \
                            if token in word2idx else word2idx['<UNK>'] \
                            for token in sequence.lower().split(' ')])
    seq_idx = seq_idx[: max_len] if len(seq_idx) < max_len else seq_idx[: max_len - 1]
    seq_idx = torch.cat(tensors=(seq_idx, torch.LongTensor([word2idx['<EOS>']])))
    seq_idx = torch.cat(tensors=(seq_idx, torch.LongTensor([word2idx['<PAD>']] \
                                          * (max_len - len(seq_idx)))))
    return seq_idx

class Image2TensorDataset(torch.utils.data.Dataset):
    """
    """
    
    def __init__(self, root_dir='data/train2014'):
        self.__root_dir = root_dir
        self.__image_paths = list(filter(lambda x: x.endswith('.jpg'), os.listdir(root_dir)))
        self.__transform = vision.transforms.Compose([
            vision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.__image_paths)

    def __getitem__(self, idx):
        file_name = self.__image_paths[idx]
        img = cv2.imread(f'{self.__root_dir}/{file_name}')
        img = image_center_crop(img)
        img = cv2.resize(img, (299, 299)).astype('float32')
        img = preprocess_input(img) # preprocess for model
        tensor = self.__transform(img)
        return {
            'image': tensor,
            'file_name': file_name
        }

class ImageTensor2CaptionDataset(torch.utils.data.Dataset):
    """Image to Caption mapping dataset.

    Args:
        word2idx (dict): word to index mapping vocab.
        max_len (int): maximum allowed length of a caption string.
        root_dir (string): directory with all the images.
        json_file (string): path to the json file with annotations.
    """

    def __init__(self, word2idx, max_len=20,
                 root_dir='data/train2014',
                 json_file='captions_train2014.json'):
        self._max_len = max_len
        self.__word2idx = word2idx
        self.__root_dir = root_dir
        self.__image_paths = list(filter(lambda x: x.endswith('.jpg'), os.listdir(root_dir)))
        self.__tensor_paths = list(filter(lambda x: x.endswith('.pt'), os.listdir(root_dir)))
        assert len(self.__image_paths) == len(self.__tensor_paths), 'conversion to tensors buggy'
        self.__captions = get_captions(
            f'data/captions_train-val2014/annotations/{json_file}',
            self.__image_paths
        )
        
    def __len__(self):
        return len(self.__tensor_paths)

    def __getitem__(self, idx):
        tensor_name = self.__tensor_paths[idx]
        image_tensor = torch.load(f"{self.__root_dir}/{tensor_name}")

        ridx = np.random.randint(5)
        caption = self.__captions[idx][ridx]
        caption = seq_to_tensor(caption, self.__word2idx, max_len=self._max_len)

        return {
            'image': image_tensor,
            'caption': caption
        }

def image_center_crop(img):
    """ Center crop images.

    Args:
    -----
        img (numpy.ndarray): Image array to crop.

    Returns:
    --------
        numpy.ndarray:
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

def encode_and_save(root_dir, cnn_encoder=get_cnn_encoder()):
    batch_size = 12

    dataset = Image2TensorDataset(root_dir=root_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=8,
                                             pin_memory=True)

    bmod = len(dataset) % batch_size
    bdiv = len(dataset) // batch_size
    total_iters = bdiv if bmod == 0 else bdiv + 1

    for _, batch in tqdm(enumerate(dataloader), total=total_iters, leave=True,
                         desc=f'Encoding images into embeddings and saving tensors to files: {root_dir}'):
        images, file_names = batch['image'], batch['file_name']
        tensors = encode(images.cuda(), cnn_encoder=cnn_encoder).cpu()
        for i, file_name in enumerate(file_names):
            torch.save(tensors[i], f"{root_dir}/{file_name.replace('.jpg', '.pt')}")

def main():
    cnn_encoder = get_cnn_encoder()
    if torch.cuda.is_available():
        cnn_encoder = cnn_encoder.cuda()

    paths = ['data/train2014', 'data/val2014']
    for path in paths:
        encode_and_save(path, cnn_encoder=cnn_encoder)

if __name__ == "__main__":
    main()
