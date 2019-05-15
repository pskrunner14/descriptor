""" RNN Decoder model for Image Captioning CRNN model. """
import torch
import torch.nn as nn
import torch.functional as F


class RNNDecoder(nn.Module):
    """ RNN Decoder model for Image Captioning model
    that generates sequences for caption of an image encoding.

    Args:

    """

    def __init__(self, img_emb_size, bottleneck_size, n_tokens,
                 embedding_dim, hidden_size, num_layers, rnn_type,
                 dropout, logit_bottleneck_size, padding_idx=1,
                 pretrained_embeddings=None):
        super(RNNDecoder, self).__init__()
        self.img_embed_to_bottleneck = nn.Sequential(
            nn.Linear(img_emb_size, bottleneck_size),
            nn.ELU()
        )
        self.img_bottleneck_to_hidden = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ELU()
        )
        self.embedding = nn.Embedding(n_tokens, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings:
            self.embedding.weight = pretrained_embeddings
        self.rnn_type = rnn_type.upper()
        if self.rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(input_size=embedding_dim, hidden_size=hidden_size,
                                                  num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            raise UserWarning('invalid RNN type!')
        self.token_logits_bottleneck = nn.Linear(hidden_size, logit_bottleneck_size)
        self.token_logits = nn.Linear(logit_bottleneck_size, n_tokens)

    def forward(self, inputs):
        pass
