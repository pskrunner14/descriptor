""" RNN Decoder model for Image Captioning CRNN model. """

import torch

from torch import nn

def glorot_normal_initializer(module):
    """Applies Glorot Normal initialization to layer parameters.

        "Understanding the difficulty of training deep feedforward neural networks"
        - Glorot, X. & Bengio, Y. (2010)
    Args:
    -----
        module (nn.Module): a particular layer whose params are to be initialized.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)

class Descriptor(nn.Module):
    """ RNN Decoder model for Image Captioning model
    that generates sequences for caption of an image encoding.

    Args:

    """

    def __init__(self, img_emb_size, n_tokens,
                 embedding_dim, hidden_size, num_layers, rnn_type,
                 dropout, padding_idx=1,
                 pretrained_embeddings=None):
        super(Descriptor, self).__init__()
        self.img_embed_to_hidden = nn.Sequential(
            nn.Linear(img_emb_size, hidden_size).apply(glorot_normal_initializer),
            nn.ELU()
        )
        self.embedding = nn.Embedding(n_tokens, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(data=pretrained_embeddings)
        self.rnn_type = rnn_type.upper()
        if self.rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.rnn = getattr(nn, self.rnn_type)(input_size=embedding_dim, hidden_size=hidden_size,
                                                  num_layers=num_layers, dropout=dropout,
                                                  batch_first=True)
        else:
            raise UserWarning('invalid RNN type!')
        self.token_logits = \
            nn.Linear(hidden_size, n_tokens).apply(glorot_normal_initializer)
        self.probs = nn.LogSoftmax(dim=1)

    def forward(self, token_idx, hidden=None, image_embeddings=None):
        """ Implements the forward pass of the char-level RNN.
        
        Args:
        -----
            token_idx (torch.LongTensor): input step token batch to feed the network.
            hidden (torch.Tensor): hidden states of the RNN from previous time-step.
            image_embeddings (torch.Tensor): embeddings of the images from CNN encoder.
        Returns:
        --------
            torch.Tensor: output log softmax probability distribution over tokens.
            torch.Tensor (or tuple of torch.Tensor for LSTM): hidden states of the RNN from current time-step.
        """
        batch_size = token_idx.size(0)
        # init the hidden state (and cell state) with
        # the latent representation of the input image
        if image_embeddings is not None and hidden is None:
            b2h = self.img_embed_to_hidden(image_embeddings)
            h2h = b2h.unsqueeze(dim=0).repeat(self.num_layers, 1, 1)
            if self.rnn_type == 'LSTM':
                hidden = (h2h, h2h)
            else:
                hidden = h2h
        vecs = self.embedding(token_idx)
        vecs_d = self.dropout(vecs)
        out, hidden = self.rnn(vecs_d.view(batch_size, 1, -1), hidden)
        logits = self.token_logits(out.view(-1, self.hidden_size))
        probs = self.probs(logits)
        return probs, hidden

    # def init_hidden(self, batch_size):
    #     if self.rnn_type == 'LSTM':
    #         return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
    #                 torch.zeros(self.num_layers, batch_size, self.hidden_size))
    #     return torch.zeros(self.num_layers, batch_size, self.hidden_size)
