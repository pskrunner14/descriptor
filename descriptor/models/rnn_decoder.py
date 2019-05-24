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
    """ Descriptor: RNN Decoder model for Image Captioning model
    that generates sequences for captioning an image encoding.

    Args:

    """

    def __init__(self, img_emb_size, n_tokens, embedding_dim=300,
                 hidden_size=256, num_layers=2, rnn_type='gru', dropout_p=0.5,
                 padding_idx=1, pretrained_embeddings=None):
        super(Descriptor, self).__init__()
        # conversion layer from encoder hidden to decoder hidden states
        self.img_embed_to_hidden = nn.Sequential(
            nn.Linear(img_emb_size, hidden_size).apply(glorot_normal_initializer),
            nn.ELU()
        )
        # bottom level input embedding layer
        # init embedding layer with pretrained embeddings if available
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(n_tokens, embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout_p)
        self._rnn_type = rnn_type.upper()
        # core rnn layers
        if self._rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.__num_layers = num_layers
            self.__hidden_size = hidden_size
            self.rnn = getattr(nn, self._rnn_type)(
                input_size=embedding_dim, hidden_size=self.__hidden_size,
                num_layers=self.__num_layers, dropout=dropout_p, batch_first=True
            )
        else:
            raise UserWarning('invalid RNN type!')
        # top level output softmax layer
        self.logit_probs = nn.Sequential(
            nn.Linear(hidden_size, n_tokens).apply(glorot_normal_initializer),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, token_idx, hidden=None, image_embeddings=None):
        """ Implements the forward pass of the Descriptor Decoder RNN.

        Args:
        -----
            token_idx (torch.LongTensor): input step token batch to feed the network.
            hidden (torch.Tensor): hidden states of the RNN from previous time-step.
            image_embeddings (torch.Tensor): embeddings of the images from CNN encoder.

        Returns:
        --------
            torch.Tensor: output log softmax probability distribution over tokens.
            torch.Tensor (or tuple of torch.Tensor for LSTM): hidden states of the
            RNN from current time-step.
        """
        batch_size = token_idx.size(0)
        # init the hidden state (and cell state) with
        # the latent representation of the input image
        if image_embeddings is not None and hidden is None:
            # compress image embedding to hidden layer dimensions
            i2h = self.img_embed_to_hidden(image_embeddings)
            # repeat tensor for each hidden layer of rnn
            i2h = i2h.unsqueeze(dim=0).repeat(self.__num_layers, 1, 1)
            # init hidden state conditioned on image embedding
            if self._rnn_type == 'LSTM':
                hidden = (i2h, i2h)
            else:
                hidden = i2h
        # convert idx to corresponding word vectors
        out = self.dropout(self.embedding(token_idx))
        # pass through rnn layers
        out, hidden = self.rnn(out.view(batch_size, 1, -1), hidden)
        # compute token logits over rnn outputs
        # apply log softmax over logits to get prob distribution
        out = self.logit_probs(out.view(-1, self.__hidden_size))
        # return prob distribution and hidden states for next time step
        return out, hidden

class BeamSearchDecoder():
    """Beam Search enabled RNN Decoder.
    """

    pass
