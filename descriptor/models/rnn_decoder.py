""" RNN Decoder model for Image Captioning CRNN model. """
import torch
import torch.nn as nn
import torch.functional as F

def glorot_normal_initializer(m):
    """ Applies Glorot Normal initialization to layer parameters.
    
    "Understanding the difficulty of training deep feedforward neural networks" 
    by Glorot, X. & Bengio, Y. (2010)
    Args:
        m (nn.Module): a particular layer whose params are to be initialized.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

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
            nn.Linear(img_emb_size, bottleneck_size).apply(glorot_normal_initializer),
            nn.ELU()
        )
        self.img_bottleneck_to_hidden = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size).apply(glorot_normal_initializer),
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
        self.token_logits_bottleneck = nn.Linear(hidden_size, logit_bottleneck_size).apply(glorot_normal_initializer)
        self.token_logits = nn.Linear(logit_bottleneck_size, n_tokens).apply(glorot_normal_initializer)

    def forward(self, inputs, hidden, image_embs=None):
        if image_embs is not None:
            e2b = self.img_embed_to_bottleneck(image_embs)
            b2h = self.img_bottleneck_to_hidden(e2b)
            h2h = b2h.view(self.num_layers, inputs.size(0), self.hidden_size)
            hidden = (h2h, h2h)
        embeds = self.embedding(inputs)
        out = self.dropout(embeds)
        outputs, hidden = self.rnn(out.view(inputs.size(0), 1, -1), hidden)
        logits = self.logits(outputs.view(-1, self.hidden_size))
        probs = F.log_softmax(logits, dim=1)
        return probs

    def initHidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
