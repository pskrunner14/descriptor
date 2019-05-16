""" Trains the Image Captioning CRNN model
on embeddings of images and sequences of captions.
"""
import logging
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from descriptor.utils.data import Image2CaptionDataset, load_vocab
from descriptor.models.cnn_encoder import get_cnn_encoder, encode
from descriptor.models.rnn_decoder import RNNDecoder

def train():
    """
    """
    vocab = load_vocab(name='6B', dim=300)
    idx2word = vocab.itos
    word2idx = vocab.stoi
    vectors = vocab.vectors

    lr = 0.005
    max_len = 20
    num_epochs = 20
    batch_size = 16
    n_tokens = len(word2idx)

    train_dataset = Image2CaptionDataset(word2idx=word2idx)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, pin_memory=True)
    print(f'Training set size: {len(train_dataset)}')

    val_dataset = Image2CaptionDataset(word2idx=word2idx, 
                                       json_file='captions_val2014.json', 
                                       root_dir='data/val2014')
    val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, num_workers=8)
    print(f'Validation set size: {len(val_dataset)}')

    cnn_model = get_cnn_encoder()
    rnn_model = RNNDecoder(2048, 120, n_tokens, 300, 256, 2, 'GRU', 0.5, 120,
                           padding_idx=word2idx['<PAD>'], pretrained_embeddings=vectors)
    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()
        rnn_model = rnn_model.cuda()

    # define training procedures and operations for training the model
    criterion = nn.NLLLoss(reduction='mean')
    optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6,
                                                     factor=0.1, patience=7, verbose=True)

    bmod = len(train_dataset) % batch_size
    bdiv = len(train_dataset) // batch_size
    total_iters = bdiv if bmod == 0 else bdiv + 1

    for epoch in range(1, num_epochs + 1):
        epoch_loss, n_iter = 0.0, 0

        # loop over batches
        for i, batch in tqdm(enumerate(train_data_loader), desc=f'Epoch[{epoch}/{num_epochs}]', leave=False, total=total_iters):
            inputs, targets = batch['image'], batch['caption']
            inputs = encode(inputs.cuda(), cnn_encoder=cnn_model)

            # optimize model parameters
            epoch_loss += optimize(rnn_model, inputs, targets, criterion, optimizer, n_tokens, max_len=max_len)
            n_iter += 1

        # evaluate model after every epoch
        val_batch = next(iter(val_data_loader))
        val_inputs, val_targets = val_batch['image'], val_batch['caption']
        val_loss = evaluate(rnn_model, val_inputs, val_targets, criterion, n_tokens, max_len=max_len)
        # lr_scheduler decreases lr when stuck at local minima
        scheduler.step(val_loss)
        # log epoch status info
        logging.info(f'Epoch[{epoch}/{num_epochs}]: train_loss - {(epoch_loss / n_iter):.4f}   val_loss - {val_loss:.4f}')

        # # sample from the model every few epochs
        # if epoch % sample_every == 0:
        #     for _ in range(num_samples):
        #         sample = generate_sample(rnn_model, token_to_idx, idx_to_token,
        #                                 max_length, n_tokens, seed_phrase=seed_phrase)
        #         logging.debug(sample)
        
def optimize(model, inputs, targets, criterion, optimizer, n_tokens, max_len=20):
    model.train()
    optimizer.zero_grad()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, n_tokens, max_len=max_len)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    # backpropagate error
    loss.backward()
    _ = nn.utils.clip_grad_norm_(model.parameters(), 50.0)
    # update model parameters
    optimizer.step()
    return loss.item()

def forward(model, inputs, n_tokens, max_len=20):
    for emb in inputs:
        hidden = model.initHidden(inputs.size(0))
        hidden = model.img_embed_to_bottleneck
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            if isinstance(hidden, tuple):
                hidden = tuple([x.cuda() for x in hidden])
            else:
                hidden = hidden.cuda()
        
        # tensor for storing outputs of each time-step
        outputs = torch.Tensor(max_len, inputs.size(0), n_tokens)
        # loop over time-steps
        for t in range(max_len):
            # t-th time-step input
            input_t = inputs[:, t]
            outputs[t], hidden = model(input_t, hidden)
        # (timesteps, batches, n_tokens) -> (batches, timesteps, n_tokens)
        outputs = outputs.permute(1, 0, 2)
        # ignore the last time-step since we don't have a target for it.
        outputs = outputs[:, :-1, :]
    # (batches, timesteps, n_tokens) -> (batches x timesteps, n_tokens)
    outputs = outputs.contiguous().view(-1, n_tokens)
    return outputs

def evaluate(model, inputs, targets, criterion, n_tokens, max_len=20):
    model.eval()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, n_tokens, max_len)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    # targets = inputs[:, 1: ].contiguous().view(-1)
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    return loss.item()
