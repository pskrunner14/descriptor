""" Trains the Image Captioning CRNN model
on embeddings of images and sequences of captions.
"""
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from descriptor.models.cnn_encoder import encode
from descriptor.models.rnn_decoder import RNNDecoder

from descriptor.utils.data import Image2CaptionDataset, load_vocab

def train():
    """
    """
    vocab = load_vocab(name='6B', dim=300)
    idx2word = vocab.itos
    word2idx = vocab.stoi
    vectors = vocab.vectors

    lr = 0.005
    max_len = 20
    batch_size = 64
    n_tokens = len(word2idx)

    train_dataset = Image2CaptionDataset(word2idx=word2idx)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, pin_memory=True)
    print(f'Training set size: {len(train_dataset)}')

    # val_dataset = Image2CaptionDataset(word2idx=word2idx)
    # val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, num_workers=8)
    # print(f'Validation set size: {len(val_dataset)}')

    model = RNNDecoder(2048, 120, n_tokens, 300, 256, 2, 'GRU', 0.5, 120,
                       padding_idx=word2idx['<PAD>'], pretrained_embeddings=vectors)
    if torch.cuda.is_available():
        model = model.cuda()

    # define training procedures and operations for training the model
    criterion = nn.NLLLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6,
                                                     factor=0.1, patience=7, verbose=True)

    num_epochs = 20
    total_iters = (len(train_dataset) // batch_size) if len(train_dataset) % batch_size == 0 else (len(train_dataset) // batch_size) + 1
    for epoch in range(1, num_epochs + 1):
        epoch_loss, n_iter = 0.0, 0

        # loop over batches
        for i, batch in tqdm(enumerate(train_data_loader), desc=f'Epoch[{epoch}/{num_epochs}]', leave=False, total=total_iters):
            inputs, truth = batch['image'], batch['caption']
            # optimize model parameters
            epoch_loss += optimize(model, inputs, truth, criterion, optimizer, n_tokens, max_len=max_len)
            n_iter += 1

        # # evaluate model after every epoch
        # val_loss = evaluate(model, val_tensors, max_length, n_tokens, criterion)
        # # lr_scheduler decreases lr when stuck at local minima
        # scheduler.step(val_loss)
        # log epoch status info
        # logging.info('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'
        #             .format(epoch, num_epochs, epoch_loss / n_iter, val_loss))

        # sample from the model every few epochs
        # if epoch % sample_every == 0:
            # for _ in range(num_samples):
            #     sample = generate_sample(model, token_to_idx, idx_to_token, 
            #                             max_length, n_tokens, seed_phrase=seed_phrase)
            #     logging.debug(sample)
        
def optimize(model, inputs, truth, criterion, optimizer, n_tokens, max_len=20):
    model.train()
    optimizer.zero_grad()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, n_tokens, max_len=max_len)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    # compute loss wrt targets
    loss = criterion(outputs, truth)
    # backpropagate error
    loss.backward()
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
    # update model parameters
    optimizer.step()
    return loss.item()

def forward(model, inputs, n_tokens, max_len=20):
    hidden = model.initHidden(inputs.size(0))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        if type(hidden) == tuple:
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

def evaluate(model, inputs, criterion, n_tokens, max_len=20):
    model.eval()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, n_tokens, max_len)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    targets = inputs[:, 1: ].contiguous().view(-1)
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    return loss.item()
