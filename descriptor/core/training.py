""" Trains the Image Captioning CRNN model
on embeddings of images and sequences of captions.
"""
import logging
import torch

from tqdm import tqdm
from torch import nn, optim

from descriptor.utils.data import Image2CaptionDataset, load_vocab, SPECIAL_TOKENS
# from descriptor.models.cnn_encoder import get_cnn_encoder, encode
from descriptor.models.rnn_decoder import Descriptor

idx2word = None
word2idx = None

def train():
    """Trains the CRNN Autoencoder model for Image Captioning.

    Args:
    -----
    """
    vocab = load_vocab(name='6B', dim=300)
    global idx2word
    idx2word = vocab.itos
    global word2idx
    word2idx = vocab.stoi
    vectors = vocab.vectors

    lr = 0.005
    max_len = 20
    num_epochs = 20
    batch_size = 2
    n_tokens = len(word2idx)

    train_dataset = Image2CaptionDataset(word2idx=word2idx)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, pin_memory=True)
    print(f'Training set size: {len(train_dataset)}')

    val_dataset = Image2CaptionDataset(
        word2idx=word2idx,
        root_dir='data/val2014',
        json_file='captions_val2014.json'
    )
    val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, 
                                                  num_workers=8, pin_memory=True)
    print(f'Validation set size: {len(val_dataset)}')

    rnn_model = Descriptor(
        2048, n_tokens, 300, 64, 1, 'GRU',
        0.5, padding_idx=word2idx['<PAD>'],
        pretrained_embeddings=vectors
    )
    if torch.cuda.is_available():
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
        for _, batch in tqdm(enumerate(train_data_loader), 
                             desc=f'Epoch[{epoch}/{num_epochs}]', 
                             leave=False, total=total_iters):

            inputs, targets = batch['image'], batch['caption']
            # inputs = encode(inputs.cuda(), cnn_encoder=cnn_model)

            # optimize model parameters
            epoch_loss += optimize(rnn_model, inputs, targets,
                                   criterion, optimizer, n_tokens,
                                   max_len=max_len)
            n_iter += 1

        # evaluate model after every epoch
        val_batch = next(iter(val_data_loader))
        val_inputs, val_targets = val_batch['image'], val_batch['caption']
        val_loss = evaluate(rnn_model, val_inputs, val_targets, 
                            criterion, n_tokens, max_len=max_len)
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
    # (batches, timesteps) -> (timesteps x batches)
    targets = targets[:, 1:].contiguous().view(-1)
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    # backpropagate error
    loss.backward()
    _ = nn.utils.clip_grad_norm_(model.parameters(), 50.0)
    # update model parameters
    optimizer.step()
    return loss.item()

def forward(model, inputs, n_tokens, max_len=20):
    """
    """
    batch_size = inputs.size(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    # init input <SOS> token for all batches
    tokens = torch.LongTensor([word2idx['<SOS>']] * batch_size).cuda()
    # tensor for storing outputs of each time-step
    outputs = torch.Tensor(max_len, batch_size, n_tokens)
    # condition the decoder hidden states on image encoding
    outputs[0], hidden = model(tokens, image_embeddings=inputs)
    # loop over time-steps
    for t in range(1, max_len):
        # t-th time-step input
        _, tokens = outputs[t - 1].topk(1, dim=1)
        outputs[t], hidden = model(tokens.cuda(), hidden)
    # (timesteps, batches, n_tokens) -> (batches, timesteps, n_tokens)
    outputs = outputs.permute(1, 0, 2)
    # ignore the last time step since we don't have reference token for it
    outputs = outputs[:, :-1, :]
    # (batches, timesteps, n_tokens) -> (batches x timesteps, n_tokens)
    outputs = outputs.contiguous().view(-1, n_tokens)
    return outputs

def evaluate(model, inputs, targets, criterion, n_tokens, max_len=20):
    """
    """
    model.eval()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, n_tokens, max_len)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    print(targets.size())
    targets = targets[:, 1:].contiguous().view(-1)
    print(targets.size())
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    return loss.item()
