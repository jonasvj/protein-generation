#!/usr/bin/env python3
import os 
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from gru_batch import GRUNet
from lstm_network import LSTMNet
from transformer_network import TransformerModel
from transformer_network2 import TransformerModel as TransformerModel2
import subprocess

class SequenceDataset(data.Dataset):
    def __init__(self, sequences):
        super(SequenceDataset, self).__init__()
        self.amino_acids = np.unique(list(''.join(sequences)))

        self.aa_to_idx = {self.amino_acids[i-1]: i
                          for i in range(1, len(self.amino_acids) + 1)}
        self.aa_to_idx['<PAD>'] = 0

        self.idx_to_aa = {i: self.amino_acids[i]
                          for i in range(len(self.amino_acids))}

        self.inputs, self.targets = self.encode_sequences(sequences)

    def encode_sequences(self, sequences):
        inputs = list()
        targets = list()
        for sequence in sequences:
            encoded_sequence = [self.aa_to_idx[aa] for aa in sequence]
            input_ = encoded_sequence[:-1]
            target = encoded_sequence[1:]

            inputs.append(input_)
            targets.append(target)
        
        return inputs, targets

    def __len__(self):
        # Return the number of sequences
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve inputs and targets at the given index
        return self.inputs[idx], self.targets[idx]

def custom_collate_fn(batch):
    """How to return a batch from the data loader"""
    batch.sort(key=lambda seq: len(seq[0]), reverse=True)
    inputs, targets = zip(*batch)
    input_lengths = [len(input_) for input_ in inputs]

    input_tensor = torch.zeros((max(input_lengths), len(inputs))).long()
    target_tensor = torch.zeros((max(input_lengths), len(inputs))).long()

    for i, length in enumerate(input_lengths):
            input_tensor[0:length, i] = torch.LongTensor(inputs[i][:])
            target_tensor[0:length, i] = torch.LongTensor(targets[i][:])
            
    return input_tensor, input_lengths, target_tensor


if __name__ == '__main__':

    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Load data
    df_train = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/train_data.txt'), sep='\t')
    df_val = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/val_data.txt'), sep='\t')

    train_data = SequenceDataset(df_train['sequence'])
    val_data = SequenceDataset(df_val['sequence'])

    train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True,
                                   pin_memory=True, num_workers=4,
                                   collate_fn=custom_collate_fn)
    
    val_loader = data.DataLoader(val_data, batch_size=64, shuffle=False,
                                 pin_memory=True, num_workers=4,
                                 collate_fn=custom_collate_fn)
    
    # Choose network model
    n_amino_acids = len(train_data.aa_to_idx)
    embedding_size = 20
    hidden_size = 500
    n_gru_layers = 5
    net = GRUNet(n_amino_acids, embedding_size, hidden_size,
                 n_gru_layers, dropout=0, bidirectional=False)
    
    #net = TransformerModel(n_amino_acids, n_amino_acids, 10, 50, 2)
    net = net.to(device=device)

    # Hyper-parameters
    num_epochs = 100

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=train_data.aa_to_idx['<PAD>'], reduction='none')
    
    optimizer = torch.optim.Adam(net.parameters())

    # Track loss and perplexity
    train_loss, val_loss = [], []

    for i in range(num_epochs):
        # Track loss
        epoch_train_loss = 0
        epoch_val_loss = 0
        
        val_start = time.time()
        net.eval()

        for inputs, input_lengths, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            print('input shape:', inputs.shape)
            print('target shape:', targets.shape)
            # Forward pass
            outputs = net(inputs, input_lengths)
            print('output shape', outputs.shape)

            # Compute loss
            loss = criterion(outputs, targets)
            print(loss.shape)
            
            sys.exit(1)
            # Update loss
            epoch_val_loss += loss.detach().cpu().numpy()
        
        val_end = time.time()
        
        train_start = time.time()
        net.train()

        for inputs, input_lengths, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs, input_lengths)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            epoch_train_loss += loss.detach().cpu().numpy()
        
        train_end = time.time()

        # Save loss
        train_loss.append(epoch_train_loss / len(train_data))
        val_loss.append(epoch_val_loss / len(val_data))

        # Print loss
        if i % 1 == 0:
            print('Epoch {}\nTraining loss: {},Validation loss: {}'.format(
                i, train_loss[-1], val_loss[-1]))
            print('Training time: {}, Validation time: {}\n'.format(
                round(train_end - train_start), round(val_end - val_start)))