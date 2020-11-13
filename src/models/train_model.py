#!/usr/bin/env python3
import os 
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from gru_network import GRUNet
from lstm_network import LSTMNet
from transformer_network import TransformerModel
import subprocess

class SequenceDataset(data.Dataset):
    def __init__(self, sequences):
        super(SequenceDataset, self).__init__()
        self.amino_acids = np.unique(list(''.join(sequences)))

        self.aa_to_idx = {self.amino_acids[i]: i
                          for i in range(len(self.amino_acids))}

        self.idx_to_aa = {i: self.amino_acids[i]
                          for i in range(len(self.amino_acids))}

        self.inputs, self.targets = self.encode_sequences(sequences)

    def encode_sequences(self, sequences):
        inputs = list()
        targets = list()
        for sequence in sequences:
            encoded_sequence = [self.aa_to_idx[aa] for aa in sequence]
            input_ = torch.tensor(encoded_sequence[:-1])
            target = torch.tensor(encoded_sequence[1:])

            inputs.append(input_)
            targets.append(target)
        
        return inputs, targets

    def __len__(self):
        # Return the number of sequences
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve inputs and targets at the given index
        return self.inputs[idx], self.targets[idx]

if __name__ == '__main__':

    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

    if torch.torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Load data
    df_train = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/train_data.txt'), sep='\t')
    df_val = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/val_data.txt'), sep='\t')

    train_data = SequenceDataset(df_train['sequence'])
    val_data = SequenceDataset(df_train['sequence'])

    train_loader = data.DataLoader(train_data, batch_size=1,
                                   shuffle=True, pin_memory=True,
                                   prefetch_factor=20,
                                   num_workers=6)
    
    val_loader = data.DataLoader(val_data, batch_size=1,
                                 shuffle=False, pin_memory=True,
                                 prefetch_factor=20,
                                 num_workers=6)
    
    # Choose network model
    n_amino_acids = len(train_data.amino_acids)
    net = GRUNet(n_amino_acids, n_amino_acids, 500, 3, dropout=0)
    net = net.to(device=device)

    # Hyper-parameters
    num_epochs = 50

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)

    # Track loss and perplexity
    train_loss, val_loss = [], []

    for i in range(num_epochs):
        # Track loss
        epoch_train_loss = 0
        epoch_val_loss = 0

        net.eval()

        # For each protein in the validation set
        for inputs, targets in val_loader:
            inputs = inputs.to(device).permute(1, 0)
            targets = targets.to(device).squeeze()

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Update loss
            epoch_val_loss += loss.detach().cpu().numpy()

        net.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(device).permute(1, 0)
            targets = targets.to(device).squeeze()

            # Forward pass
            outputs = net.forward(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward ()
            optimizer.step()

            # Update loss
            epoch_train_loss += loss.detach().cpu().numpy()

        # Save loss
        train_loss.append(epoch_train_loss / len(train_data))
        val_loss.append(epoch_val_loss / len(val_data))

        # Print loss every 10 epochs
        if i % 1 == 0:
            print(f'Epoch {i}, training loss: {train_loss[-1]}, validation loss: {val_loss[-1]}')
