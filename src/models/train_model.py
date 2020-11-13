#!/usr/bin/env python3
import os 
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from gru_network import GRUModel
import subprocess

class SequenceDataset(data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.amino_acids = np.unique(list(''.join(self.sequences)))

        self.aa_to_idx = {self.amino_acids[i]: i
                          for i in range(len(self.amino_acids))}

        self.idx_to_aa = {i: self.amino_acids[i]
                          for i in range(len(self.amino_acids))}

        self.inputs, self.targets = self.encode_sequences()

    def encode_sequences(self):
        inputs = list()
        targets = list()
        for sequence in self.sequences:
            encoded_sequence = [self.aa_to_idx[aa] for aa in sequence]

            input_ = torch.tensor(
                encoded_sequence[:-1]).unsqueeze(1).to(device='cuda')
            target = torch.tensor(encoded_sequence[1:]).to(device='cuda')

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
    
    # Load data
    df_train = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/train_data.txt'), sep='\t')
    df_val = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/val_data.txt'), sep='\t')

    train_data = SequenceDataset(df_train['sequence'])
    val_data = SequenceDataset(df_train['sequence'])

    # Choose network model
    n_amino_acids = len(train_data.amino_acids)
    net = GRUModel(n_amino_acids, n_amino_acids, 500, 3, dropout=0)
    if torch.cuda.is_available():
        net.cuda()

    # Hyper-parameters
    num_epochs = 50

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)

    # Track loss and perplexity
    train_loss, val_loss = [], []

    for i in range(num_epochs):
        print(i)
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        net.eval()

        # For each protein in the validation set
        for inputs, targets in val_data:

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Update loss
            epoch_validation_loss += loss.detach().cpu().numpy()

        net.train()

        for inputs, targets in train_data:

            # Forward pass
            outputs = net.forward(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward ()
            optimizer.step()

            # Update loss
            epoch_training_loss += loss.detach().cpu().numpy()

        # Save loss
        train_loss.append(epoch_training_loss / len(train_data))
        val_loss.append(epoch_validation_loss / len(val_data))

        # Print loss every 10 epochs
        if i % 1 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
