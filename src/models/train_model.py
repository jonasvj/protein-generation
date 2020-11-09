#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
from torch.utils import data

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
            inputs.append(encoded_sequence[:-1])
            targets.append(encoded_sequence[1:])
        
        return inputs, targets

    def __len__(self):
        # Return the number of sequences
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve inputs and targets at the given index
        return self.inputs[idx], self.targets[idx]

train_data = torch.load('../../data/processed/train_data.pt')
print(train_data[0])

