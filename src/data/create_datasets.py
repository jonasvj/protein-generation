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


if __name__ == '__main__':

    raw_data = pd.read_csv('../../data/raw/uniprot_table.txt', sep='\t')
    raw_data.columns = ['entry', 'entry_name', 'protein_names', 'organism_id',
                        'keywords', 'pfam', 'sequence']

    # Filter data 
    # (for example by organism, protein family, keywords)
    organism_filter = [559292]
    protein_fam_filter = []

    filtered_data = raw_data.copy()
    if len(organism_filter) != 0:
        filtered_data = filtered_data[filtered_data['organism_id'].isin(
            organism_filter)]
    
    if len(protein_fam_filter) != 0:
        filtered_data = filtered_data[filtered_data['pfam'].isin(
            protein_fam_filter)]
    
    data_set = SequenceDataset(filtered_data['sequence'])

    # Split data sets into train, validation and test
    train_data, val_data, test_data = data.random_split(data_set,
                                                        [4000, 1721, 1000])
    
    # Save data sets
    torch.save(train_data, '../../data/processed/train_data.pt')
    torch.save(val_data, '../../data/processed/val_data.pt')
    torch.save(test_data, '../../data/processed/test_data.pt')
    torch.save(data_set, '../../data/processed/data.pt')

