#!/usr/bin/env python3
import os
import random
import subprocess
import numpy as np
import pandas as pd

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

def random_split(df, props, seed=42):
    """Randomly splits data frame into defined proportions.
    
    Args:
        df: Data frame to split.
        props: List of proportions.
        seed: Seed for random split.
    
    Returns:
        List of data frames of specified proportions.
    """
    # Data splits
    n = len(df)
    splits = np.cumsum([int(n*prop) for prop in props])

    # Lists of (random) indexes of specificied proportions 
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    idxs = np.split(idxs, splits[:-1])

    # Split data frame 
    dfs = [df.iloc[idx,:] for idx in idxs]
    print([len(df1) for df1 in dfs])

    return dfs

if __name__ == '__main__':

    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()
    
    raw_data = pd.read_csv(
        os.path.join(repo_dir, 'data/raw/uniprot_table.txt'), sep='\t')
    
    # Rename columns 
    raw_data.columns = ['entry', 'entry_name', 'protein_names', 'organism_id',
                        'keywords', 'pfam', 'sequence']

    # Filter data 
    organism_filter = [559292]
    protein_fam_filter = []
    filtered_data = raw_data.copy()

    if len(organism_filter) != 0:
        filtered_data = filtered_data[filtered_data['organism_id'].isin(
            organism_filter)]
    
    if len(protein_fam_filter) != 0:
        filtered_data = filtered_data[filtered_data['pfam'].isin(
            protein_fam_filter)]
    
    # Keep only entry ID and sequence
    filtered_data = filtered_data[['entry','sequence']]
  
    # Split data into train, validation and test
    train_df, val_df, test_df = random_split(filtered_data, [0.8, 0.1, 0.1])

    # Save data sets
    train_df.to_csv(
        os.path.join(repo_dir, 'data/processed/train_data.txt'),
        sep='\t', index=False)
    val_df.to_csv(
        os.path.join(repo_dir, 'data/processed/val_data.txt'),
        sep='\t', index=False)
    test_df.to_csv(
        os.path.join(repo_dir, 'data/processed/test_data.txt'),
        sep='\t', index=False)