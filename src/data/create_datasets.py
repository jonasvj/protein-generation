#!/usr/bin/env python3
import os
import random
import subprocess
import numpy as np
import pandas as pd

def random_split(data, props, seed=42):
    """Randomly splits data frame into defined proportions.
    
    Args:
        data: np.array of data to split
        props: List of proportions.
        seed: Seed for random split.
    
    Returns:
        List of data frames of specified proportions.
    """
    # Data splits
    n = len(data)
    splits = np.cumsum([int(n*prop) for prop in props])

    # Lists of (random) indexes of specificied proportions 
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    idxs = np.split(idxs, splits[:-1])

    # Split data frame 
    data_splitted = [data[idx] for idx in idxs]

    return data_splitted

if __name__ == '__main__':
    np.random.seed(42)

    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

    # Load data and rename columns
    raw_data = pd.read_csv(
        os.path.join(repo_dir, 'data/interim/uniprot_table_tidy.txt'),
        sep='\t')

    raw_data.columns = ['entry', 'organism', 'bp', 'cc', 'mf',
                        'pfam', 'sequence', 'insulin']

    # Filter data
    filtered_data = raw_data[raw_data['insulin'] == 'Yes']
    non_insulin_data = raw_data[raw_data['insulin'] == 'No']
    sampled_entries = np.random.choice(non_insulin_data.entry.unique(),
                                       size=len(filtered_data.entry.unique()),
                                       replace=False)

    non_insulin_data = non_insulin_data.loc[non_insulin_data['entry'].isin(
        sampled_entries)]
    
    filtered_data = filtered_data.append(non_insulin_data)

    # Keep only entry ID and sequence
    filtered_data = filtered_data[['entry', 'organism', 'bp', 'cc', 'mf',
                                   'insulin', 'sequence']]
    
    # Split data into train, validation and test
    train_entries, val_entries, test_entries = random_split(
        filtered_data.entry.unique(), [0.8, 0.1, 0.1])

    train_df = filtered_data.loc[filtered_data['entry'].isin(train_entries)]
    val_df = filtered_data.loc[filtered_data['entry'].isin(val_entries)]
    test_df = filtered_data.loc[filtered_data['entry'].isin(test_entries)]

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
    
    