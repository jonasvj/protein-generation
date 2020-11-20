#!/usr/bin/env python3
import os
import random
import subprocess
import numpy as np
import pandas as pd

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

    #Load data and rename columns
    raw_data = pd.read_csv(
        os.path.join(repo_dir, 'data/raw/uniprot_table_eukaryote.txt'), sep='\t')

    raw_data.columns = ['entry', 'entry_name', 'protein_names', 'organism_id',
                        'keywords', 'pfam', 'sequence', 'insulin']

    #Filter data
    filtered_data = raw_data[raw_data['insulin'] == 'Yes']
    not_insulin_data = raw_data[raw_data['insulin'] == 'No']
    not_insulin_data = not_insulin_data.sample(n = len(filtered_data))

    filtered_data = filtered_data.append(not_insulin_data)

    """
    organisms = list(set(list(not_insulin_data['organism_id'])))
    n = 50

    for Id in organisms:
        df_organism = not_insulin_data[not_insulin_data['organism_id'] == Id]
        if len(df_organism) >= n:
            df_organism = df_organism.sample(n=n)
        filtered_data = filtered_data.append(df_organism)
    """

    # Keep only entry ID and sequence
    filtered_data = filtered_data[['entry', 'keywords', 'insulin', 'sequence']]
  
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
    
    