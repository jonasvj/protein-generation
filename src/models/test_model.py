#!/usr/bin/env python3
import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils import data
from collections import defaultdict
from src.models import GruNet, LstmNet, WaveNet, WaveNetX, TransformerModel
from src.utils import (SequenceDataset, custom_collate_fn, get_repo_dir,
    get_device, set_seeds, load_pickle_obj)

def get_ppl_emb(test_loader, net, pad_idx, device):
    """Gets mean perplexities and embeddings of test sequences"""
    perplexities = list()
    emb_1 = list()
    emb_2 = list()
    i = 0
    for inputs, targets, lengths in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if net.model in ['transformer', 'wavenet']:
            net_inputs = [inputs]
        elif net.model in ['gru', 'lstm']:
            net_inputs = [inputs, lengths]
        elif net.model == 'wavenetX':
            n_globals = net.n_globals
            global_inputs = inputs[:,:n_globals]
            inputs = inputs[:,n_globals:]
            targets = targets[:,n_globals:]
            net_inputs = [inputs, global_inputs]

        # Forward pass
        output = net(*net_inputs)
        outputs = output['output']
        emb_1.append(output['emb_1'].detach().cpu())
        emb_2.append(output['emb_2'].detach().cpu())

        # Calculate perplexity
        loss = F.cross_entropy(outputs, targets, reduction='mean',
            ignore_index=pad_idx).item()
        perplexities.append(np.exp(loss))
    
    return perplexities, emb_1, emb_2

def test_model(stats_dict, net, df_test, device):
    """Tests a model"""
    seed = 42
    set_seeds(seed)
    model_args = stats_dict['model_args']

    # Get vocabulary dictionaries
    token_to_idx = defaultdict(lambda: 1, stats_dict['token_to_idx'])
    idx_to_token = defaultdict(lambda: '<UNK>', stats_dict['idx_to_token'])
        
    if model_args['kw_method'] == 'sample':
        new_rows = list()
        for entry in df_test.entry.unique():
            temp_df = df_test.loc[df_test.entry == entry]
            sample = temp_df.sample(n=1, axis=0).to_numpy().tolist()[0]
            new_rows.append(sample)
        
        df_test = pd.DataFrame(new_rows, columns=df_test.columns)
    
    test_data = SequenceDataset(entries=df_test['entry'],
                                sequences=df_test['sequence'],
                                keywords=df_test[['organism', 'bp', 'cc','mf',
                                    'insulin']].to_numpy(),
                                kw_method='permute',
                                token_to_idx=token_to_idx)
    
    test_loader = data.DataLoader(test_data,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=4,
                                 collate_fn=custom_collate_fn)
    
    # Calculate perplexities and get embeddings
    perplexities, emb_1, emb_2 = get_ppl_emb(test_loader, net,
                                             model_args['pad_idx'],
                                             device)

    # Add perplexities and embeddings to data frame
    df_test['perplexity'] = perplexities

    df_emb_1 = pd.DataFrame(data=torch.cat(emb_1, dim=0).numpy())
    df_emb_1.columns = ['emb_1_' + str(i) for i in range(emb_1[0].shape[1])]

    df_emb_2 = pd.DataFrame(data=torch.cat(emb_2, dim=0).numpy())
    df_emb_2.columns = ['emb_2_' + str(i) for i in range(emb_2[0].shape[1])]

    df_test = pd.concat([df_test, df_emb_1, df_emb_2], axis=1)
    
    return df_test

if __name__ == '__main__':
    repo_dir = get_repo_dir()
    device = get_device()
    model_name = sys.argv[1]

    # Get dictionary with model stats, arguments etc.
    stats_dict = load_pickle_obj(
        os.path.join(repo_dir, 'models/' + model_name + '_stats_dict.pickle'))

    # Load model
    net = torch.load(os.path.join(
        repo_dir,'models/' + model_name + '_pytorch_model.pt'),
        map_location=device)
    net.eval()
        
    # Load test data
    df_test = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/test_data.txt'),
        sep='\t', dtype='str')
    
    # Test model
    test_results = test_model(stats_dict, net, df_test, device)

    # Save test results
    test_results.to_csv(os.path.join(
        repo_dir, 'models/' + model_name + '_test_results.csv'))