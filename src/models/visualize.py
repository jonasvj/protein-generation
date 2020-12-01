#!/usr/bin/env python3
import os
import sys
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils import data
from wavenetX import WaveNetX
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from collections import defaultdict
from utils import SequenceDataset, custom_collate_fn, get_repo_dir, get_device, set_seeds

def plot_learning_curves(stats_dict):
    """Plot learning curves"""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    # x-axis
    ax1.set_xlabel('Epoch')
    ax1.tick_params(axis='x')
    # y-axis for loss
    ax1.set_ylabel('Loss', color='C1')
    ax1.tick_params(axis='y', labelcolor='C1')
    ax1.plot(stats_dict['train_loss'][1:], c='C1', label='Training loss')
    ax1.plot(stats_dict['val_loss'][1:], c='C1', linestyle='--',
        label='Validation loss')
    # Second y-axis for perplexity
    ax2.set_ylabel('Perplexity', c='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    ax2.plot(stats_dict['train_perplexity'][1:], c='C0',
        label='Training perplexity')
    ax2.plot(stats_dict['val_perplexity'][1:], c='C0', linestyle='--',
        label='Validation perplexity')
    # Title, legend and grid
    ax1.set_title("Training metrics")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=0)
    ax1.grid(True)

    return fig

def calculate_perplexities(n_test, test_loader, net):
    perplexities = np.zeros((n_test,))
    emb_mean = list()
    emb_max = list()
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
            emb_mean.append(output['emb_1'].detach().cpu())
            emb_max.append(output['emb_2'].detach().cpu())

            # Loss
            loss = F.cross_entropy(outputs, targets, reduction='none',
                ignore_index=model_args['pad_idx']).detach()
            perplexity = torch.exp(loss)
            #mean_perplexity = perplexity.sum(1)/(loss!=0).sum(1)
            #perplexities[i:i+inputs.shape[0]] = mean_perplexity.cpu().numpy()
            #i += inputs.shape[0]

            loss = F.cross_entropy(outputs, targets).detach().cpu().numpy()
            perplexities[i] = float(np.exp(loss))
            i += 1
            #print(float(np.exp(loss)))
    
    return perplexities, emb_mean, emb_max
    

if __name__ == '__main__':
    set_seeds(42)

    model_name = sys.argv[1]
    repo_dir = get_repo_dir()
    device = get_device()
    device='cpu'

    # Get dictionary with model stats, arguments etc.
    stats_dict_file = open(
        os.path.join(repo_dir, 'models/' + model_name + '.pickle'), 'rb')
    stats_dict = pickle.load(stats_dict_file)
    stats_dict_file.close()
    model_args = stats_dict['model_args']
   
    # Load model 
    net = torch.load(os.path.join(repo_dir, 'models/' + model_name + '.pt'))
    net = net.to(device)
    net.eval()
    
    # Get vocabulary dictionaries
    token_to_idx = defaultdict(lambda: 1, stats_dict['token_to_idx'])
    idx_to_token = defaultdict(lambda: '<UNK>', stats_dict['idx_to_token'])

    # Create and save learning curves
    learning_curves = plot_learning_curves(stats_dict)
    learning_curves.savefig(
        os.path.join(repo_dir, 'models/' + model_name + '_l_curves.pdf'))
    
    # Load test data
    df_test = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/test_data.txt'),
        sep='\t', dtype='str')
    test_data = SequenceDataset(entries=df_test['entry'],
                               sequences=df_test['sequence'], 
                               keywords=df_test[
                                   ['organism', 'bp', 'cc','mf']].to_numpy(),
                               kw_method=model_args['kw_method'],
                               token_to_idx=token_to_idx)
    test_loader = data.DataLoader(test_data,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=4,
                                 collate_fn=custom_collate_fn)
    
    # Calculate perplexities
    perplexities, emb_mean, emb_max = calculate_perplexities(len(test_data), test_loader, net)
    emb_mean = torch.cat(emb_mean, dim=0).numpy()
    emb_max = torch.cat(emb_max, dim=0).numpy()
    print(emb_max.shape)
    
    df_test['perplexity'] = perplexities
    X_emb_max = TSNE(n_components=2).fit_transform(emb_mean)

    fig, ax = plt.subplots(figsize=(8,4))
    plt.scatter(X_emb_max[:, 0], X_emb_max[:, 1], c=df_test.mf.factorize()[0], s=2, alpha=0.2)
    fig.savefig(
        os.path.join(repo_dir, 'models/' + model_name + '_emb_plot.pdf'))

    # Box plot of perplexities
    fig, ax = plt.subplots(figsize=(8,4))
    plt.suptitle('')
    df_test.boxplot(column=['perplexity'], by='cc', ax=ax)
    fig.savefig(
        os.path.join(repo_dir, 'models/' + model_name + '_box_plots.pdf'))


    # Visualize protein embedding

    

    
    """
    # Do protein generation (calculate BLEU)
    # Number of keywords and amino acids to start generation with
    n_kws = 4
    n_aas = 1
    max_len = df_test.sequence.map(len).max()
    seqs = np.array([list(seq.ljust(max_len, '0')) for seq in df_test.sequence], dtype='object')
    seqs[seqs == '0'] = '<PAD>'

    kw_seqs = np.hstack(
        df_test[['organism', 'bp', 'cc','mf']].to_numpy(dtype='object'),
        seqs)
    
    kw_seqs = kw_seqs[:,:n_kws+n_aas]
    """


    









