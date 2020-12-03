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
from Bio.pairwise2 import align
from Bio.SubsMat import MatrixInfo as matlist
from utils import SequenceDataset, custom_collate_fn, get_repo_dir, get_device, set_seeds

def plot_learning_curves(stats_dict):
    """Plot learning curves"""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    # x-axis
    ax1.set_xlabel('Epoch')
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
    ax2.grid(True)

    return fig

def get_ppl_emb(test_loader, net, pad_idx):
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

if __name__ == '__main__':
    set_seeds(42)

    model_name = sys.argv[1]
    repo_dir = get_repo_dir()
    device = get_device()
    device = 'cuda:1'

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
    
    # Calculate perplexities
    perplexities, emb_1, emb_2 = get_ppl_emb(test_loader, net,
                                             model_args['pad_idx'])
    df_test['perplexity'] = perplexities
    emb_1 = torch.cat(emb_1, dim=0).numpy()
    emb_2 = torch.cat(emb_2, dim=0).numpy()
    
    # Project embeddings to 2-dimensional space
    X_emb_1 = TSNE(n_components=2).fit_transform(emb_1)
    X_emb_2 = TSNE(n_components=2).fit_transform(emb_2)

    # Plot sequence embedding
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    ax1.scatter(X_emb_1[:, 0], X_emb_1[:, 1], c=df_test.mf.factorize()[0],
        s=2, alpha=0.8)
    ax2.scatter(X_emb_2[:, 0], X_emb_2[:, 1], c=df_test.mf.factorize()[0],
        s=2, alpha=0.8)
    fig.savefig(
        os.path.join(repo_dir, 'models/' + model_name + '_emb_plot.pdf'))
    
    # Box plot of perplexities
    fig, ax = plt.subplots(figsize=(8,4))
    df_test.boxplot(column=['perplexity'], by='cc', ax=ax)
    fig.savefig(
        os.path.join(repo_dir, 'models/' + model_name + '_box_plot.pdf'))
    
    # Protein generation

    def generate_sequence(net, net_inputs, eos_idx, max_len, k):
        """Generates a sequence using a trained model"""
        prediction = torch.tensor(0)
        while prediction.item() != eos_idx and net_inputs[0].size(1) < max_len:
            # Get logits for next token
            logits = net(*net_inputs)['output'][0,:,-1]
            # Sample a token using top-k sampling
            prediction = top_k_sampling(k, logits)

            # Add prediction to input
            net_inputs[0] = torch.cat(
                (net_inputs[0], prediction.unsqueeze(1)), dim=1)

            if net.model in ['gru', 'lstm']:
                net_inputs[1][0] += 1
        
        return net_inputs[0]


    def top_k_sampling(k, logits):
        # Make sure logits are non-negative
        logits = torch.exp(logits)
        # k-largest element
        k_elem = torch.topk(logits, k).values[-1].item()
        # Zero out smallest logits
        logits[logits < k_elem] = 0
        # Sample from multinomial
        prediction = torch.multinomial(logits, 1)

        return prediction
    """
    max_len = 278
    n_keywords = 5
    k = 3
    mutation_rate = 0.5
    context_props = np.arange(0.1, 1, 0.1)
    amino_acids = test_data.amino_acids
    gen_results = np.zeros((len(df_test), len(context_props)))
    mut_results = np.zeros((len(df_test), len(context_props)))
    
    # Do protein generation
    for i, (inputs, targets, lengths) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Convert sequence to actual AA sequence
        true_seq = [idx_to_token[idx.item()] for idx in inputs[0]]
        true_seq = true_seq[n_keywords:]
        seq_len = len(true_seq)
        max_gen_len = seq_len
   
        # Generate protein using increasing context
        for j, prop in enumerate(context_props):
            context = int(seq_len*prop)

            # Prepare network input
            if net.model in ['transformer', 'wavenet']:
                net_inputs = [inputs[:,:n_keywords+context]]
            elif net.model in ['gru', 'lstm']:
                net_inputs = [inputs[:,:n_keywords+context],
                              [n_keywords+context]]
            elif net.model == 'wavenetX':
                global_inputs = inputs[:,:n_keywords]
                net_inputs = [inputs[:,n_keywords:n_keywords+context],
                              global_inputs]
                max_gen_len -= n_keywords
            
            # Generate sequence with model
            gen_seq = generate_sequence(net, net_inputs, token_to_idx['<EOS>'],
                                        max_gen_len, k)
            
            # Convert generated sequence to actual AA sequence
            gen_seq = [idx_to_token[idx.item()] for idx in gen_seq[0]
                       if idx_to_token[idx.item()] in amino_acids]
            gen_seq = ''.join(gen_seq)

            # If only <EOS> were generated, add 1 random AA 
            if len(gen_seq) == context:
                gen_seq += random.choice(amino_acids)
            
            # Generate mutated sequence
            mutation_idx = random.sample(range(context, seq_len),
                int(mutation_rate*(seq_len-context)))
            mutated_seq = true_seq[:]
            for idx in mutation_idx:
                mutated_seq[idx] = random.choice(
                    amino_acids[amino_acids != mutated_seq[idx]])
        
            mutated_seq = ''.join(mutated_seq)
            seq = ''.join(true_seq)

            # Generate alignments
            gen_align = align.globalds(seq[context:], gen_seq[context:],
                matlist.blosum62, -0.5, -0.1, one_alignment_only=True)[0]
            mut_align = align.globalds(seq[context:], mutated_seq[context:],
                matlist.blosum62, -0.5, -0.1, one_alignment_only=True)[0]

            gen_results[i, j] = gen_align.score / (gen_align.end
                - gen_align.start)
            mut_results[i, j] = mut_align.score / (mut_align.end
                - mut_align.start)
    
    gen_mean = gen_results.mean(axis=0)
    gen_std = gen_results.std(axis=0)
    mut_mean = mut_results.mean(axis=0)
    mut_std = mut_results.std(axis=0)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.errorbar(context_props, gen_mean, yerr=gen_std, alpha=.75, fmt='.:',
        capsize=3, capthick=1, c='C0', label='Generated')
    ax.errorbar(context_props, mut_mean, yerr=mut_std, alpha=.75, fmt='.:',
        capsize=3, capthick=1, c='C1', label='Mutation baseline')
    ax.fill_between(context_props, gen_mean - gen_std, gen_mean + gen_std, 
        alpha=.25, color='C0')
    ax.fill_between(context_props, mut_mean - mut_std, mut_mean + mut_std, 
        alpha=.25, color='C1')
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=0)
    ax.grid(True)
    ax.set_xlabel('Context proportion')
    ax.set_ylabel('Similarity')
    fig.savefig(
        os.path.join(repo_dir, 'models/' + model_name + '_sim_plot.pdf'))

    np.save(
        os.path.join(repo_dir, 'models/' + model_name + '_gen_results.npy'),
        gen_results)

    np.save(
        os.path.join(repo_dir, 'models/' + model_name + '_mut_results.npy'),
        mut_results)
    """






