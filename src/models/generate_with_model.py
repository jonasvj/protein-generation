#!/usr/bin/env python3
import os
import sys
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils import data
from Bio.pairwise2 import align
from collections import defaultdict
from Bio.SubsMat import MatrixInfo as matlist
from src.models import GruNet, LstmNet, WaveNet, WaveNetX, TransformerModel
from src.utils import (SequenceDataset, custom_collate_fn, get_repo_dir,
    get_device, set_seeds, load_pickle_obj)

def top_k_sampling(k, logits):
    """Function for top-k sampling."""
    # Make sure logits are non-negative
    logits = torch.exp(logits)
    # k-largest element
    k_elem = torch.topk(logits, k).values[-1].item()
    # Zero out smallest logits
    logits[logits < k_elem] = 0
    # Sample from multinomial
    prediction = torch.multinomial(logits, 1)

    return prediction

def generate_sequence(net, net_inputs, eos_idx, max_len, k):
    """Generates a sequence using a trained model."""
    prediction = torch.tensor(0)
    while prediction.item() != eos_idx and net_inputs[0].size(1) < max_len:
        # Get logits for next token
        logits = net(*net_inputs)['output'][0,:,-1]
        # Sample a token using top-k sampling
        prediction = top_k_sampling(k, logits)

        # Add prediction to input
        net_inputs[0] = torch.cat(
            (net_inputs[0], prediction.unsqueeze(1)), dim=1)

        # Add 1 to sequence length
        if net.model in ['gru', 'lstm']:
            net_inputs[1][0] += 1
        
    return net_inputs[0]


if __name__ == '__main__':
    seed = 42
    set_seeds(seed)
    repo_dir = get_repo_dir()
    device = get_device()
    model_name = sys.argv[1]

    # Get dictionary with model stats, arguments etc.
    stats_dict = load_pickle_obj(os.path.join(
        repo_dir, 'models/' + model_name + '_stats_dict.pickle'))
    
    model_args = stats_dict['model_args']
   
    # Load model 
    net = torch.load(os.path.join(
        repo_dir, 'models/' + model_name + '_pytorch_model.pt'),
        map_location=device)
    net.eval()
    
    # Get vocabulary dictionaries
    token_to_idx = defaultdict(lambda: 1, stats_dict['token_to_idx'])
    idx_to_token = defaultdict(lambda: '<UNK>', stats_dict['idx_to_token'])
    
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
    
    max_len = 278*2
    n_keywords = 5
    k = 1
    mutation_rate = 0.5
    context_props = np.arange(0.1, 1, 0.1)
    amino_acids = test_data.amino_acids
    gen_results = np.zeros((len(df_test), len(context_props)))
    mut_results = np.zeros((len(df_test), len(context_props)))
    gen_results[:] = np.nan
    mut_results[:] = np.nan
    
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
                                        max_len, k)
            
            # Convert generated sequence to actual AA sequence
            gen_seq = [idx_to_token[idx.item()] for idx in gen_seq[0]
                       if idx_to_token[idx.item()] in amino_acids]
            gen_seq = ''.join(gen_seq)

            # If only <EOS> were generated, add 1 random AA 
            if len(gen_seq) == context:
                continue
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
                - gen_align.start + 1)
            mut_results[i, j] = mut_align.score / (mut_align.end
                - mut_align.start + 1)
    
    # Save results
    np.save(os.path.join(
        repo_dir, 'models/' + model_name + '_similarity_scores_generation.npy'),
        gen_results)
    np.save(os.path.join(
        repo_dir, 'models/' + model_name + '_similarity_scores_mutation.npy'),
        mut_results)