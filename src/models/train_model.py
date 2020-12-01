#!/usr/bin/env python3
import os
import sys
import time
import torch
import pickle
import numpy as np
import pandas as pd
from gru import GruNet
from lstm import LstmNet
from wavenet import WaveNet
from torch.utils import data
from wavenetX import WaveNetX
from transformer import TransformerModel
from utils import (SequenceDataset, custom_collate_fn, train_model_cli, 
    get_repo_dir, get_device, set_seeds)

if __name__ == '__main__':
    total_start = time.time()
    all_model_args = train_model_cli()
    model_args = all_model_args.copy()
    model = model_args.pop('model')
    mb_size = model_args.pop('mb_size')
    learning_rate = model_args.pop('learning_rate')
    weight_decay = model_args.pop('weight_decay')
    num_epochs = model_args.pop('epochs')
    kw_method = model_args.pop('kw_method')
    include_non_insulin = model_args.pop('include_non_insulin')
    non_insulin_frac = model_args.pop('non_insulin_frac')
    include_reverse = model_args.pop('include_reverse')
    seed = model_args.pop('seed')
    output_file = model_args.pop('output_file')
    repo_dir = get_repo_dir()
    device = get_device()
    set_seeds(seed)
    
    # Load data
    df_train = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/train_data.txt'),
        sep='\t', dtype='str')
    df_val = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/val_data.txt'),
        sep='\t', dtype='str')
    
    if include_non_insulin is False:
        # Exclude non-insulin proteins 
        df_train = df_train.loc[df_train.insulin == 'Yes']
        df_train.reset_index(drop=True, inplace=True)
    elif include_non_insulin is True and non_insulin_frac != 1:
        # Sample subset of non-insulin proteins
        insulin_data = df_train.loc[df_train.insulin == 'Yes']
        non_insulin_data = df_train.loc[df_train.insulin == 'No']
        sampled_entries = np.random.choice(
            non_insulin_data.entry.unique(),
            size=int(len(non_insulin_data.entry.unique())*non_insulin_frac),
            replace=False)
        non_insulin_data = non_insulin_data.loc[non_insulin_data.entry.isin(
        sampled_entries)]
        df_train = insulin_data.append(non_insulin_data)
        df_train.reset_index(drop=True, inplace=True)
    
    # Create data sets
    train_data = SequenceDataset(entries=df_train['entry'],
                                 sequences=df_train['sequence'],
                                 keywords=df_train[['organism', 'bp', 'cc',
                                 'mf', 'insulin']].to_numpy(),
                                 kw_method=kw_method,
                                 include_rev=include_reverse)

    val_data = SequenceDataset(entries=df_val['entry'],
                               sequences=df_val['sequence'], 
                               keywords=df_val[['organism', 'bp', 'cc',
                               'mf', 'insulin']].to_numpy(),
                               kw_method=kw_method,
                               token_to_idx=train_data.token_to_idx)
    
    # Maximum length of keywords + sequence
    max_len = 0
    for input_, target in train_data:
        max_len = max(max_len, len(input_))
    
    # Prepare data loaders
    train_loader = data.DataLoader(train_data,
                                   batch_size=mb_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=4,
                                   collate_fn=custom_collate_fn)
    
    val_loader = data.DataLoader(val_data,
                                 batch_size=mb_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=4,
                                 collate_fn=custom_collate_fn)
    
    all_model_args['n_globals'] = 5 # Number of keywords
    all_model_args['n_train'] = len(train_data)
    all_model_args['n_val'] = len(val_data)
    all_model_args['n_outputs'] = len(train_data.amino_acids) + 3
    model_args['n_tokens'] = len(train_data.token_to_idx)
    model_args['pad_idx'] = train_data.token_to_idx['<PAD>']

    # Choose network model
    if model == 'gru':
        net = GruNet(**model_args)
    elif model == 'lstm':
        net = LstmNet(**model_args)
    elif model == 'transformer':
        model_args['max_len'] = max_len
        net = TransformerModel(**model_args)
    elif model == 'wavenet':
        use_global_input = model_args.pop('X')
        if use_global_input is True:
            n_globals = all_model_args['n_globals']
            model_args['n_globals'] = n_globals
            model_args['n_outputs'] = all_model_args['n_outputs']
            net = WaveNetX(**model_args)
        else:
            net = WaveNet(**model_args)
    
    net = net.to(device=device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=model_args['pad_idx'],
        reduction='mean')
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # Track loss and perplexity
    train_loss, val_loss = [], []
    train_perplexity, val_perplexity = [], []
    learning_rates = []
    
    for i in range(num_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_perplexity = 0
        epoch_val_perplexity = 0
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Training
        train_start = time.time()
        net.train()
        for inputs, targets, lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if net.model in ['transformer', 'wavenet']:
                net_inputs = [inputs]
            elif net.model in ['gru', 'lstm']:
                net_inputs = [inputs, lengths]
            elif net.model == 'wavenetX':
                global_inputs = inputs[:,:n_globals]
                inputs = inputs[:,n_globals:]
                targets = targets[:,n_globals:]
                net_inputs = [inputs, global_inputs]

            # Forward pass
            output = net(*net_inputs)
            outputs = output['output']

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            loss = loss.detach().cpu().numpy()
            epoch_train_loss += loss
            epoch_train_perplexity += np.exp(loss)
        
        train_end = time.time()
        
        # Validation
        val_start = time.time()
        net.eval()
        for inputs, targets, lengths in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if net.model in ['transformer', 'wavenet']:
                net_inputs = [inputs]
            elif net.model in ['gru', 'lstm']:
                net_inputs = [inputs, lengths]
            elif net.model == 'wavenetX':
                global_inputs = inputs[:,:n_globals]
                inputs = inputs[:,n_globals:]
                targets = targets[:,n_globals:]
                net_inputs = [inputs, global_inputs]
            
            # Forward pass
            output = net(*net_inputs)
            outputs = output['output']

            loss = criterion(outputs, targets).detach().cpu().numpy()

            # Update loss
            epoch_val_loss += loss
            epoch_val_perplexity += np.exp(loss)
        
        val_end = time.time()

        # Save loss and perplexity
        train_loss.append(epoch_train_loss
            / np.ceil(all_model_args['n_train'] / mb_size))
        val_loss.append(epoch_val_loss
            / np.ceil(all_model_args['n_val'] / mb_size))

        train_perplexity.append(epoch_train_perplexity
            / np.ceil(all_model_args['n_train'] / mb_size))
        val_perplexity.append(epoch_val_perplexity
            / np.ceil(all_model_args['n_val'] / mb_size))
        
        # Update learning rate
        scheduler.step(val_perplexity[-1])

        # Print loss
        if i % 1 == 0:
            print('Epoch {}\nTraining loss: {}, Validation loss: {}'.format(
                i, train_loss[-1], val_loss[-1]))
            print('Training perplexity: {}, Validation perplexity: {}'.format(
                train_perplexity[-1], val_perplexity[-1]))
            print('Learning rate: {}'.format(learning_rates[-1]))
            print('Training time: {}, Validation time: {}\n'.format(
                round(train_end - train_start), round(val_end - val_start)))
    
    torch.save(net, os.path.join(repo_dir, 'models/' + output_file + '.pt'))

    # Convert defaultdicts to regular dicts that can be pickled
    token_to_idx = dict(train_data.token_to_idx)
    idx_to_token = dict(train_data.idx_to_token)
    all_model_args.update(model_args)
    total_end = time.time()

    stats_dict = dict()
    stats_dict['idx_to_token'] = idx_to_token
    stats_dict['token_to_idx'] = token_to_idx
    stats_dict['train_loss'] = train_loss
    stats_dict['val_loss'] = val_loss
    stats_dict['train_perplexity'] = train_perplexity
    stats_dict['val_perplexity'] = val_perplexity
    stats_dict['learning_rates'] = learning_rates
    stats_dict['model_args'] = all_model_args
    stats_dict['total_time'] = round(total_end - total_start)
    stats_dict['n_parameters'] = sum(
        p.numel() for p in net.parameters() if p.requires_grad)

    out_file = open(
        os.path.join(repo_dir, 'models/' + output_file + '.pickle'), 'wb')
    pickle.dump(stats_dict, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    out_file.close()