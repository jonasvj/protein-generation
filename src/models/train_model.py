#!/usr/bin/env python3
import os 
import sys
import time
import torch
import pickle
import subprocess
import numpy as np
import pandas as pd
from gru import GruNet
from lstm import LstmNet
from torch.utils import data
from collections import defaultdict
from transformer import TransformerModel

class SequenceDataset(data.Dataset):
    def __init__(self, entries, sequences, keywords, kw_method='permute',
    include_rev=False, token_to_idx=None):
        super(SequenceDataset, self).__init__()
        self.amino_acids = np.unique(list(''.join(sequences)))
        self.keywords = np.unique(keywords)
        self.tokens = np.hstack((['<PAD>', '<UNK>', '<EOS>'],
                                 self.amino_acids,
                                 self.keywords))
        
        self.token_to_idx, self.idx_to_token = self.token_idx_map(token_to_idx)

        self.inputs, self.targets = self.encode_sequences(entries,
                                                          sequences,
                                                          keywords,
                                                          kw_method,
                                                          include_rev)
    
    def token_idx_map(self, token_to_idx):
        if token_to_idx is None:
            token_to_idx = defaultdict(lambda: 1)
            idx_to_token = defaultdict(lambda: '<UNK>')
            for idx, token in enumerate(self.tokens):
                token_to_idx[token] = idx
                idx_to_token[idx] = token
        else:
            idx_to_token = defaultdict(lambda: '<UNK>')
            for token, idx in token_to_idx.items():
                idx_to_token[idx] = token
        
        return token_to_idx, idx_to_token
    
    def add_reverse(self, entries, sequences, keywords):
        rev_sequences = np.empty(sequences.shape)
        rev_entries = np.empty(entries.shape)

        for i, sequence in enumerate(sequences):
            rev_sequences[i] = sequence[::-1]
            rev_entries[i] = entries[i] + '_reverse'
        
        sequences = np.hstack((sequences, rev_sequences))
        entries = np.hstack((entries, rev_entries))
        keywords = np.vstack((keywords, keywords))

        return entries, sequences, keywords       
    
    def encode_sequences(self, entries, sequences, keywords, kw_method,
    include_rev):
        inputs = list()
        targets = list()

        if include_rev is True:
            entries, sequences, keywords = self.add_reverse(entries,
                                                            sequences,
                                                            keywords)
        
        if kw_method == 'merge':
            
            for entry in entries.unique():
                entry_idxs = np.where(entries == entry)[0]

                # Keywords
                entry_keywords = keywords[entry_idxs,:].flatten('F')
                # Unique keywords preserving order
                entry_keywords = entry_keywords[np.sort(
                    np.unique(entry_keywords, return_index=True)[1])]
                encoded_keywords = [self.token_to_idx[kw]
                                    for kw in entry_keywords]
                
                # Sequence 
                entry_sequence = sequences[entry_idxs[0]]
                encoded_sequence = [self.token_to_idx[token]
                                    for token in entry_sequence]
                encoded_sequence.append(self.token_to_idx['<EOS>'])
                
                # Use keywords and sequence as input
                combined_input = encoded_keywords + encoded_sequence
                
                input_ = combined_input[:-1]
                target = combined_input[1:]
                inputs.append(input_)
                targets.append(target)
        
        if kw_method == 'sample':

            for entry in entries.unique():
                entry_idxs = np.where(entries == entry)[0]

                # Keywords
                entry_keywords = keywords[entry_idxs,:]

                # Sample a keyword in each category
                entry_keywords = [np.random.choice(entry_keywords[:,j], 1)[0]
                                  for j in range(entry_keywords.shape[1])]
                
                encoded_keywords = [self.token_to_idx[kw]
                                    for kw in entry_keywords]
                
                # Sequence 
                entry_sequence = sequences[entry_idxs[0]]
                encoded_sequence = [self.token_to_idx[token]
                                    for token in entry_sequence]
                encoded_sequence.append(self.token_to_idx['<EOS>'])
                
                # Use keywords and sequence as input
                combined_input = encoded_keywords + encoded_sequence
                
                input_ = combined_input[:-1]
                target = combined_input[1:]
                inputs.append(input_)
                targets.append(target)

        elif kw_method == 'permute':

            for i in range(len(sequences)):
                # Keywords
                entry_keywords = keywords[i,:]
                encoded_keywords = [self.token_to_idx[kw]
                                    for kw in entry_keywords]

                # Sequence
                entry_sequence = sequences[i]
                encoded_sequence = [self.token_to_idx[token]
                                    for token in entry_sequence]
                encoded_sequence.append(self.token_to_idx['<EOS>'])

                # Use keywords and sequence as input
                combined_input = encoded_keywords + encoded_sequence
                
                input_ = combined_input[:-1]
                target = combined_input[1:]
                inputs.append(input_)
                targets.append(target)
        
        return inputs, targets

    def __len__(self):
        # Return the number of sequences
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve inputs and targets at the given index
        return self.inputs[idx], self.targets[idx]

def custom_collate_fn(batch):
    """How to return a batch from the data loader"""
    # Sort batch by sequence lengths
    batch.sort(key=lambda seq: len(seq[0]), reverse=True)
    inputs, targets = zip(*batch)
    lengths = [len(input_) for input_ in inputs]

    # Tensors are of shape (batch, max_seq_length)
    input_tensor = torch.zeros((len(inputs), max(lengths))).long()
    target_tensor = torch.zeros((len(inputs), max(lengths))).long()

    for i, length in enumerate(lengths):
            input_tensor[i, 0:length] = torch.LongTensor(inputs[i][:])
            target_tensor[i, 0:length] = torch.LongTensor(targets[i][:])
            
    return input_tensor, target_tensor, lengths


if __name__ == '__main__':

    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Load data
    df_train = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/train_data.txt'),
        sep='\t', dtype='str')
    
    df_val = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/val_data.txt'),
        sep='\t', dtype='str')
    
    train_data = SequenceDataset(entries=df_train['entry'][:500],
                                 sequences=df_train['sequence'][:500],
                                 keywords=df_train[['organism', 'bp', 'cc',
                                 'mf', 'insulin']].to_numpy(),
                                 kw_method='sample')

    val_data = SequenceDataset(entries=df_val['entry'][:500],
                               sequences=df_val['sequence'][:500], 
                               keywords=df_val[['organism', 'bp', 'cc',
                               'mf', 'insulin']].to_numpy(),
                               kw_method='sample',
                               token_to_idx=train_data.token_to_idx)

    mb_size=64
    
    train_loader = data.DataLoader(train_data, batch_size=mb_size,
                                   shuffle=True, pin_memory=True,
                                   num_workers=4, collate_fn=custom_collate_fn)
    
    val_loader = data.DataLoader(val_data, batch_size=mb_size, shuffle=False,
                                 pin_memory=True, num_workers=4,
                                 collate_fn=custom_collate_fn)
    
    # Choose network model
    n_tokens = len(train_data.token_to_idx)
    embedding_size = 20
    hidden_size = 500
    n_layers = 5

    net = GruNet(n_tokens, embedding_size, hidden_size,
                 n_layers, dropout=0, bidirectional=False)
    
    #net = LstmNet(n_tokens, embedding_size, hidden_size,
    #             n_layers, dropout=0, bidirectional=False)
    
    net = net.to(device=device)

    # Hyper-parameters
    num_epochs = 2

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=train_data.token_to_idx['<PAD>'],
        reduction='mean')
    
    optimizer = torch.optim.Adam(net.parameters())

    # Track loss and perplexity
    train_loss, val_loss = [], []
    train_perplexity, val_perplexity = [], []

    for i in range(num_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_perplexity = 0
        epoch_val_perplexity = 0
        
        val_start = time.time()
        net.eval()

        for inputs, targets, lengths in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs, hidden = net(inputs, lengths)

            loss = criterion(outputs, targets).detach().cpu().numpy()

            # Update loss
            epoch_val_loss += loss
            epoch_val_perplexity += np.exp(loss)
        
        val_end = time.time()
        
        train_start = time.time()
        net.train()

        for inputs, targets, lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs, hidden = net(inputs, lengths)

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

        # Save loss and perplexity
        train_loss.append(epoch_train_loss / np.ceil(len(train_data)/mb_size))
        val_loss.append(epoch_val_loss / np.ceil(len(val_data)/mb_size))

        train_perplexity.append(
            epoch_train_perplexity / np.ceil(len(train_data)/mb_size))
        val_perplexity.append(
            epoch_val_perplexity / np.ceil(len(val_data)/mb_size))

        # Print loss
        if i % 1 == 0:
            print('Epoch {}\nTraining loss: {}, Validation loss: {}'.format(
                i, train_loss[-1], val_loss[-1]))
            print('Training perplexity: {}, Validation perplexity: {}'.format(
                i, train_perplexity[-1], val_perplexity[-1]))
            print('Training time: {}, Validation time: {}\n'.format(
                round(train_end - train_start), round(val_end - val_start)))
    
    torch.save(net, os.path.join(repo_dir, 'models/gru_network.pt'))

    # Convert defaultdicts to regular dicts that can be pickled
    token_to_idx = dict(train_data.token_to_idx)
    idx_to_token = dict(train_data.idx_to_token)
    
    out_file = open(os.path.join(repo_dir, 'models/token_to_idx.pickle'), 'wb')
    pickle.dump(token_to_idx, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    out_file.close()
    
    out_file = open(os.path.join(repo_dir, 'models/idx_to_token.pickle'), 'wb')
    pickle.dump(idx_to_token, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    out_file.close()
