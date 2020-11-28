#!/usr/bin/env python3
import torch
import argparse
import subprocess
import numpy as np
from torch.utils import data
from collections import defaultdict

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

def train_model_cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        description='Scrip for training DL models')
    parser.add_argument(
        '--embedding_size',
        help='Size of embedding',
        type=int,
        default=20)
    parser.add_argument(
        '--learning_rate',
        help='learning rate during optimization',
        type=float,
        default=0.001)
    parser.add_argument(
        '--mb_size',
        help='learning rate during optimization',
        type=float,
        default=64)
    parser.add_argument(
        '--epochs',
        help='Number of training epochs',
        type=int,
        default=500
    )
    parser.add_argument(
        '--kw_method',
        help='Method for including protein keywords',
        choices=['merge', 'sample', 'permute'],
        default='sample'
    )
    parser.add_argument(
        'output_file',
        help='Name of output file')
    
    subparsers = parser.add_subparsers(
        help='Type of DL model',
        dest='model')

    # Parser for gru
    parser_gru = subparsers.add_parser(
        'gru',
        help='GRU model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_gru.add_argument(
        '--n_layers',
        help='Number of hidden layers',
        type=int,
        default=2)
    parser_gru.add_argument(
        '--hidden_size',
        help='Number of units in hidden layers',
        type=int,
        default=128)
    parser_gru.add_argument(
        '--dropout',
        help='Dropout rate',
        type=float,
        default=0.1)
    
    # Parser for lstm
    parser_lstm = subparsers.add_parser(
        'lstm',
        help='LSTM model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_lstm.add_argument(
        '--n_layers',
        help='Number of hidden layers',
        type=int,
        default=2)
    parser_lstm.add_argument(
        '--hidden_size',
        help='Number of units in hidden layers',
        type=int,
        default=128)
    parser_lstm.add_argument(
        '--dropout',
        help='Dropout rate',
        type=float,
        default=0.1)
    
    # Parser for transformer
    parser_transformer = subparsers.add_parser(
        'transformer',
        help='Transformer model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_transformer.add_argument(
        '--n_layers',
        help='Number of encoder layers',
        type=int,
        default=2)
    parser_transformer.add_argument(
        '--n_heads',
        help='Number of multi headed attentions per encoder layer',
        type=int,
        default=10)
    parser_transformer.add_argument(
        '--hidden_size',
        help='Number of units in feed-forward layers',
        type=int,
        default=128)
    parser_transformer.add_argument(
        '--dropout',
        help='Dropout rate',
        type=float,
        default=0.1)
    
    # Parser for wavenet
    parser_wavenet = subparsers.add_parser(
        'wavenet',
        help='WaveNet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_wavenet.add_argument(
        '--n_dilations',
        help='Number of dilations (hidden layers)',
        type=int,
        default=2)
    parser_wavenet.add_argument(
        '--kernel_size',
        help='Size of convolution kernel',
        type=int,
        default=2)
    parser_wavenet.add_argument(
        '--stride',
        help='Stride length in convolution',
        type=int,
        default=1)
    parser_wavenet.add_argument(
        '--res_channels',
        help='Number of channels in residual blocks',
        type=int,
        default=16)
    parser_wavenet.add_argument(
        '--f_channels',
        help='Number of channels in final convolution',
        type=int,
        default=16)
    parser_wavenet.add_argument(
        '--X',
        help='Use global inputs',
        action='store_true',
        default=False)

    return vars(parser.parse_args())

def get_repo_dir():
    """Gets root directory of github repository"""
    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()
    
    return repo_dir

if __name__ == '__main__':
    pass