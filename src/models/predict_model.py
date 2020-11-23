#!/usr/bin/env python3

#Imports=====
import subprocess
import pandas as pd
import os
import torch
import pickle
import sys
import numpy as np
import torch.nn.functional as f
import nltk
from collections import defaultdict
from torch.utils import data
from gru_batch import GRUNet
from lstm_network import LSTMNet
from transformer_network import TransformerModel
from train_model_batch import SequenceDataset, custom_collate_fn

#Initializations=====
repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

with open(os.path.join(repo_dir, 'models/idx_to_token.pickle'), 'rb') as handle:
    idx_to_token = pickle.load(handle)

with open(os.path.join(repo_dir, 'models/token_to_idx.pickle'), 'rb') as handle:
    token_to_idx = pickle.load(handle)

idx_to_token = defaultdict(lambda: '<UNK>').update(idx_to_token)
token_to_idx = defaultdict(lambda: 1).update(token_to_idx)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

#Load test set=====
df_test = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/test_data.txt'), sep='\t', dtype = 'str')

test_data = SequenceDataset(df_test['entry'], df_test['sequence'], df_test[['organism', 'bp', 'cc', 'mf', 'insulin']].to_numpy(), kw_method='merge', token_to_idx = token_to_idx)

mb_size=1
test_loader = data.DataLoader(test_data, batch_size=mb_size, shuffle=True,
                                   pin_memory=True, num_workers=4,
                                   collate_fn=custom_collate_fn)

#Load Model=====
net = torch.load(os.path.join(repo_dir, 'models/gru_network.pt')) #GruNet
#net = torch.load(os.path.join(repo_dir, 'models/lstm_network.pt')) #LstmNet
#net = torch.load(os.path.join(repo_dir, 'models/transformer_network.pt')) #TransformerNet
#net = torch.load(os.path.join(repo_dir, 'models/wave_network.pt')) #WaveNet

net = net.to(device)

net.eval()
softmax = torch.nn.Softmax(dim = 2)

#Initialize vectors=====
perplexity = []
bleu = []

#Loop over entries in test set=====
with torch.no_grad():
    for inputs, input_lengths, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)

        #Perplexity
        loss = f.cross_entropy(outputs.permute(1,2,0), targets.permute(1,0))
        perplexity.append(float(torch.exp(loss)))

        #Predict sequence
        """
        While pred != <EOS>:
            outputs, new hidden state = net(input, hidden state) #First input of the test sequence, outputs a list to sample from
            prediction = max(output[-1]) #Sample the largest value, that is the prediction
            new inputs = input + prediction #Update inputs for next run of the network
        """
        
        #Sample the predicted protein
        predicted = softmax(predicted).squeeze()

        
        #BLEU Score
        #BLEUscore = nltk.translate.bleu_score.sentence_bleu([targets], outputs)
        
        print(outputs.shape)
        sys.exit(1)

        #Evaluate predicted sequence based on different score metrics
        #BLEU score
        #Similarity using BLOSUM62


print("Average perplexity is: ")
print(np.mean(perplexity))

print("Average BLEU score is: ")
print(np.mean(bleu))