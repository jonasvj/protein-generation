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
#from gru import GruNet
#from lstm import LstmNet
from transformer_network import TransformerModel
from train_model import SequenceDataset, custom_collate_fn

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

test_data = SequenceDataset(df_test['entry'][:50],df_test['sequence'][:50],df_test[['organism', 'bp', 'cc', 'mf', 'insulin']].to_numpy(),kw_method='merge', token_to_idx = token_to_idx)

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
    for inputs, targets, lengths in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if net.model == "transformer":
            net_inputs = [inputs]
        elif net.model in ['gru', 'lstm']:
            net_inputs = [inputs, lengths]

        output = net(*net_inputs)
        outputs = output['output']

        #Perplexity
        loss = f.cross_entropy(outputs, targets).detach().cpu().numpy()
        perplexity.append(float(np.exp(loss)))

        #Prepare data for the predictions
        num = inputs.cpu().numpy()[0]
        input_ = num[num > 25]
        input_ = torch.tensor([input_]).to(device)
        softmax = torch.nn.Softmax(dim = 0)
        hidden_state = [len(input_[0])]
        
        while len(input_[0]) < 100:
            output = net(input_, [len(input_[0])]) #Forward pass
            hidden_state = output['hidden'] #Save the hidden state

            prediction = output['output'][0]
            prediction = torch.transpose(prediction, 0, 1)[-1] #Choose the last line of the output, as that corresponds to the next token
            prediction = softmax(prediction) #Softmax
            prediction = torch.argmax(prediction).unsqueeze(0) #Get index of highest value
            input_ = input_.squeeze()

            input_ = torch.cat((input_, prediction), 0).unsqueeze(0)
        
        #BLEU Score
        outputs = input_
        #BLEUscore = nltk.translate.bleu_score.sentence_bleu([targets], outputs)

print("Average perplexity is: ")
print(np.mean(perplexity))

#print("Average BLEU score is: ")
#print(np.mean(bleu))