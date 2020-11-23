#!/usr/bin/env python3

#Imports=====
import subprocess
import pandas as pd
import os
import torch
from gru_batch import GRUNet
from lstm_network import LSTMNet
from transformer_network import TransformerModel

#Load test set=====
repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

df_test = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/test_data.txt'), sep='\t')

#Load Model=====
model = torch.load(os.path.join(repo_dir, 'models/gru_network.pt'))
model.eval()

print(df_test)

with torch.no_grad():
    
#Loop over entries in test set=====

    #Extract keywords from entry

    #Predict sequence based on keywords

    #Evaluate predicted sequence based on different score metrics
        #Perplexity
        #BLEU score
        #Similarity using BLOSUM62

        #Save different scores as a vector