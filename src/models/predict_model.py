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
from transformer_network import TransformerModel
from train_model import SequenceDataset, custom_collate_fn

#Initializations=====
model_type = 'wavenetX'

repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()

with open(os.path.join(repo_dir, 'models/' + model_type + '.pickle'), 'rb') as handle:
    stats_dict = pickle.load(handle)

idx_to_token = defaultdict(lambda: '<UNK>')
token_to_idx = defaultdict(lambda: 1)

for key, value in stats_dict['token_to_idx'].items():
    token_to_idx[key] = value

for key, value in stats_dict['idx_to_token'].items():
    idx_to_token[key] = value

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

#Load test set=====
df_test = pd.read_csv(
        os.path.join(repo_dir, 'data/processed/test_data.txt'), sep='\t', dtype = 'str')

test_data = SequenceDataset(df_test['entry'][:1],df_test['sequence'][:1],df_test[['organism', 'bp', 'cc', 'mf', 'insulin']].to_numpy(),kw_method='sample', token_to_idx = token_to_idx)

mb_size=1
test_loader = data.DataLoader(test_data, batch_size=mb_size, shuffle=True,
                                   pin_memory=True, num_workers=4,
                                   collate_fn=custom_collate_fn)

#Load Model=====
net = torch.load(os.path.join(repo_dir, 'models/' + model_type + '.pt'))
net = net.to(device)
net.eval()

softmax = torch.nn.Softmax(dim = 0)
n_globals = 5

#Initialize vectors=====
perplexity = []
bleu = []

#Loop over entries in test set=====
with torch.no_grad():
    for inputs, targets, lengths in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        #=====Perplexity
        if net.model in ['gru', 'lstm']:
            net_inputs = [inputs, [len(inputs[0])]]
        elif net.model in ['transformer', 'wavenet']:
            net_inputs = [inputs]
        elif net.model == 'wavenetX':
            global_inputs = inputs[:,:n_globals]
            inputs = inputs[:,n_globals:]
            targets = targets[:,n_globals:]
            net_inputs = [inputs, global_inputs]

        output = net(*net_inputs)
        outputs = output['output']

        loss = f.cross_entropy(outputs, targets).detach().cpu().numpy()
        perplexity.append(float(np.exp(loss)))

        #=====Predictions
        #Prepare data
        if net.model == 'wavenetX':
            input_ = inputs[:,0:1]
        else:
            num = inputs.cpu().numpy()[0]
            input_ = num[num > 25] #Drop indexes, which do not correspond to amino acids
            input_ = torch.tensor([input_]).to(device)
        prediction = None
        
        #while prediction != torch.tensor(token_to_idx['<EOS>']):
        while len(input_[0]) <100:
            if net.model in ['gru', 'lstm']:
                net_inputs = [input_, [len(input_[0])]]
            elif net.model in ['transformer', 'wavenet']:
                net_inputs = [input_]
            elif net.model == 'wavenetX':
                net_inputs = [input_, global_inputs]

            output = net(*net_inputs) #Forward pass

            prediction = output['output'][0]
            prediction = torch.transpose(prediction, 0, 1)[-1] #Choose the last line of the output, as that corresponds to the next token
            prediction = softmax(prediction) #Softmax
            prediction = torch.argmax(prediction).unsqueeze(0) #Get index of highest value

            input_ = input_.squeeze(0) #Squeeze, so it can be concatenated with prediction

            input_ = torch.cat((input_, prediction), 0).unsqueeze(0)
        
        #=====BLEU Score
        #Translate outputs to amino acids
        output_protein = list(np.vectorize(idx_to_token.get)(input_.cpu().numpy()[0]))
        target_protein = list(np.vectorize(idx_to_token.get)(targets.cpu().numpy()[0]))

        print(output_protein)

        ######Can this get more effective?
        predicted_protein = []
        desired_protein = []
        for i in range(len(output_protein)):
            if len(output_protein[i]) == 1 or output_protein[i] == '<EOS>':
                predicted_protein.append(output_protein[i])
        for i in range(len(target_protein)):
            if len(target_protein[i]) == 1 or target_protein[i] == '<EOS>':
                desired_protein.append(target_protein[i])

        BLEUscore = nltk.translate.bleu_score.sentence_bleu([desired_protein], predicted_protein)
        bleu.append(BLEUscore)

#=====Evaluation of the model
print("\nThe evaluated model was " + model_type)

print("\nAverage perplexity is: ")
print(np.mean(perplexity))

print("\nAverage BLEU score is: ")
print(np.mean(bleu))
print()