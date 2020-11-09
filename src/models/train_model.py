#!/usr/bin/env python3
#==Initializtion=================================
import torch
import numpy as np
import pandas as pd
from torch.utils import data
import sys

sys.path.append('../data')

from create_datasets import SequenceDataset

#==Load data=====================================
train_data = torch.load('../../data/processed/train_data.pt')
val_data = torch.load('../../data/processed/val_data.pt')
test_data = torch.load('../../data/processed/test_data.pt')

#==Import network models=========================
#Gru network
#LSTM network
#Transformer network

#Choose network model
#net = 

#==Training loop=================================
#Hyper-parameters
num_epochs = 5

#Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

#Track loss
training_loss, validation_loss = [], []

#For each epoch
for i in range(num_epochs):

    #Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0

    net.eval()

    #For each protein in the validation set
    for inputs, targets in val_data:

        #Forward pass
        outputs = net(inputs)

        #Compute loss
        loss = criterion(outputs, targets)

        #Update loss
        epoch_validation_loss += loss.detach().numpy()

    net.train()

    for inputs, targets in train_data:

        #Forward pass
        outputs = net.forward(inputs)

        #Compute loss
        loss = criterion(outputs, targets)

        #Backward pass
        optimizer.zero_grad()
        loss.backward ()
        optimizer.step()

        #Update loss
        epoch_training_loss += loss.detach().numpy()

    #Save loss
    training_loss.append(epoch_training_loss / len(train_data))
    validation_loss.append(epoch_validation_loss / len(val_data))

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')