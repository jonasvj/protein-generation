#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import get_repo_dir

def plot_learning_curves(stats_dict, model_name):
    """Plot learning curves"""
    fig, ax1 = plt.subplots(figsize=(8,4))
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
    ax1.set_title(model_name)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=0)
    ax1.grid(True)

    return fig

if __name__ == '__main__':
    model_name = sys.argv[1]
    repo_dir = get_repo_dir()

    # Get dictionary with model stats, arguments etc.
    stats_dict_file = open(os.path.join(
        repo_dir, 'models/' + model_name + '_stats_dict.pickle'), 'rb')
    stats_dict = pickle.load(stats_dict_file)
    stats_dict_file.close()

    # Create and save learning curves
    learning_curves = plot_learning_curves(stats_dict, model_name)
    learning_curves.savefig(os.path.join(
        repo_dir, 'reports/figures/' + model_name + '_learning_curves.pdf'))
    