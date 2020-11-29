#!/usr/bin/env python3
import os
import sys
import pickle
from utils import get_repo_dir
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_name = sys.argv[1]
    repo_dir = get_repo_dir()

    model_dict_file = open(
        os.path.join(repo_dir, 'models/' + model_name + '.pickle'), 'rb')
    model_dict = pickle.load(model_dict_file)
    model_dict_file.close()

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    # x-axis
    ax1.set_xlabel('Epoch')
    ax1.tick_params(axis='x')
    # y-axis for loss
    ax1.set_ylabel('Loss', color='C1')
    ax1.tick_params(axis='y', labelcolor='C1')
    ax1.plot(model_dict['train_loss'][1:], c='C1', label='Training loss')
    ax1.plot(model_dict['val_loss'][1:], c='C1', linestyle='--',
        label='Validation loss')
    # Second y-axis for perplexity
    ax2.set_ylabel('Perplexity', c='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    ax2.plot(model_dict['train_perplexity'][1:], c='C0',
        label='Training perplexity')
    ax2.plot(model_dict['val_perplexity'][1:], c='C0', linestyle='--',
        label='Validation perplexity')
    # Title, legend and grid
    ax1.set_title("Training metrics")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=0)
    ax1.grid(True)
    plt.savefig(os.path.join(repo_dir, 'models/' + model_name + '_plots.pdf'))




