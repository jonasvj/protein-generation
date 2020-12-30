#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import get_repo_dir

def perplexity_box_plot(perplexities, model_names):
    """Box plot of perplexities grouped by model"""
    data = list()

    # Create column of perplexities and model name (repeated) for each model
    for i in range(len(perplexities)):
        data.append(np.append(
            perplexities[i],
            np.full(perplexities[i].shape, model_names[i], dtype='object'),
            axis=1))
    
    # Merge data
    data = np.concatenate(data, axis=0)
    data = pd.DataFrame(data, columns=['Perplexity', 'Model'])

    # Create figure
    fig, ax = plt.subplots(figsize=(8,4))
    data.boxplot(column=['Perplexity'], by="Model", ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel('Test perplexity')
    ax.set_title(None)
    fig.suptitle('Test perplexity grouped by model')

    return fig

if __name__ == '__main__':
    model_names = sys.argv[1:]
    repo_dir = get_repo_dir()
    perplexities = list()

    # Load perplexities
    for model_name in model_names:
        test_results = pd.read_csv(os.path.join(
            repo_dir, 'models/' + model_name + '_test_results.csv'))
        perplexities.append(
            test_results['perplexity'].to_numpy()[:, np.newaxis])
    
    box_plot = perplexity_box_plot(perplexities, model_names)
    box_plot.savefig(os.path.join(
        repo_dir, 'reports/figures/test_perplexity_box_plot.pdf'))