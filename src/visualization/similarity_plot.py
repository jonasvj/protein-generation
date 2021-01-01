#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_repo_dir, set_font_sizes

def similarity_plot(context_props, mutation_rate, model_name,
    gen_results, mut_results):
    """Plots sequence similarity versus context proportion"""
    mutation_label = "{}% mutation baseline".format(int(mutation_rate*100))

    # Mean and standard deviation of results
    gen_mean = np.nanmean(gen_results, axis=0)
    gen_std = np.nanstd(gen_results, axis=0)
    mut_mean = np.nanmean(mut_results, axis=0)
    mut_std = np.nanstd(mut_results, axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(context_props, gen_mean, yerr=gen_std, alpha=.75, fmt='.:',
        capsize=3, capthick=1, c='C0', label='Generated')
    ax.errorbar(context_props, mut_mean, yerr=mut_std, alpha=.75, fmt='.:',
        capsize=3, capthick=1, c='C1', label=mutation_label)
    ax.fill_between(context_props, gen_mean - gen_std, gen_mean + gen_std, 
        alpha=.25, color='C0')
    ax.fill_between(context_props, mut_mean - mut_std, mut_mean + mut_std, 
        alpha=.25, color='C1')
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=0)
    ax.grid(True)
    ax.set_xlabel('Proportion of sequence as context')
    ax.set_ylabel('Similarity')
    ax.set_title(model_name)

    return fig


if __name__ == '__main__':
    repo_dir = get_repo_dir()
    set_font_sizes(small=18, medium=18, large=18)
    model_name = sys.argv[1]

    gen_results = np.load(os.path.join(
        repo_dir, 'models/' + model_name + '_similarity_scores_generation.npy'))
    mut_results = np.load(os.path.join(
        repo_dir, 'models/' + model_name + '_similarity_scores_mutation.npy'))

    context_props = np.arange(0.1, 1, 0.1)
    mutation_rate = 0.5

    similarity_plot = similarity_plot(context_props, mutation_rate, model_name,
        gen_results, mut_results)

    similarity_plot.savefig(os.path.join(
        repo_dir, 'reports/figures/' + model_name + '_similarity_plot.pdf'))