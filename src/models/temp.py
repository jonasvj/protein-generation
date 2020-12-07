#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def get_repo_dir():
    """Gets root directory of github repository"""
    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()
    
    return repo_dir

repo_dir = get_repo_dir()
model_name = 'tf1'
context_props = np.arange(0.1, 1, 0.1)
gen_results = np.load(os.path.join(repo_dir, 'models/' + model_name + '_gen_results.npy'))
mut_results = np.load(os.path.join(repo_dir, 'models/' + model_name + '_mut_results.npy'))

gen_mean = gen_results.mean(axis=0)
gen_std = gen_results.std(axis=0)
mut_mean = mut_results.mean(axis=0)
mut_std = mut_results.std(axis=0)
fig, ax = plt.subplots(figsize=(8,4))
ax.errorbar(context_props, gen_mean, yerr=gen_std, alpha=.75, fmt='.:',
    capsize=3, capthick=1, c='C0', label='Generated')
ax.errorbar(context_props, mut_mean, yerr=mut_std, alpha=.75, fmt='.:',
    capsize=3, capthick=1, c='C1', label='Mutation baseline')
ax.fill_between(context_props, gen_mean - gen_std, gen_mean + gen_std, 
    alpha=.25, color='C0')
ax.fill_between(context_props, mut_mean - mut_std, mut_mean + mut_std, 
    alpha=.25, color='C1')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc=0)
ax.grid(True)
ax.set_xlabel('Context proportion')
ax.set_ylabel('Similarity')
fig.savefig(
    os.path.join(repo_dir, 'models/' + model_name + '_sim_plot.pdf'))
