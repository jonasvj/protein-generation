#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from src.utils import get_repo_dir, set_seeds

def embedding_plot(emb, test_results, model_name):
    """Plot sequence embeddings"""
    fig, ax = plt.subplots(figsize=(8,6))

    # Color by molecular function
    colors = test_results.mf.factorize()[0]
    labels = test_results.mf.factorize()[1].to_numpy()
    labels[labels == "Developmental protein"] = "Developmental\nprotein"

    mitogen_missing = False
    if 'Mitogen' not in labels:
        mitogen_missing = True
        labels = np.insert(labels, 2, 'Mitogen')
        colors[colors >= 2] += 1

    # Scatter plot
    scatter = ax.scatter(emb[:, 0], emb[:, 1],
                        c=colors,
                        cmap=plt.get_cmap("tab10"),
                        s=2,
                        alpha=0.8)
    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')

    # Markers for legend
    markers = scatter.legend_elements()[0]
    if mitogen_missing is True:
        markers = np.insert(markers, 2, mlines.Line2D([], [], marker='o',
            linestyle='', alpha=0.8, markeredgecolor='C2', color='C2'))
    
    plt.legend(markers, labels, loc=0,
        borderaxespad=0.1, title='Molecular function', framealpha=0.6)

    ax.set_title(model_name)
    plt.tight_layout()

    return fig

if __name__ == '__main__':
    seed = 42
    set_seeds(seed)
    repo_dir = get_repo_dir()
    model_name = sys.argv[1]

    # Load data
    test_results = pd.read_csv(os.path.join(
            repo_dir, 'models/' + model_name + '_test_results.csv'))
    
    emb_1 = test_results.loc[:, test_results.columns.str.startswith('emb_1_')]
    emb_2 = test_results.loc[:, test_results.columns.str.startswith('emb_2_')]
    emb_1 = emb_1.to_numpy()
    emb_2 = emb_2.to_numpy()

    # Project embeddings to 2-dimensional space
    X_emb_1 = TSNE(n_components=2, random_state=seed).fit_transform(emb_1)
    X_emb_2 = TSNE(n_components=2, random_state=seed).fit_transform(emb_2)

    # Plot sequence embeddings
    emb_plot_1 = embedding_plot(X_emb_1, test_results, model_name)
    emb_plot_2 = embedding_plot(X_emb_2, test_results, model_name)

    # Save figuress
    emb_plot_1.savefig(os.path.join(
        repo_dir, 'reports/figures/' + model_name + '_emb_plot_1.pdf'))
    emb_plot_2.savefig(os.path.join(
        repo_dir, 'reports/figures/' + model_name + '_emb_plot_2.pdf'))
