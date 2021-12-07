"""Helpers"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap(df, titlesize=20, labelsize=15, tickssize=13):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    sns.heatmap(df, cmap="YlGnBu", ax=ax)

    ax.set_title("Action classification", fontsize=titlesize)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Method")
    
    ax.yaxis.label.set_size(labelsize)
    ax.xaxis.label.set_size(labelsize)

    ax.tick_params(axis='x', labelsize=tickssize)
    ax.tick_params(axis='y', labelsize=tickssize)

    plt.show()