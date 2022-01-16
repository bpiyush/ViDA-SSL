"""Helpers"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


colors_v1 = ["silver", "hotpink", "gold", "yellowgreen", "cornflowerblue", "purple", "blue", "red", "darkkhaki", "black"]
colors_v2 = ["gray", "gold", "skyblue", "green", "darkorange", "lightseagreen", "blue", "red", "magenta", "black"]
colors_v3 = ["darkgray", "violet", "indigo", "blue", "green", "gold", "orange", "red", "magenta", "black"]
colors_v4 = ["lightgray", "gold", "skyblue", "green", "darkorange", "lightseagreen", "blue", "red", "magenta", "black"]


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