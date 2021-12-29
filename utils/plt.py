"""Matplotlib helpers."""
import matplotlib.pyplot as plt
import seaborn as sns


def configure_ax(
        ax, H, W,
        titlesize=None,
        legendsize=None,
        xlabelsize=None,
        ylabelsize=None,
        xtickssize=None,
        ytickssize=None,
    ):
    """Configure matplotlib axes."""
    
    if titlesize is None:
        titlesize = W * 2.5
    
    if legendsize is None:
        legendsize = W * 1.9
    
    if xlabelsize is None:
        xlabelsize = W * 1.8
    
    if ylabelsize is None:
        ylabelsize = H * 1.8
    
    if xtickssize is None:
        xtickssize = W * 1.5
    
    if ytickssize is None:
        ytickssize = H * 1.5
    
    ax.set_title(ax.get_title(), fontsize=titlesize)
    ax.legend(loc="best", fontsize=legendsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=xlabelsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=ylabelsize)
    ax.tick_params(axis='both', which='major', labelsize=xtickssize)
    ax.tick_params(axis='both', which='minor', labelsize=ytickssize)
    
    ax.grid()
    
    return ax
    
    
