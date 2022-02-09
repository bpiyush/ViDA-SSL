"""Helpers"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


colors_v1 = ["silver", "hotpink", "gold", "yellowgreen", "cornflowerblue", "purple", "blue", "red", "darkkhaki", "limegreen", "black"]
markers_v1 = ["o", "v", "s", "P", "*", "x", "p", "H", "D", "+", "^"]
colors_v2 = ["gray", "gold", "skyblue", "green", "darkorange", "lightseagreen", "blue", "red", "magenta", "black"]
colors_v3 = ["darkgray", "violet", "indigo", "blue", "green", "gold", "orange", "red", "magenta", "black"]
colors_v4 = ["lightgray", "gold", "skyblue", "green", "darkorange", "lightseagreen", "blue", "red", "magenta", "black"]


BASE_URL = 'https://docs.google.com/spreadsheets/d/'
SHEET_ID = "1PU5rWDZYtIdxdoXRh1EmhFtmNOazvWkt3iV30FJLgdA"
GID = {
    "granularity": "733587682",
    "domain_shift_linear": "2014896504",
    "domain_shift_finetune": "890573593",
    "dataset_size": "1134984848",
    "ss-v2-granularity": "1950972304",
}


def read_spreadsheet(sheet_id=SHEET_ID, gid=None, gid_key="granularity", **kwargs):
    if gid is None:
        gid = GID[gid_key]

    df = df = pd.read_csv(BASE_URL + SHEET_ID + f'/export?gid={gid}&format=csv', **kwargs)
    return df


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


def find_sub_element_in_list(x, L):
    """Finds an element in list L of which x is a substring"""
    for y in L:
        if x in y:
            return y


def load_domain_shift_results(remove_K400=True):
    df_linear = read_spreadsheet(gid_key="domain_shift_linear", index_col=0)
    df_finetune = read_spreadsheet(gid_key="domain_shift_finetune", index_col=0)

    # re-order VSSL methods based on UCF performance
    ref_dataset = find_sub_element_in_list("UCF", df_finetune.columns)
    df_finetune.sort_values(ref_dataset, inplace=True)
    df_linear = df_linear.loc[list(df_finetune.index)]
    
    # remove column for `K400`
    k400_values = dict()
    k400_values["Method"] = list(df_finetune.index)
    if remove_K400:
        if "K400" in df_linear.columns:
            k400_linear = df_linear["K400"].values
            k400_values["linear"] = k400_linear
            df_linear.drop(columns=["K400"], inplace=True)
        if "K400" in df_finetune.columns:
            k400_finetune = df_finetune["K400"].values
            k400_values["finetune"] = k400_finetune
            df_finetune.drop(columns=["K400"], inplace=True)
    
    # re-order datasets
    if remove_K400:
        correct_order = ["UCF", "NTU", "Gym", "SS", "EPIC"]
    else:
        correct_order = ["K400", "UCF", "NTU", "Gym", "SS", "EPIC"]
    reorder_colums = [find_sub_element_in_list(x, df_linear.columns) for x in correct_order]
    df_linear = df_linear[reorder_colums]
    df_finetune = df_finetune[reorder_colums]

    return k400_values, df_linear, df_finetune



def scatter_with_correlation(
        xvalues, yvalues, labels, colors=colors_v1,
        title="", xlabel="X", ylabel="Y", ax=None, legend=False, show=False, 
        titlesize=25, xlabelsize=20, ylabelsize=20, legendsize=20, markersize=80, size_alpha=1.0,
        legend_kwargs=dict(loc='upper center', bbox_to_anchor=(1.3, 0.9))
    ):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.grid()

    # xvalues = dfs["Full"]["Accuracy"].values
    # yvalues = dfs["Coarse"]["Accuracy"].values
    # labels = dfs["Coarse"]["Method"].values
    # colors = colors_v1

    for (x, y, l, c) in zip(xvalues, yvalues, labels, colors):
        ax.scatter(x, y, label=l, color=c, s=markersize * size_alpha)

    sns.regplot(x=xvalues, y=yvalues, ax=ax, scatter=False)

    ax.set_xlabel(xlabel, fontsize=xlabelsize * size_alpha)
    ax.set_ylabel(ylabel, fontsize=ylabelsize * size_alpha)

    corr = np.round(np.corrcoef(xvalues, yvalues)[0, 1], decimals=3)
    corr = f"$\\rho = {corr}$"
    ax.set_title("{} ({})".format(title, corr), fontsize=titlesize * size_alpha)
    
    if legend:
        if "fontsize" not in legend_kwargs:
            legend_kwargs.update(dict(fontsize=legendsize * size_alpha))
        ax.legend(**legend_kwargs)
    
    if show:
        plt.show()