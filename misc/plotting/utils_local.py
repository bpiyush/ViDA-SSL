"""Helpers"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


colors_v1 = ["silver", "hotpink", "gold", "yellowgreen", "cornflowerblue", "purple", "blue", "red", "darkkhaki", "limegreen", "black"]
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