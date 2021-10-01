"""Util functions for misc/"""
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from fast_pytorch_kmeans import KMeans
from transformers import AutoTokenizer, AutoModel


def get_phrase_embedding(model, tokenizer, phrase, layers=[-4, -3, -2, -1], agg_method="mean"):

    encoded = tokenizer.encode_plus(phrase, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states

    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    phrase_word_embeddings = output[1:-1]
    phrase_embedding = getattr(torch, agg_method)(phrase_word_embeddings, dim=0)
    
    return phrase_embedding