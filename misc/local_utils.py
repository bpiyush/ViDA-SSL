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


def get_sentence_embedding(model, tokenizer, sentence):
    encoded = tokenizer.encode_plus(sentence, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded)
    
    last_hidden_state = output.last_hidden_state
    assert last_hidden_state.shape[0] == 1
    assert last_hidden_state.shape[-1] == 768
    
    # only pick the [CLS] token embedding (sentence embedding)
    sentence_embedding = last_hidden_state[0, 0]
    
    return sentence_embedding


def get_sent_emb_from_word_emb(model, tokenizer, sentence):
    encoded = tokenizer.encode_plus(sentence, return_tensors="pt")

    with torch.no_grad():
        sent_emb = model.embeddings.word_embeddings(encoded["input_ids"]).mean(1)[0]
    
    return sent_emb
