"""Dataset object for NTU dataset"""
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd


class NTU:
    """Dataset object for NTU dataset"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def _load_annotations(self, filename="class_labels.txt", task: str = "action-clf"):
        annot_file = join(self.data_dir, "annotations", task, filename)

        with open(annot_file, "rb") as f:
            lines = f.read()
            lines = lines.decode("utf-8")
            lines = lines.split("\n")
            
        class_label_dict = dict()
        for line in lines:
            if len(line):
                class_id, class_phrase, _ = line.split(".")
                class_label_dict[class_id] = class_phrase.strip()
        
        return class_label_dict
