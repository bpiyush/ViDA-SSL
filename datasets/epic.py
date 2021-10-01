"""Dataset object for EPIC-KITCHENS dataset"""
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd


class EPIC:
    """Dataset object for EPIC-KITCHENS dataset"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def _load_annotations(self, filename="EPIC_100_verb_classes.csv", task: str = "action-clf"):
        annot_file = join(self.data_dir, "annotations", task, filename)
        df = pd.read_csv(annot_file)
        class_label_dict = {k: df['key'][k] for k in df['id'].values}
        
        return class_label_dict
