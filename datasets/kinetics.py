"""Dataset object for Kinetics dataset"""
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd


class Kinetics:
    """Dataset object for Kinetics dataset"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def _load_annotations(self, filename="kinetics_400_labels.csv", task: str = "action-clf"):
        annot_file = join(self.data_dir, "annotations", task, filename)
        df = pd.read_csv(annot_file)

        class_ids = df["id"].values
        class_labels = df["name"].values
        class_label_dict = dict(zip(class_ids, class_labels))

        return class_label_dict
