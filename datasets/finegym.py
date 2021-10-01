"""Dataset object for FineGym dataset"""
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd


class FineGym:
    """Dataset object for FineGym dataset"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def _load_annotations(self, filename="gym99_categories.txt", task: str = "action-clf"):
        annot_file = join(self.data_dir, "annotations", task, filename)
        with open(annot_file, "rb") as f:
            lines = f.read()
            lines = lines.decode("utf-8")
            lines = lines.split("\n")

        class_ids = [x.split(";")[0].split(":")[-1].strip() for x in lines[:-1]]
        class_labels = [x.split(";")[-1].strip() for x in lines[:-1]]
        class_label_dict = dict(zip(class_ids, class_labels))

        return class_label_dict
