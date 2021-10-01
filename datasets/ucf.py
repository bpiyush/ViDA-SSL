"""Dataset object for UCF dataset"""
from os.path import join
from tqdm import tqdm
import re
import numpy as np
import pandas as pd


class UCF:
    """Dataset object for UCF dataset"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def camel_to_snake(self, string):
        """
        Thanks: shorturl.at/gyJQX
        """
        return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()
    
    def _load_annotations(self, filename="Class Index.txt", task: str = "action-clf"):
        annot_file = join(self.data_dir, "annotations", task, filename)
        with open(annot_file, "rb") as f:
            lines = f.read()
            lines = lines.decode("utf-8")
            lines = lines.split("\r\n")

        class_label_dict = {
            x.split(" ")[0]: self.camel_to_snake(x.split(" ")[-1]).replace("_", " ") for x in lines[:-1]
        }
        return class_label_dict
