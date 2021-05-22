"""
"""
# Standard Library imports:
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from shutil import copy2 as copy
from copy import deepcopy
import csv

# 3rd Party imports:
import numpy as np
from pycocotools.coco import COCO
import pandas


def xview_public_split(
    train_txt, val_txt
) -> Tuple[List[Any], List[Any]]:
    """
    """
    train_ids = [line.split(".")[0] for line in open(train_txt)]
    val_ids = [line.split(".")[0] for line in open(val_txt)]

    return train_ids, val_ids

def get_split_dict_from_csv(
    csv_path
) -> Dict:
    """
    """
    split_dict = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            split_dict[row['\ufeffimage_id']] = row['split'] #TODO address the '\ufeff' problem directly
    return split_dict

def get_img_index_from_csv(
    csv_path
) -> Dict:
    """
    Returns a dictionary that maps from and image id to a dict of image metadata e.g. split, width, height, etc.
    """
    img_index = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_index[row['\ufeffimage_id']] = { #TODO address the '\ufeff' problem directly
                'split': row['split'],
                'width': row['image_width'],
                'height': row['image_height']
            }
    return img_index