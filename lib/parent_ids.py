"""
cocohelpers is a module with helper classes and functions related to the MS
COCO API. Includes helpers for building COCO formatted json, inspecting class
distribution, and generating a train/val split.
"""
# Standard Library imports:
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
import copy

# 3rd Party imports:
import numpy as np
from pycocotools.coco import COCO


def get_pids(coco):
    # Get ids of parent images
    imgs = coco.dataset["images"]

    parent_ids = set()
    for img in imgs:
        parent_ids.add(img["parent_id"])

    return parent_ids

def get_pidToImgIds(coco):
    imgs = coco.dataset["images"]

    parent_ids = get_pids(coco)

    pidToImgIds={}
    for pid in parent_ids:
        pidToImgIds[pid] = []

    for img in imgs:
        pidToImgIds[img["parent_id"]].append(img["id"])

    return pidToImgIds

def get_cat_names(coco):
    cats = coco.loadCats(coco.getCatIds())
    return [cat['name'] for cat in cats]


def get_pids_w_query(query_class, coco):
    
    # Ensure query class is valid
    cat_names = get_cat_names(coco)
    assert query_class in cat_names, f"Query class {query_class} doesn't match \
                                       any of the following coco categories: {cat_names}"

    # Get image ids that contain the query class
    query_cat_ids = coco.getCatIds(catNms=[query_class])
    imids_w_query = coco.getImgIds(catIds=query_cat_ids)

    pids = get_pids(coco)

    pid_to_imids = get_pidToImgIds(coco)

    imid_to_anns={}
    for imid in imids_w_query:
        imid_to_anns[imid] = []
        annids_w_query = coco.getAnnIds(imgIds=imid, catIds=query_cat_ids, iscrowd=None)
        for annid in annids_w_query:
            imid_to_anns[imid].append(annid)

    # Create parentIdToAnnCount
    pid_to_ann_count={}
    for pid in pids:
        count=0
        imids = pid_to_imids[pid]
        for imid in imids:
            if imid in imid_to_anns.keys():
                count += len(imid_to_anns[imid])
        pid_to_ann_count[pid] = count

    pid_counts = pid_to_ann_count.values()
    #pid_nonzero_indexes = [i for i, e in enumerate(pid_counts) if e != 0]
    pid_nonzero_indexes = [k for k, v in pid_to_ann_count.items() if v != 0]

    return set(pid_nonzero_indexes)