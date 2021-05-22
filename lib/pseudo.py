"""
coco_pseudo merges coco instances with pseudo labels.  It provides options
in terms of how to select pseudo labels based on prediction confidence and
comparisons with original labels.
"""
from pycocotools.remap import update_mapping
import copy
import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import copy
import itertools
from pycocotools import mask as maskUtils
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

# Standard Library imports:
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from shutil import copy2 as copy
from copy import deepcopy
import csv
import yaml

# 3rd Party imports:
import numpy as np
from pycocotools.coco import COCO
import pandas

# utils
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def load_json(json_path: Path):

    tic = time.time()
    with open(json_path) as f:
        loaded_data = json.load(f)
        print(
            f"LOADING {len(loaded_data)} predictions from: {json_path}."
        )
    print('Done (t={:0.2f}s)'.format(time.time()- tic))
    return loaded_data


def load_config(self):
    with open(self.config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    self.config = config


def box_to_poly(box):
    '''
    Create a sequence of segmentation points
    (Len has to be even and >= 6 e.g. a triangle)
    '''
    topleft = (box[0], box[1])
    topright = (box[0]+box[2], box[1])
    bottomright = (box[0]+box[2], box[1]+box[3])
    bottomleft = (box[0], box[1]+box[3])

    return [[
        topleft[0], topleft[1],
        topright[0], topright[1],
        bottomright[0], bottomright[1],
        bottomleft[0], bottomleft[1]
    ]]


class COCO_PSEUDO(COCO):
    '''
    coco_pseudo merges coco instances with pseudo labels.  It provides options
    in terms of how to select pseudo labels based on prediction confidence and
    comparisons with original labels.
    '''
    def __init__(self, annotation_file=None, preds_file=None, conf_thresh=0.0, join_type='concat', create_mapping=False, mapping_csv=None, write_to_JSON=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :param create_mapping (boolean): if True, remaps labels/classes according to remap.py.
        If a dataset has not yet been remapped, use the dataset's annotation_file and set create_mapping to True.
        Remapping generates a JSON file containing the remapped dataset.
        This JSON file should be used as the annotation_file for each subsequent creation of COCO instances if mapping already complete.
        When using the remapped JSON file, create_mapping should be set to False.
        :param mapping_csv (str): location of csv file containing the remapping scheme.
        :param write_to_JSON (str): name of desired JSON file to be written containing new dataset.
        :return:
        """
        # load dataset
        if create_mapping:
            assert not mapping_csv == None, 'mapping_csv required'
            assert not write_to_JSON == None, 'write_to_JSON path required'
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.preds_file = preds_file
        self.conf_thresh = conf_thresh
        coco = COCO(annotation_file)
        # Add the 'ispseudo' key to each annotation
        for ann in coco.dataset['annotations']:
            ann['ispseudo'] = 0
        self.max_ann_id = int(list(coco.anns.keys())[-1])
        if join_type == 'swap':
            self.dataset = self.get_ss_dataset(coco)
        else: #concat
            self.dataset = self.get_joined_dataset(coco)
        if create_mapping:
            update_mapping(self.dataset, mapping_csv, write_to_JSON)
        self.createIndex()


    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], ispseudo=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
            catIds  (int array)     : get anns for given cats
            areaRng (float array)   : get anns for given area range (e.g. [0 inf])
            ispseudo (boolean)       : get anns for pseudo labels (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not ispseudo == None:
            ids = [ann['id'] for ann in anns if ann['ispseudo'] == ispseudo]
        else:
            ids = [ann['id'] for ann in anns]
        return ids


    def get_pseudo_anns(self):
        preds = load_json(self.preds_file)
        pseudo_anns = []
        # start ann counter at the last ann id from
        # base dataset
        ann_counter = self.max_ann_id
        for pred in preds:
            if pred['score'] > self.conf_thresh:
                ann_counter += 1
                pseudo_anns.append(
                    {
                        "segmentation": box_to_poly(pred['bbox']),
                        "bbox": pred['bbox'],
                        "area": pred['bbox'][2]*pred['bbox'][3],
                        "id": ann_counter,
                        "image_id": pred['image_id'], # TODO: fix xview construction so image id's are zero idx
                        "category_id": pred['category_id'],
                        "iscrowd": 0, # matters for coco_eval
                        "ispseudo": 1
                    }
                )
        return pseudo_anns

    def get_pseudo_img_ids(self, pseudo_anns):
        '''
        Get a list of the image ids that have pseudo annotations
        '''
        img_ids = set()
        for ann in pseudo_anns:
            img_ids.add(ann['image_id'])
        return list(img_ids)


    def concat_pseudo_anns(self, coco):
        '''
        Return a new coco object with the pseudo labels
        concatenated to list of annotations
        '''
        return coco.dataset['annotations'] + self.get_pseudo_anns()


    def swap_pseudo_anns(self, coco):
        '''
        Return a new coco object with a portion of the
        original labels, swapped out for pseudo labels
        '''
        pseudo_anns = self.get_pseudo_anns()
        pseudo_img_ids = self.get_pseudo_img_ids(pseudo_anns)
        base_anns = []
        # reconstruct non-pseudo annotations
        for ann in coco.dataset['annotations']:
            if ann['image_id'] not in pseudo_img_ids:
                base_anns.append(
                    {
                        "segmentation": ann['segmentation'],
                        "bbox": ann['bbox'],
                        "area": ann['area'],
                        "id": ann['id'],
                        "image_id": ann['image_id'],
                        "category_id": ann['category_id'],
                        "iscrowd": 0, # matters for coco_eval
                        "ispseudo": 0
                    }
                )
        return base_anns + pseudo_anns


    def get_joined_dataset(self, coco):
        return {
            "categories": coco.dataset['categories'],
            "images": coco.dataset['images'],
            "annotations": self.concat_pseudo_anns(coco),
        }


    def get_ss_dataset(self, coco):
        return {
            "categories": coco.dataset['categories'],
            "images": coco.dataset['images'],
            "annotations": self.swap_pseudo_anns(coco),
        }

    # def get_joined_coco(self):
    #     '''
    #     '''
    #     coco = deepcopy(self.coco)
    #     joined_dataset = self.get_joined_dataset()
    #     anns, cats, imgs = {}, {}, {}
    #     imgToAnns, catToImgs = defaultdict(list),defaultdict(list)
    #     for ann in joined_dataset['annotations']:
    #         imgToAnns[ann['image_id']].append(ann)
    #         anns[ann['id']] = ann

    #     for img in joined_dataset['images']:
    #         imgs[img['id']] = img

    #     for cat in joined_dataset['categories']:
    #         cats[cat['id']] = cat

    #     for ann in joined_dataset['annotations']:
    #         catToImgs[ann['category_id']].append(ann['image_id'])

    #     coco.anns = anns
    #     coco.imgToAnns = imgToAnns
    #     coco.catToImgs = catToImgs
    #     coco.imgs = imgs
    #     coco.cats = cats
    #     coco.dataset = joined_dataset
    #     return coco


    def save_dataset_to_json(self, save_path):
        print(f"Writing output to: '{save_path}'")
        with open(save_path, "w") as coco_file:
            coco_file.write(json.dumps(self.dataset))


    # def save_joined_dataset(self, save_path) -> None:
    #     '''
    #     Saves the joined dataset w/ pseudo labels
    #     to a json file at <save_path>.
    #     '''
    #     self.save_dataset_to_json(
    #         save_path,
    #         self.get_joined_dataset()
    #     )

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['ispseudo'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['ispseudo'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])