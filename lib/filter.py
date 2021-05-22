"""
coco_filter is a class that filters the annotations, images, and/or categories of a COCO instance.
"""
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

class COCO_FILTER:

    def __init__(self, coco_path, split, config_path=None):
        self.config = None
        self.config_path = config_path
        self.coco = COCO(coco_path)
        self.split = split
        self.dataset = self.coco.dataset
        self.cat_ids = set(self.coco.cats.keys())
        self.img_ids = set(self.coco.imgs.keys())
        self.ann_ids = set(self.coco.anns.keys())

        print(f"Starting with {len(self.img_ids)} image ids")

        # these "filters" are sets of ids (img_ids, cat_ids, and ann_ids) that are to be filtered out i.e. thrown away
        self.cat_filter = set()
        self.img_filter = set()
        self.ann_filter = set()
        # self.area_range = []

        self.load_config()

        self.remap_dataset()

        for x in ['categories', 'images', 'annotations']:
            print(f"{len(self.dataset[x])} {x} after remap")

        self.reindex_dataset()

        self.reset_coco()

        for x in ['categories', 'images', 'annotations']:
            print(f"{len(self.coco.dataset[x])} coco {x} after coco reset")

        self.build_indexes()

        self.build_filters()

        self.filter_dataset()

        for x in ['categories', 'images', 'annotations']:
            print(f"{len(self.dataset[x])} {x} after filter")

        self.reindex_dataset()

        self.reset_coco()

        print("Coco build complete.")
        print("The new coco dataset has...")
        print(f"{len(self.coco.cats)} categories")
        print(f"{len(self.coco.imgs)} images")
        print(f"{len(self.coco.anns)} annotations")

    def load_config(self):
        with open(self.config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.config = config

    def build_indexes(self):
        # print(self.config)
        if self.config['FILTER_BY']['CAT']['INSTANCES']['ENABLED']:
            # indexes for categorical filtering
            self.cat_name_to_id = {cat_dict['name']: cat_dict['id'] for cat_dict in self.coco.dataset['categories']}
            self.cat_id_to_instance_count = {cat_id: len(self.coco.getAnnIds(catIds=[cat_id])) for cat_id in self.coco.getCatIds()}

        if self.config['FILTER_BY']['IMG']['ENABLED']:
            # indexes for image-based filtering
            self.img_id_to_width = {img_dict['id']: img_dict['width'] for img_dict in self.coco.dataset['images']}
            self.img_id_to_height = {img_dict['id']: img_dict['height'] for img_dict in self.coco.dataset['images']}

        if self.config['FILTER_BY']['ANN']['ENABLED']:
            pass

    def build_filters(self):
        if self.config['FILTER_BY']['CAT']['ENABLED']:
            self.build_cat_filter()

        # print(f'len image filter before creating filter: {len(self.img_filter)}')
        if self.config['FILTER_BY']['IMG']['ENABLED']:
            self.build_img_filter()

        if self.config['FILTER_BY']['ANN']['ENABLED']:
            self.build_ann_filter()

        print(f'len image filter before add empty img ids to filter: {len(self.img_filter)}')

        # filter empty images depending on previous filters
        keep_empty_images = self.config['FILTER_BY']['IMG']['KEEP_EMPTIES'][self.split.upper()]
        if not keep_empty_images:
            self.add_empty_imgs_to_filter()

        # print(f'len image filter after add empty img ids to filter: {len(self.img_filter)}')

    def build_cat_filter(self):
        keep_cat_names = self.config['FILTER_BY']['CAT']['NAME']['KEEP']
        keep_cat_ids = self.config['FILTER_BY']['CAT']['ID']['KEEP']
        keep_cat_count_greater_than = self.config['FILTER_BY']['CAT']['INSTANCES']['KEEP_GREATER_THAN']
        keep_cat_count_less_than = self.config['FILTER_BY']['CAT']['INSTANCES']['KEEP_LESS_THAN']

        print("keep_cat_names")
        print(keep_cat_names)

        print("keep_cat_ids")
        print(keep_cat_ids)

        for cat_dict in self.coco.dataset['categories']:

            filter_out_this_cat_id = False

            cat_id = cat_dict['id']
            cat_name = cat_dict['name']
            cat_count = self.cat_id_to_instance_count[cat_id]

            if keep_cat_names:
                if cat_name not in keep_cat_names:
                    print(f'filtering out {cat_name} b/c not in keep_cat_names')
                    filter_out_this_cat_id = True
            if keep_cat_ids:
                if cat_id not in keep_cat_ids:
                    print(f'filtering out {cat_id} b/c not in keep_cat_ids')
                    filter_out_this_cat_id = True
            if keep_cat_count_greater_than:
                if cat_count <= keep_cat_count_greater_than:
                    print(f'filtering out {cat_id} b/c cat count {cat_count} is <= {keep_cat_count_greater_than}')
                    filter_out_this_cat_id = True
            if keep_cat_count_less_than:
                if cat_count >= keep_cat_count_less_than:
                    print(f'filtering out {cat_id} b/c cat count {cat_count} is >= {keep_cat_count_less_than}')
                    filter_out_this_cat_id = True

            if filter_out_this_cat_id:
                self.cat_filter.add(cat_id)

    def build_img_filter(self):
        keep_img_width_greater_than = self.config['FILTER_BY']['IMG']['WIDTH']['KEEP_GREATER_THAN']
        keep_img_width_less_than = self.config['FILTER_BY']['IMG']['WIDTH']['KEEP_LESS_THAN']
        keep_img_height_greater_than = self.config['FILTER_BY']['IMG']['HEIGHT']['KEEP_GREATER_THAN']
        keep_img_height_less_than = self.config['FILTER_BY']['IMG']['HEIGHT']['KEEP_LESS_THAN']

        for img_dict in self.coco.dataset['images']:

            filter_out_this_img_id = False

            img_id = img_dict['id']
            img_width = int(img_dict['width'])
            img_height = int(img_dict['height'])

            if keep_img_width_greater_than:
                if img_width <= keep_img_width_greater_than:
                    print(f'Adding img {img_id} to filter b/c width <= {keep_img_width_greater_than}')
                    filter_out_this_img_id = True
            if keep_img_width_less_than:
                if img_width >= keep_img_width_less_than:
                    print(f'Adding img {img_id} to filter b/c width >= {keep_img_width_less_than}')
                    filter_out_this_img_id = True
            if keep_img_height_greater_than:
                if img_height <= keep_img_height_greater_than:
                    print(f'Adding img {img_id} to filter b/c height <= {keep_img_height_greater_than}')
                    filter_out_this_img_id = True
            if keep_img_height_less_than:
                if img_height >= keep_img_height_less_than:
                    print(f'Adding img {img_id} to filter b/c height >= {keep_img_height_less_than}')
                    filter_out_this_img_id = True

            if filter_out_this_img_id:
                self.img_filter.add(img_id)

    def build_ann_filter(self):
        keep_ann_width_greater_than = self.config['FILTER_BY']['ANN']['WIDTH']['KEEP_GREATER_THAN']
        keep_ann_width_less_than = self.config['FILTER_BY']['ANN']['WIDTH']['KEEP_LESS_THAN']
        keep_ann_height_greater_than = self.config['FILTER_BY']['ANN']['HEIGHT']['KEEP_GREATER_THAN']
        keep_ann_height_less_than = self.config['FILTER_BY']['ANN']['HEIGHT']['KEEP_LESS_THAN']

        print(f"The cat filter used to build the ann filter is..")
        print(self.cat_filter)

        for ann_dict in self.dataset['annotations']:

            filter_out_this_ann_id = False

            ann_id = ann_dict['id']
            ann_width = int(ann_dict['bbox'][2])
            ann_height = int(ann_dict['bbox'][3])
            ann_img_id = int(ann_dict['image_id'])
            ann_cat_id = int(ann_dict['category_id'])

            if keep_ann_width_greater_than:
                if ann_width <= keep_ann_width_greater_than:
                    filter_out_this_ann_id = True
            if keep_ann_width_less_than:
                if ann_width >= keep_ann_width_less_than:
                    filter_out_this_ann_id = True
            if keep_ann_height_greater_than:
                if ann_height <= keep_ann_height_greater_than:
                    filter_out_this_ann_id = True
            if keep_ann_height_less_than:
                if ann_height >= keep_ann_height_less_than:
                    filter_out_this_ann_id = True
            if not filter_out_this_ann_id: # check first to avoid many list comarisons
                if ann_cat_id in self.cat_filter:
                    filter_out_this_ann_id = True
            if not filter_out_this_ann_id: # check first to avoid many list comarisons
                if ann_img_id in self.img_filter:
                    filter_out_this_ann_id = True

            if filter_out_this_ann_id:
                self.ann_filter.add(ann_id)

    def add_empty_imgs_to_filter(self):
        '''
        Adds the ids of empty images to the image filter.
        '''
        anns_keep = list(self.ann_ids - self.ann_filter)
        imgs_w_anns = set()

        # get the ids for images that have annotations 
        # (this should be applied after setting the ann filter, since some images might become empty based on 
        #  after that filter is set)

        for ann_dict in self.coco.loadAnns(ids=anns_keep):
            imgs_w_anns.add(ann_dict['image_id'])

        # add empty image ids to the image filter
        empty_img_ids = list(self.img_ids - imgs_w_anns)
        for empty_img_id in empty_img_ids:
            self.img_filter.add(empty_img_id)

        print(f'There were {len(anns_keep)} annotations to keep.')
        print(f'There were {len(empty_img_ids)} empty images that were added to the filter.')

    def filter_dataset(self):
        '''
        Build the filtered dataset dictionaries from the filters (i.e. the sets of ids to filter out).
        '''
        filtered_ann_ids = self.ann_ids - self.ann_filter
        filtered_img_ids = self.img_ids - self.img_filter
        filtered_cat_ids = self.cat_ids - self.cat_filter

        anns = self.coco.loadAnns(
            ids=self.coco.getAnnIds(
                catIds=filtered_cat_ids,
                imgIds=filtered_img_ids
            )
        )
        imgs = self.coco.loadImgs(
            ids=filtered_img_ids
        )
        cats = self.build_categories()
        self.dataset = {"categories": cats, "images": imgs, "annotations": anns}

    def build_categories(self):
        '''
        Builds a new (filtered) list of category dictionaries based on the cat_filter.
        '''
        cats_to_keep = self.cat_ids - self.cat_filter
        new_categories = []
        for cat_dict in self.dataset['categories']:
            if cat_dict['id'] in cats_to_keep:
                new_cat_dict = deepcopy(cat_dict)
                new_categories.append(new_cat_dict)
        return new_categories

    def reset_coco(self):
        coco = deepcopy(self.coco)
        # create index
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        coco.anns = anns
        coco.imgToAnns = imgToAnns
        coco.catToImgs = catToImgs
        coco.imgs = imgs
        coco.cats = cats
        coco.dataset = self.dataset
        self.coco = coco
        self.cat_ids = set(self.coco.cats.keys())
        self.img_ids = set(self.coco.imgs.keys())
        self.ann_ids = set(self.coco.anns.keys())

    def reindex_dataset(self, idx_base=1):
        '''
        Reindexes a coco-formatted dataset dictionary
        '''
        anns = self.dataset['annotations']
        imgs = self.dataset['images']
        cats = self.dataset['categories']

        id2name = {str(cat['id']): cat['name'] for i, cat in enumerate(cats)}
        id2id = {str(cat['id']): i+idx_base for i, cat in enumerate(cats)}

        print(f"remap id2id")
        print(id2id)

        print(f"remap id2name")
        print(id2name)

        new_cats = [
            {
                "supercategory": cat['supercategory'],
                "id": id2id[str(cat['id'])],
                "name": cat['name'],
            } 
            for cat in cats
            ]

        new_imgs = deepcopy(imgs)

        new_anns = [
            {
                "segmentation": ann['segmentation'],
                "bbox": ann['bbox'],
                "area": ann['area'],
                "id": ann['id'],
                "image_id": ann['image_id'],
                "category_id": id2id[str(ann['category_id'])],
                "iscrowd": 0 # matters for coco_eval
            } 
            for ann in anns
            ]
        
        self.dataset = {"categories": new_cats, "images": new_imgs, "annotations": new_anns}

    def remap_dataset(self):
        '''
        Remaps the categories and annotations of a coco-formatted dataset dictionary based on self.config.
        '''
        id2id = self.config['MAP_BY']['ID_TO_ID']
        id2name = self.config['MAP_BY']['ID_TO_NAME']
        anns = self.dataset['annotations']
        imgs = self.dataset['images']
        cats = self.dataset['categories']

        print(f"remap id2id")
        print(id2id)

        print(f"remap id2name")
        print(id2name)

        new_cats = [
            {
                "supercategory": cat['supercategory'],
                "id": cat['id'],
                "name": id2name[str(cat['id'])],
            } 
            for cat in cats if str(cat['id']) in list(id2id.values())
            ]

        new_imgs = deepcopy(imgs)

        new_anns = [
            {
                "segmentation": ann['segmentation'],
                "bbox": ann['bbox'],
                "area": ann['area'],
                "id": ann['id'],
                "image_id": ann['image_id'],
                "category_id": id2id[str(ann['category_id'])],
                "iscrowd": 0 # matters for coco_eval
            }
            for ann in anns if str(ann['category_id']) in list(id2id.keys())
            ]
        
        self.dataset = {"categories": new_cats, "images": new_imgs, "annotations": new_anns}
        