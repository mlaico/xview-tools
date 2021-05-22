"""
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import skimage.io as io
from pycocotools.coco import COCO
import random

"""
Functions for visualizing images and annotations.
"""


def show_boxes(boxes, color):
    """
    Display the specified boxes.
    :param boxes (list of boxes): annotations to display
    :return: None
    """
    if len(boxes) == 0:
        return 0

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    colors = []
    for box in boxes:
        # c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        # [bbox_x, bbox_y, bbox_w, bbox_h] = box
        # poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        [x_min, y_min, x_max, y_max] = box
        poly = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
        np_poly = np.array(poly).reshape((4,2))
        polygons.append(Polygon(np_poly))
        colors.append(color)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_unlabeled_boxes(boxes):
    color = [0.4622517545478549, 0.7561357322671833, 0.8822836702498775] # light blue
    # print("unlabeled color: ", color)
    show_boxes(boxes, color=color)

def show_pred_boxes(boxes):
    color = [0.9437141046482208, 0.63501777919006, 0.6784018434261119] # light red
    # print("pred color: ", color)
    show_boxes(boxes, color=color)

def show_match_boxes(boxes):
    color = [0.48272915054409526, 0.9798221446387896, 0.47497991850654653] # bright green
    # print("match color: ", color)
    show_boxes(boxes, color=color)

def show_region_boxes(boxes):
    color = [1.0, 1.0, 1.0] # white
    # print("region color: ", color)
    show_boxes(boxes, color=color)

def visualize_coco_anns(
    ann_file,
    img_dir,
    out_dir,
    num_rand_images=1,
    data_type="train",
    image_ids=None,
    cat_ids=None
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print("Loading ann file:",ann_file)
    coco = COCO(ann_file)
    # all_img_ids = coco.getImgIds(catIds=cat_ids)
    all_img_ids = coco.getImgIds()
    print(f"{len(all_img_ids)} img ids:", all_img_ids)
    img_ids = image_ids if image_ids else random.sample(all_img_ids, num_rand_images)
    imgs = coco.loadImgs(img_ids)
    for img in imgs:
        img_id = img['id']
        file_name = img['file_name']
        # I = io.imread(img_dir / data_type / file_name)
        I = io.imread(img_dir / file_name)
        # annIds = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        plt.axis('off')
        plt.imshow(I)
        coco.showAnns(anns)
        plt.savefig(out_dir / f'{data_type}_imgid_{img_id}.png')
        plt.clf()

def vis_coco_anns(
    I,
    coco,
    imgs,
    out_dir,
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for img in imgs:
        img_id = img['id']
        file_name = img['file_name']
        # annIds = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        plt.axis('off')
        plt.imshow(I)
        coco.showAnns(anns)
        plt.savefig(out_dir / f'vanilla_imgid_{img_id}.png')
        plt.clf()