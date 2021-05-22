"""
"""
# Standard Library imports:
import io as _io
import json
import os
from pathlib import Path

# 3rd Party imports:
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm

import lib.image as xv_img
import lib.chip as xv_chip
import lib.viz as xv_vz

"""
Functions for generating xView meta-data.
"""

def generate_info():
    info_json = {
        "description": "XView Dataset",
        "url": "http://xviewdataset.org/",
        "version": "1.0",
        "year": 2018,
        "contributor": "Defense Innovation Unit Experimental (DIUx)",
        "date_created": "2018/02/22",
    }
    return info_json

def generate_licenses():
    licenses = []
    license = {
        "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
    }
    licenses.append(license)
    return licenses

def generate_categories(input_json):
    class_ids = []
    categories = []
    print(input_json)
    with open(input_json, "r") as coco_file:
        anns_file = json.load(coco_file)
        categories = anns_file["categories"]
        for entry in categories:
            class_ids.append(entry["id"])
    return categories, class_ids

def generate_images_and_annotations(
    class_ids,
    input_json,
    source_imgs_dir,
    dest_imgs_dir,
    img_tag,
    vis_out,
    width=512,
    height=512,
):

    coco = COCO(input_json)

    ann_len = len(coco.anns)

    all_coords = np.zeros((ann_len, 4))
    all_chips = np.zeros(ann_len, dtype="object")
    all_classes = np.zeros(ann_len)

    i = 0
    for ann_id in coco.anns.keys():
        ann = coco.anns[ann_id]
        img_id = ann["image_id"]

        all_chips[i] = img_id

        bbox = ann["bbox"]
        coord = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        all_coords[i] = np.array([int(num) for num in coord])

        all_classes[i] = ann["category_id"]
        i += 1

    images = []
    file_index = 1
    annotation_index = 1
    annotations = []

    print(f"Chipping {len(coco.imgs.keys())} images")


    # debug_img_ids = [list(coco.imgs.keys())[5]] # TODO: DEBUG REMOVE
    # print("\nDEBUG IMG IDS", debug_img_ids)  # TODO: DEBUG REMOVE
    # for img_id in tqdm(debug_img_ids):  # TODO: DEBUG REMOVE

    for img_id in tqdm(coco.imgs.keys()):
        img = coco.loadImgs([img_id])[0]
        I = io.imread("%s/%s" % (source_imgs_dir, img["file_name"]))

        # xv_vz.vis_coco_anns(I, coco, [img], vis_out)

        chip_name = img["id"]
        coords = all_coords[all_chips == chip_name]
        classes = all_classes[all_chips == chip_name].astype(np.int64)
        chips, chip_boxes, chip_classes = xv_chip.chip_image_w_overlap(I, coords, classes)

        for i in range(len(chips)):
            image_dict = {}
            image_dict["license"] = 1
            image_dict["file_name"] = (
                "XVIEW_" + img_tag + "_" + str(file_index).zfill(12) + ".jpg"
            )
            image_dict["coco_url"] = ""
            image_dict["width"] = 512
            image_dict["height"] = 512
            image_dict["date_captured"] = "2018-02-22 00:00:00"
            image_dict["flickr_url"] = ""
            image_dict["id"] = file_index
            image_dict["parent_id"] = img_id # Add the parent image id for domain adaptation stuff
            images.append(image_dict)
            new_image = xv_img.convertToJpeg(chips[i])
            with open(
                os.path.join(
                    dest_imgs_dir,
                    "XVIEW_" + img_tag + "_" + str(file_index).zfill(12) + ".jpg",
                ),
                "wb",
            ) as image_file:
                image_file.write(new_image)

            for j in range(len(chip_boxes[i])):
                class_id = int(chip_classes[i][j])

                if not class_id in class_ids:
                    continue

                box = chip_boxes[i][j]
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                if x_min == y_min == x_max == y_max == 0:
                    continue

                x, y = int(x_min), int(y_min)
                w, h = int(x_max - x_min), int(y_max - y_min)

                annotation_dict = {}
                annotation_dict["bbox"] = [x, y, w, h]
                annotation_dict["segmentation"] = [
                    [x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]
                ]
                annotation_dict["area"] = w * h
                annotation_dict["iscrowd"] = 0
                annotation_dict["image_id"] = file_index
                annotation_dict["category_id"] = class_id
                annotation_dict["id"] = str(annotation_index)

                annotations.append(annotation_dict)
                annotation_index += 1
            file_index += 1

    return images, annotations