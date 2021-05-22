"""
"""
# Standard Library imports:
import io as _io
import json
import os
from pathlib import Path, PosixPath
from typing import List

# 3rd Party imports:
from PIL import Image
import numpy as np
import skimage.io as io
from tqdm import tqdm

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: E402, E501, F401
from h4dlib.config import h4dconfig
from h4dlib.data.comsat_helpers import ComsatToCoco
from h4dlib.data.cocohelpers import CocoClassDistHelper, CocoJsonBuilder
from h4dlib.data.comsat_helpers import split
from pycocotools.coco import COCO
import h4dlib.data.image_util as wv

def create_split(coco_json, output_path, output_json_name, train_txt, val_txt):
    """
    Creates train/val split for the coco-formatted dataset defined by
    input_json. params: input_json: full path or Path object to coco-formatted
    input json file. output_path: full path or Path object to directory where
    outputted json will be saved. output_json_name:
    """
    coco = COCO(coco_json)

    train_img_ids = {line.split(".")[0] for line in open(train_txt)}
    val_img_ids = {line.split(".")[0] for line in open(val_txt)}

    # train_img_ids, val_img_ids = split(
    #     coco,
    #     input_csv,
    #     img_min_dim
    # )

    # Generate coco-formatted json's for train and val:
    def generate_coco_json(coco, split_type, img_ids):
        coco_builder = CocoJsonBuilder(
            coco.dataset["categories"],
            dest_path=output_path,
            dest_name=output_json_name.format(split_type)
        )
        for idx, img_id in enumerate(img_ids):
            coco_builder.add_image(coco.imgs[int(img_id)], coco.imgToAnns[int(img_id)])
        coco_builder.save()

    generate_coco_json(coco, "train", list(train_img_ids))
    generate_coco_json(coco, "val", list(val_img_ids))
    return coco

def generate_images_and_annotations(
    class_ids,
    input_json,
    source_imgs_dir,
    dest_imgs_dir,
    img_tag,
    fixed_size=512
):

    coco = COCO(input_json)

    img_ids = list(coco.imgs.keys())
    # get the corresponding anns
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    ann_len = len(ann_ids)

    all_coords = np.zeros((ann_len, 4))
    all_chips = np.zeros(ann_len, dtype="object")
    all_classes = np.zeros(ann_len)

    i = 0
    for ann_id in ann_ids:
        ann = coco.loadAnns(ids=[ann_id])[0]
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
    thin_filter_counter = 0

    print("Chipping images")
    for img_id in tqdm(img_ids):

        img = coco.loadImgs(img_id)[0]
        I = io.imread("%s/%s" % (source_imgs_dir, img["file_name"]))

        chip_name = img["id"]
        coords = all_coords[all_chips == chip_name]
        classes = all_classes[all_chips == chip_name].astype(np.int64)

        chips, chip_width, chip_height, chip_boxes, chip_classes = wv.chip_image(
            I, coords, classes, fixsz=fixed_size
        )

        for i in range(len(chips)):
            image_dict = {}
            image_dict["license"] = 1
            image_dict["file_name"] = (
                "COMSAT_" + img_tag + "_" + str(file_index).zfill(12) + ".jpg"
            )
            image_dict["coco_url"] = ""
            image_dict["width"] = chip_width
            image_dict["height"] = chip_height
            image_dict["date_captured"] = "2018-02-22 00:00:00"
            image_dict["flickr_url"] = ""
            image_dict["id"] = file_index
            image_dict["parent_id"] = img_id # Add the parent image id for domain adaptation stuff
            images.append(image_dict)
            new_image = convertToJpeg(chips[i])
            with open(
                os.path.join(
                    dest_imgs_dir,
                    "COMSAT_" + img_tag + "_" + str(file_index).zfill(12) + ".jpg",
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

                # filter out thin boxes
                if w < 5 or h < 5:
                    thin_filter_counter += 1
                    continue

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
        
    print(f"\n{thin_filter_counter} boxes were filtered out for being too thin. \n")

    return images, annotations


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


def convertToJpeg(im):
    """
    (copied from tfr_util.py, so we don't have to import tensorflow)
    Converts an image array into an encoded JPEG string.
    Args:
        im: an image array
    Output:
        an encoded byte string containing the converted JPEG image.
    """
    with _io.BytesIO() as f:
        im = Image.fromarray(im)
        im.save(f, format="JPEG")
        return f.getvalue()

def main():
    """Main function"""

    IMG_MIN_DIM = 1 # any parent image below this size gets excluded (this happens before chipping)
    MIN_CHIP_SIZE = -1 # the min size that parent images will get chipped to.  Max chip size is (2 * MIN_CHIP_SIZE - 1)
    FIXED_CHIP_SIZE = 512 # the size that parent images will get chipped to

    # ROOT PATHS
    DATA_ROOT: Path = Path('/home/laielli/data')
    DATASET_ROOT: Path = DATA_ROOT / 'Xview'
    # DOWNLOAD_ROOT: Path = COMSAT_ROOT / 'seq/MIZZOU'
    VANILLA_ROOT: Path = DATASET_ROOT / 'coco_limdis'
    CHIP_ROOT: Path = DATASET_ROOT / f'coco_{FIXED_CHIP_SIZE}'

    # I/O FILE PATHS
    VANILLA_JSON = VANILLA_ROOT / 'xview_coco_limdis_v0.json'
    # SPLIT_CSV = COMSAT_ROOT / 'comsat_images.csv'
    SPLIT_TXT_TRAIN = DATASET_ROOT / 'xview_public_train_labels_020419.txt'
    SPLIT_TXT_VAL = DATASET_ROOT / 'xview_public_val_labels_020419.txt'

    # OTHER SETTINGS
    # IMG_EXT = "png" # the file extension to look for in the download folder
    SPLIT_JSON= "xview_coco_public_{}.json" # used for train/val splits in both vanilla and chip sets

    # split into train/val
    original_coco = create_split(
        VANILLA_JSON,
        VANILLA_ROOT,
        SPLIT_JSON,
        SPLIT_TXT_TRAIN,
        SPLIT_TXT_VAL
        )

    # chip (based on min size)
    for datatype in ["train", "val"]:
        INPUT_JSON = VANILLA_ROOT / SPLIT_JSON.format(datatype)
        DEST_JSON = CHIP_ROOT / SPLIT_JSON.format(datatype)
        DEST_IMGS_DIR = CHIP_ROOT / datatype
        IMG_TAG = datatype

        if not os.path.exists(DEST_IMGS_DIR):
            os.makedirs(DEST_IMGS_DIR)

        print(f"Processing {datatype}")
        with open(DEST_JSON, "w") as coco_file:
            root_json = {}
            categories_json, class_ids = generate_categories(INPUT_JSON)
            root_json["categories"] = categories_json
            root_json["info"] = generate_info()
            root_json["licenses"] = generate_licenses()
            images, annotations = generate_images_and_annotations(
                class_ids,
                INPUT_JSON,
                VANILLA_ROOT,
                DEST_IMGS_DIR,
                IMG_TAG,
                fixed_size=FIXED_CHIP_SIZE
            )
            root_json["images"] = images
            root_json["annotations"] = annotations
            coco_file.write(json.dumps(root_json))


if __name__ == "__main__":
    main()
