# Standard Library imports:
import io as _io
import json
import os
from pathlib import Path

import lib.data_gen as xv_dg
import lib.viz as xv_vz


def main(datatype):
    VERSION = "xview_coco_v2"
    NAME = "coco_chipped_overlap"
    DATADIR: Path = Path("/home/laielli/data")
    INPUT_JSON = DATADIR / f"Xview/coco_vanilla/{VERSION}_{datatype}.json"
    DST_JSON = DATADIR / "Xview" / NAME / f"{VERSION}_{datatype}_chipped.json"
    SOURCE_IMGS_DIR = DATADIR / "Xview/coco_vanilla"
    DST_DIR = DATADIR / "Xview" / NAME
    DST_IMGS_DIR = DST_DIR / datatype
    DST_VIZ = DST_DIR / "viz"
    IMG_TAG = datatype

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
    if not os.path.exists(DST_IMGS_DIR):
        os.makedirs(DST_IMGS_DIR)
    if not os.path.exists(DST_VIZ):
        os.makedirs(DST_VIZ)

    print(f"Processing {datatype}")
    with open(DST_JSON, "w") as coco_file:
        root_json = {}
        categories_json, class_ids = xv_dg.generate_categories(INPUT_JSON)
        root_json["categories"] = categories_json
        root_json["info"] = xv_dg.generate_info()
        root_json["licenses"] = xv_dg.generate_licenses()
        images, annotations = xv_dg.generate_images_and_annotations(
            class_ids,
            INPUT_JSON,
            SOURCE_IMGS_DIR,
            DST_IMGS_DIR,
            IMG_TAG,
            DST_VIZ,
            width=512,
            height=512,
        )
        root_json["images"] = images
        root_json["annotations"] = annotations
        coco_file.write(json.dumps(root_json))
    xv_vz.visualize_coco_anns(
        DST_JSON,
        DST_IMGS_DIR,
        DST_VIZ,
        data_type=datatype,
        num_rand_images=100,
    )

if __name__ == "__main__":
    main("val")
    main("train")
