"""
cocohelpers is a module with helper classes and functions related to the MS
COCO API. Includes helpers for building COCO formatted json, inspecting class
distribution, and generating a train/val split.
"""
# Standard Library imports:
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, List, OrderedDict, Tuple
import copy

# 3rd Party imports:
import numpy as np
from pycocotools.coco import COCO

__all__ = ["CocoJsonBuilder", "COCOShrinker", "CocoClassDistHelper", "split"]


class CocoJsonBuilder(object):
    """
    A class used to help build coco-formatted json from scratch.
    """

    def __init__(self, categories: List[Dict[str, object]], dest_path="", dest_name=""):
        """
        Args:
            categories: this can be the COCO.dataset['categories'] property if you
                are building a COCO json derived from an existing COCO json and don't
                want to modify the classes. It's a list of dictionary objects. Each dict has
                three keys: "id":int = category id, "supercatetory": str = name of parent
                category, and a "name": str = name of category.
            dest_path: str or pathlib.Path instance, holding the path to directory where
                the new COCO formatted annotations
            file (dest_name) will be saved.
            dest_name: str of the filename where the generated json will be saved to.
        """
        self.categories = categories
        self.dest_path = Path(dest_path)
        self.dest_name = dest_name
        self.images = []
        self.annotations: List[Dict[str, object]] = []
        assert self.dest_path.exists(), f"dest_path: '{self.dest_path}' does not exist"
        assert (
            self.dest_path.is_dir()
        ), f"dest_path: '{self.dest_path}' is not a directory"

    def generate_info(self) -> Dict[str, str]:
        """returns: dictionary of descriptive info about the dataset."""
        info_json = {
            "description": "XView Dataset",
            "url": "http://xviewdataset.org/",
            "version": "1.0",
            "year": 2018,
            "contributor": "Defense Innovation Unit Experimental (DIUx)",
            "date_created": "2018/02/22",
        }
        return info_json

    def generate_licenses(self) -> Dict[str, str]:
        """Returns the json hash for the licensing info."""
        return [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ]

    def add_image(self, img: Dict[str, Any], annotations: List[Dict]) -> None:
        """
        Add an image and it's annotations to the coco json.

        Args:
            img: A dictionary of image attributes. This gets added verbatim to the
                json, so in typical use cases when you are building a coco json from an
                existing coco json, you would just pull the entire coco.imgs[img_id]
                object and pass it as the value for this parameter.
            annotations: annotations of the image to add. list of dictionaries.
                Each dict is one annotation, it contains all the properties of the
                annotation that should appear in the coco json. For example, when using
                this json builder to build JSON's for a train/val split, the
                annotations can be copied straight from the coco object for the full
                dataset, and passed into this parameter.

        Returns: None
        """
        self.images.append(img)
        for ann in annotations:
            self.annotations.append(ann)

    def get_json(self) -> Dict[str, object]:
        """Returns the full json for this instance of coco json builder."""
        root_json = {}
        root_json["categories"] = self.categories
        root_json["info"] = self.generate_info()
        root_json["licenses"] = self.generate_licenses()
        root_json["images"] = self.images
        root_json["annotations"] = self.annotations
        return root_json

    def save(self) -> None:
        """Saves the json to the dest_path/dest_name location."""
        file_path = self.dest_path / self.dest_name
        print(f"Writing output to: '{file_path}'")
        root_json = self.get_json()
        with open(file_path, "w") as coco_file:
            coco_file.write(json.dumps(root_json))


class COCOShrinker:
    """Shrinker takes an MS COCO formatted dataset and creates a tiny version of it."""

    def __init__(self, dataset_path: Path) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path

    def shrink(self, target_filename: str, size: int = 512) -> None:
        """
        Create a toy sized version of dataset so we can use it just for testing if code
        runs, not for real training.

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Nothing, but the output dataset is saved to disk in the same directory
            where the input .json lives, with the same filename but with "_tiny" added
            to the filename.
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(
            f"Creating subset of {self.dataset_path}, of size: {size}, at: {dest_path}"
        )
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"], dest_path.parent, dest_path.name
        )
        subset_img_ids = coco.getImgIds()[:size]
        for img_id in subset_img_ids:
            builder.add_image(coco.imgs[img_id], coco.imgToAnns[img_id])
        builder.save()
        return dest_path


class COCOSubset:
    """Subset takes an MS COCO formatted dataset and creates a subset according to COCO parent ids provided as a lsit."""

    def __init__(self, dataset_path: Path) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path

    def shrink(
        self, target_filename: str, subset_par_ids: list, size: int = 512
    ) -> None:
        """
        Create a toy sized version of dataset so we can use it just for testing if code
        runs, not for real training.

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Nothing, but the output dataset is saved to disk in the same directory
            where the input .json lives, with the same filename but with "_tiny" added
            to the filename.
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(
            f"Creating subset of {self.dataset_path}, of size: {size}, at: {dest_path}"
        )
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"], dest_path.parent, dest_path.name
        )

        # subset_img_ids = coco.getImgIds()[:size]

        # create index to map from parent id to list of image ids
        parent_id_to_img_ids = {}
        imgs = coco.dataset["images"]
        parent_ids = set()
        for img in imgs:
            parent_ids.add(img["parent_id"])
        for pid in parent_ids:
            parent_id_to_img_ids[pid] = []
        for img in imgs:
            parent_id_to_img_ids[img["parent_id"]].append(img["id"])

        for par_id in subset_par_ids:
            for img_id in parent_id_to_img_ids[par_id]:
                builder.add_image(coco.imgs[img_id], coco.imgToAnns[img_id])
        builder.save()
        return dest_path


class COCORedundant:
    """Creates a version of xview that creates perfect redundancy by replacing vxiew images with a single empty image (no labels)."""

    def __init__(self, dataset_path: Path) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path

    def redundify(
        self, target_filename: str, redundant_img_fn: str, percent_redundant: float
    ) -> None:
        """

        Args:
            target_filename: filename to save the new dataset to.
            redundant_img_fn: name of the image file that will be used to make redundant copies
            percent_redundant: Percentage of images to make redundant

        Returns: Nothing
        """
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(f"Creating subset of {self.dataset_path} at: {dest_path}")
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"], dest_path.parent, dest_path.name
        )

        # get all image ids
        all_chip_ids = coco.getImgIds()

        num_samples = int(percent_redundant * len(all_chip_ids))

        sampled_chip_ids = random.sample(all_chip_ids, num_samples)

        empty_anns = []

        # make each sampled chip redundant; add to builder
        for chipid in sampled_chip_ids:
            cocoimg = coco.imgs[chipid]
            cocoimg["file_path"] = redundant_img_fn
            builder.add_image(cocoimg, empty_anns)

        # add the rest of the chips to the builder
        rest_of_chip_ids = list(set(all_chip_ids) - set(sampled_chip_ids))
        for chipid in rest_of_chip_ids:
            builder.add_image(coco.imgs[chipid], coco.imgToAnns[chipid])

        builder.save()

        print(
            f"Total chips in new dataset: {len(builder.images)} (should match the original size of {len(all_chip_ids)})"
        )
        return dest_path


class COCOVideoFrames:
    """Creates a version of xview that mimics sequential video frame redundancy by making perfectly redundant copies of image chips."""

    def __init__(self, dataset_path: Path) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path

    def vidify(
        self, target_filename: str, num_chips: int, num_copies: int, debug: bool = False
    ) -> None:
        """

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Nothing
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(f"Creating subset of {self.dataset_path} at: {dest_path}")
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"], dest_path.parent, dest_path.name
        )

        # subset_img_ids = coco.getImgIds()[:size]

        # create index to map from parent id to list of image ids
        parent_id_to_chipids = {}
        imgs = coco.dataset["images"]
        parent_ids = set()
        for img in imgs:
            parent_ids.add(img["parent_id"])

        for pid in parent_ids:
            parent_id_to_chipids[pid] = []
        for img in imgs:
            parent_id_to_chipids[img["parent_id"]].append(img["id"])

        # initialize counters
        chip_counter = 0
        ann_counter = 0
        num_chips = 4
        num_copies = 10

        # DEBUG COUNTERS
        pid_ = 2
        chipid_ = 2
        i_ = 2

        # DEBUG RANGES
        # pids_ = list(range(2))
        # chipids_ = list(range(2))
        # is_ = list(range(2))

        # for each parents id
        for pid in parent_ids:

            this_par_chipids = parent_id_to_chipids[pid]

            # randomly sample num_chips chips
            random_chipids = random.sample(this_par_chipids, num_chips)

            # for each chip
            for chipid in random_chipids:
                img = coco.imgs[chipid]

                # DEBUG
                if debug:
                    if (
                        pid in list(parent_ids)[:pid_]
                        and chipid in random_chipids[:chipid_]
                    ):
                        print("Original coco image...")
                        print(img)
                        print("")

                # for num_copies
                for i in range(num_copies):
                    new_cocoim = copy.deepcopy(img)
                    new_cocoim["id"] = chip_counter

                    # DEBUG
                    if debug:
                        if (
                            pid in list(parent_ids)[:pid_]
                            and chipid in random_chipids[:chipid_]
                        ) and i in list(range(i_)):
                            print("New coco image...")
                            print(new_cocoim)
                            print("")

                    anns = coco.imgToAnns[chipid]

                    # DEBUG
                    if debug and anns:
                        if (
                            pid in list(parent_ids)[:pid_]
                            and chipid in random_chipids[:chipid_]
                        ) and i in list(range(i_)):
                            print("Original annotation (first)...")
                            print(anns[0])
                            print("")

                    new_anns = []
                    for ann in anns:
                        new_ann = copy.deepcopy(ann)
                        new_ann["image_id"] = chip_counter
                        new_ann["id"] = ann_counter
                        new_anns.append(new_ann)
                        ann_counter += 1

                    builder.add_image(new_cocoim, new_anns)

                    # DEBUG
                    if debug and new_anns:
                        if (
                            pid in list(parent_ids)[:pid_]
                            and chipid in random_chipids[:chipid_]
                        ) and i in list(range(i_)):
                            print("New annotations...")
                            for ann in new_anns[:3]:
                                print(ann)
                            print("")

                    chip_counter += 1

        builder.save()

        print(f"total chips created: {len(builder.images)}")
        return dest_path


class COCOBoxNoise:
    """Creates a version of xview that induces synthetic noise on the spatial accuracy of the bounding boxes."""

    def __init__(self, dataset_path: Path) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path

    def apply_box_shift(self, ann: dict, shift_vec: tuple) -> None:
        """
        Args:
            ann: A single coco annotation
            shift_vec: An (x,y) tuple where each coord controls the distance
            to shift the box in each dimension in terms of a factor of box width/height
            e.g. (1,0) causes a horizontal shift of '1' box width to the right and zero vertical shift

        Returns: Nothing
        """
        x, y, w, h = ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h

        # Calc shift
        x_shift = int(shift_vec[0] * w)
        y_shift = int(shift_vec[1] * h)

        # Apply shift
        ann["bbox"] = [x + x_shift, y + y_shift, w, h]
        ann["segmentation"] = [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]]

    def random_shift(self, shift_coeff: float) -> tuple:
        """
        Args:
            shift_coeff: A float the controls the magnitude of the synthetic box shift

        Returns: A tuple that desribes the shift in each of the x and y directions
        """

        shift_x = shift_coeff * random.uniform(-1, 1)
        shift_y = shift_coeff * random.uniform(-1, 1)

        return (shift_x, shift_y)

    def adjust_if_out_of_bounds(self, ann: dict, img: dict) -> None:
        """
        Handles the case when a bounding box annotation breaches the image boundaries

        Args:
            ann: A single coco annotation
            img: The coco image corresponding to the annotation

        Returns: Nothing
        """

        im_w, im_h = img["width"], img["height"]

        x, y, w, h = ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h

        # check case where:

        # box completely out of bounds
        if (x_min >= im_w or x_max <= 0) or (y_min >= im_h or y_max <= 0):
            ann = {}
            return

        # box breaks left boundary
        if x_min < 0:
            x, x_min = 0, 0

        # box breaks top boundary
        if y_min < 0:
            y, y_min = 0, 0

        # box breaks right boundary
        if x_max > im_w:
            x_max = im_w
            w = x_max - x_min

        # box breaks bottom boundary
        if y_max > im_h:
            y_max = im_h
            h = y_max - y_min

        ann["bbox"] = [x, y, w, h]
        ann["segmentation"] = [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]]

        return

    def box_noisify(
        self,
        target_filename: str,
        box_noise_coeff: float,
        chip_ratio: float,
        debug: bool = False,
    ) -> None:
        """

        Args:
            name: filename to save the tiny dataset to.
            noise_coeff: Value that controls magnitude of box shift (can be greater than 1, or less than 0).
            chip_ratio: the ratio of chips within each parent image that receive noise (btw 0 and 1)
            debug: Setting to 'True' activates helpful print statements

        Returns: Nothing
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(f"Creating subset of {self.dataset_path} at: {dest_path}")
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"], dest_path.parent, dest_path.name
        )

        # create index to map from parent id to list of image ids
        parent_id_to_chipids = {}
        imgs = coco.dataset["images"]
        parent_ids = set()
        for img in imgs:
            parent_ids.add(img["parent_id"])
        for pid in parent_ids:
            parent_id_to_chipids[pid] = []
        for img in imgs:
            parent_id_to_chipids[img["parent_id"]].append(img["id"])

        # DEBUG COUNTERS
        pid_ = 2
        chipid_ = 2
        i_ = 2

        # for each parents id
        for pid in parent_ids:

            this_par_chipids = parent_id_to_chipids[pid]

            num_chips = int(chip_ratio * len(this_par_chipids))

            # randomly sample num_chips chips
            noisy_chipids = random.sample(this_par_chipids, num_chips)

            other_chipids = list(set(this_par_chipids) - set(noisy_chipids))

            # for each chip
            for chipid in noisy_chipids:
                img = coco.imgs[chipid]

                # DEBUG
                if debug:
                    if (
                        pid in list(parent_ids)[:pid_]
                        and chipid in noisy_chipids[:chipid_]
                    ):
                        print("Original coco image...")
                        print(img)
                        print("")

                anns = coco.imgToAnns[chipid]

                # DEBUG
                if debug and anns:
                    if (
                        pid in list(parent_ids)[:pid_]
                        and chipid in noisy_chipids[:chipid_]
                    ):
                        print("Original annotation (first)...")
                        print(anns[0])
                        print("")

                new_anns = []
                for ann in anns:
                    new_ann = copy.deepcopy(ann)

                    # shift box label
                    xy_shift = self.random_shift(box_noise_coeff)
                    self.apply_box_shift(new_ann, xy_shift)
                    self.adjust_if_out_of_bounds(new_ann, img)

                    new_anns.append(new_ann)

                    if debug and new_anns:
                        if (
                            pid in list(parent_ids)[:pid_]
                            and chipid in noisy_chipids[:chipid_]
                        ):
                            if ann["id"] == anns[0]["id"]:
                                print("XY shift for anns[0]...")
                                print(xy_shift)
                                print("")

                builder.add_image(img, new_anns)

                # DEBUG
                if debug and new_anns:
                    if (
                        pid in list(parent_ids)[:pid_]
                        and chipid in noisy_chipids[:chipid_]
                    ):
                        print("New annotations...")
                        for ann in new_anns[:3]:
                            print(ann)
                        print("")

            for chipid in other_chipids:
                builder.add_image(coco.imgs[chipid], coco.imgToAnns[chipid])

        builder.save()

        print(f"total chips created: {len(builder.images)}")
        return dest_path


class COCONoisyCleanMerge:
    """Subset takes an MS COCO formatted dataset and creates a subset according to COCO parent ids provided as a lsit."""

    def __init__(
        self, noisy_dataset_path: Path, clean_dataset_path: Path, indexes_path: Path
    ) -> None:
        assert (
            noisy_dataset_path.exists()
        ), f"noisy_dataset_path: '{noisy_dataset_path}' does not exist"
        assert (
            noisy_dataset_path.is_file()
        ), f"noisy_dataset_path: '{noisy_dataset_path}' is not a file"
        self.noisy_base_path: Path = noisy_dataset_path.parent
        self.noisy_dataset_path: Path = noisy_dataset_path

        assert (
            clean_dataset_path.exists()
        ), f"clean_dataset_path: '{clean_dataset_path}' does not exist"
        assert (
            clean_dataset_path.is_file()
        ), f"clean_dataset_path: '{clean_dataset_path}' is not a file"
        self.clean_base_path: Path = clean_dataset_path.parent
        self.clean_dataset_path: Path = clean_dataset_path

        assert indexes_path.exists(), f"indexes_path: '{indexes_path}' does not exist"
        assert indexes_path.is_file(), f"indexes_path: '{indexes_path}' is not a file"
        self.indexes_base_path: Path = indexes_path.parent
        self.indexes_path: Path = indexes_path

    def load_indexes_from_json(self, index_json_path: Path):
        with open(index_json_path) as f:
            loaded_data = json.load(f)
            print(
                f"LOADING {len(loaded_data['current_indexes'])} indexes from: {index_json_path}."
            )
            current_indexes, unlabelled_indexes = (
                loaded_data["current_indexes"],
                loaded_data["unlabelled_indexes"],
            )
            return current_indexes, unlabelled_indexes

    # def get_sampled_batch_indices(split_a: float, split_b: float, al_algo) -> typing.Set[int]:
    #     """
    #     Given two splits (a, and b), returns the indices that were sampled to move
    #     from split a to split b. Second return value is the labelled set
    #     as of the start of split b.
    #     """
    #     labelled_a, unlabelled_a = utils.load_indexes(args, split_a, al_algo)
    #     labelled_b, unlabelled_b = utils.load_indexes(args, split_b, al_algo)
    #     return set(labelled_b) - set(labelled_a), set(labelled_b)

    def merge_noisy_clean(self, target_filename: str) -> None:
        """
        Create a toy sized version of dataset so we can use it just for testing if code
        runs, not for real training.

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Nothing, but the output dataset is saved to disk in the same directory
            where the input .json lives, with the same filename but with "_tiny" added
            to the filename.
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.clean_base_path / target_filename
        print(f"Creating subset of {self.clean_dataset_path} at: {dest_path}")
        coco_noisy = COCO(self.noisy_dataset_path)
        coco_clean = COCO(self.clean_dataset_path)
        builder = CocoJsonBuilder(
            coco_noisy.dataset["categories"], dest_path.parent, dest_path.name
        )

        # subset_img_ids = coco.getImgIds()[:size]

        # get the initial noisy indexes

        init_indexes, _ = self.load_indexes_from_json(self.indexes_path)

        clean_imgs = coco_clean.dataset["images"]

        for img in clean_imgs:
            anns = coco_clean.imgToAnns[img["id"]]
            if img["id"] in set(init_indexes):
                # img = coco_noisy.imgs[img["id"]]
                anns = coco_noisy.imgToAnns[img["id"]]
            builder.add_image(img, anns)
        builder.save()

        return dest_path


class CocoClassDistHelper(COCO):
    """
    A subclass of pycococtools.coco that adds a method(s) to calculate class
    distribution.
    """

    def __init__(
        self,
        annotation_file: str = None,
        create_mapping: bool = False,
        mapping_csv: str = None,
        write_to_JSON: bool = None,
    ):
        super().__init__(annotation_file, create_mapping, mapping_csv, write_to_JSON)
        # list of dictionaries. 3 keys each: (supercategory, id, name):
        self.cats = self.loadCats(self.getCatIds())
        list.sort(self.cats, key=lambda c: c["id"])
        # Dictionaries to lookup category and supercategory names from category
        # id:
        self.cat_name_lookup = {c["id"]: c["name"] for c in self.cats}
        self.supercat_name_lookup = {c["id"]: c["supercategory"] for c in self.cats}
        # List of integers, image id's:
        self.img_ids = self.getImgIds()
        # List of strings, each is an annotation id:
        self.ann_ids = self.getAnnIds(imgIds=self.img_ids)
        self.anns_list = self.loadAnns(self.ann_ids)
        print(f"num images: {len(self.img_ids)}")
        # print(F"num annotation id's: {len(self.ann_ids)}")
        print(f"num annotations: {len(self.anns)}")
        #         print(F"First annotation: {self.anns[0]}")
        #         Create self.img_ann_counts, a dictionary keyed off of img_id. For
        #         each img_id it stores a collections.Counter object, that has a count
        #         of how many annotations for each category/class there are for that
        #         img_id
        self.img_ann_counts = {}
        for img_id in self.imgToAnns.keys():
            imgAnnCounter = Counter({cat["name"]: 0 for cat in self.cats})
            anns = self.imgToAnns[img_id]
            for ann in anns:
                imgAnnCounter[self.cat_name_lookup[ann["category_id"]]] += 1
            self.img_ann_counts[img_id] = imgAnnCounter
        self.num_cats = len(self.cats)
        self.cat_img_counts: Dict[int, int] = {
            c["id"]: float(len(np.unique(self.catToImgs[c["id"]]))) for c in self.cats
        }
        self.cat_ann_counts: Dict[int, int] = defaultdict(int)
        for ann in self.anns.values():
            self.cat_ann_counts[ann["category_id"]] += 1
        self.cat_ann_counts = OrderedDict(sorted(self.cat_ann_counts.items()))

    def get_class_dist(self, img_ids: List[int] = None):
        """
        Args:
            img_ids: List of image id's. If None, distribution is calculated for
                all image id's in the dataset.

        Returns: A dictionary representing the class distribution. Keys are category
            names Values are counts (e.g., how many annotations are there with that
            category/class label) np.array of class percentages. Entries are sorted by
            category_id (same as self.cats)
        """
        cat_counter = Counter({cat["name"]: 0 for cat in self.cats})
        if img_ids is None:
            img_ids = self.imgToAnns.keys()

        for img_id in img_ids:
            if img_id not in self.imgToAnns:
                continue
            cat_counter += self.img_ann_counts[img_id]

        # Convert to np array where entries correspond to cat_id's sorted asc.:
        total = float(sum(cat_counter.values()))
        cat_names = [c["name"] for c in self.cats]
        cat_percents = np.zeros((self.num_cats))
        for idx, cat_name in enumerate(sorted(cat_names)):
            cat_percents[idx] = cat_counter[cat_name] / total

        return cat_counter, cat_percents

    def get_class_img_counts(self):
        """
        Returns dictionary whose keys are class_id's and values are number of images with one or
        more instances of that class
        """
        return self.cat_img_counts

    def get_class_ann_counts(self):
        """
        Returns dictionary whose keys are class_id's and values are number of annotations available
        for that class
        """
        return self.cat_ann_counts


def split(
    data: List, test_size: float = 0.2, random_state=None
) -> Tuple[List[Any], List[Any]]:
    """
    Similar to scikit learn, creates train/test splits of the passed in data.

    Args:
        data: A list or iterable type, of data to split.
        test_size: value in [0, 1.0] indicating the size of the test split.
        random_state: an int or RandomState object to seed the numpy randomness.

    Returns: 2-tuple of lists; (train, test), where each item in data has been placed
        into either the train or test split.
    """
    n = len(data)
    num_test = int(np.ceil(test_size * n))
    #     print(F"n:{n}, num_test:{num_test}, num_train:{num_train}")
    np.random.seed(random_state)
    test_idx = set(np.random.choice(range(n), num_test))
    data_test, data_train = list(), list()
    for idx, datum in enumerate(data):
        if idx in test_idx:
            data_test.append(data[idx])
        else:
            data_train.append(data[idx])
    return data_train, data_test


@dataclass
class bbox:
    """
    Data class to store a bounding box annotation instance
    """

    img_id: int
    cat_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class Img:
    """A helper class to store image info and annotations."""

    anns: List[bbox]

    def __init__(self, id: int, filename: str, width: float, height: float) -> None:
        self.id: int = id
        self.filename: str = filename
        self.width: float = width
        self.height: float = height
        self.anns = []

    def add_ann(self, ann: bbox) -> None:
        """Add an annotation to the image"""
        self.anns.append(ann)

    def get_anns(self) -> List[bbox]:
        """
        Gets annotations, possibly filters them in prep for converting to yolo/Darknet
        format.
        """
        return self.anns

    def to_darknet(self, box: bbox) -> bbox:
        """Convert a BBox from coco to Darknet format"""
        # COCO bboxes define the topleft corner of the box, but yolo expects the x/y
        # coords to reference the center of the box. yolo also requires the coordinates
        # and widths to be scaled by image dims, down to the range [0.0, 1.0]
        return bbox(
            self.id,
            box.cat_id,
            (box.x_center + (box.width / 2.0)) / self.width,
            (box.y_center + (box.height / 2.0)) / self.height,
            box.width / self.width,
            box.height / self.height,
        )

    def write_darknet_anns(self, label_file) -> None:
        """Writes bounding boxes to specified file in yolo/Darknet format"""
        # It's a bit leaky abstraction to have Img handle writing to file but it's
        # convenient b/c we have access to img height and width here to scale the bbox
        # dims. Same goes for .to_darknet()
        anns = self.get_anns()
        for box in anns:
            box = self.to_darknet(box)
            label_file.write(
                f"{box.cat_id} {box.x_center} {box.y_center} {box.width} {box.height}\n"
            )

    def has_anns(self) -> bool:
        """
        Returns true if this image instance has at least one bounding box (after any
        filters are applied)
        """
        # TODO: Can add filter to only return true if annotations have non-zero area: I
        # saw around ~5 or 6 annotations in the v2_train_chipped.json that had zero
        # area, not sure if those might cause problems for yolo
        return self.anns

    def get_label_path(self, base_path: Path) -> str:
        return base_path / self.filename.replace("jpeg", "txt").replace("jpg", "txt")

    def get_img_path(self, base_path: Path, dataset_name: str, data_split: str) -> str:
        return (
            base_path
            / dataset_name.replace("_tiny", "")
            / "images"
            / data_split
            / self.filename
        )


class CocoToDarknet:
    """Class that helps convert an MS COCO formatted dataset to yolo/Darknet format"""

    @staticmethod
    def convert(
        ann_path: Path, base_path: Path, dataset_name: str, data_split: str
    ) -> None:
        """Convert specified dataset to Darknet format.

        Details:
            - Labels are written to base_path/<dataset_name>/labels/<data_split>/*.txt
            - A file containing list of category names, is written to
                <base_path>/<dataset_name>.names
        """
        coco = COCO(ann_path)
        images = CocoToDarknet.build_db(coco)
        # Make paths:
        labels_path = base_path / dataset_name / "labels" / data_split
        labels_path.mkdir(parents=True, exist_ok=True)
        names_path = base_path / f"{dataset_name}.names"
        image_paths = CocoToDarknet.generate_label_files(
            images, labels_path, base_path, dataset_name, data_split
        )
        CocoToDarknet.generate_image_list(
            base_path, dataset_name, image_paths, data_split
        )
        CocoToDarknet.generate_names(names_path, coco)

    @staticmethod
    def generate_names(names_path: Path, coco: COCO) -> None:
        categories = [c["name"] + "\n" for c in coco.dataset["categories"]]
        with open(names_path, "w") as names_file:
            names_file.writelines(categories)

    @staticmethod
    def generate_label_files(
        images: Dict[int, Img],
        labels_path: Path,
        base_path: Path,
        dataset_name: str,
        data_split: str,
    ) -> List[str]:
        """
        Generates one .txt file for each image in the coco-formatted dataset. The .txt
        files contain the annotations in yolo/Darknet format.
        """
        # Convert:
        img_paths = set()
        for img_id, img in images.items():
            if img.has_anns():
                label_path = labels_path / img.get_label_path(labels_path)
                with open(label_path, "w") as label_file:
                    img.write_darknet_anns(label_file)
                img_path = img.get_img_path(base_path, dataset_name, data_split)
                assert img_path.exists(), f"Image doesn't exist {img_path}"
                img_paths.add(str(img_path) + "\n")
        return list(img_paths)

    @staticmethod
    def generate_image_list(
        base_path: Path, dataset_name: str, image_paths: List[str], data_split: str
    ) -> None:
        """Generates train.txt, val.txt, etc, txt file with list of image paths."""
        listing_path = base_path / dataset_name / f"{data_split}.txt"
        print("Listing path: ", listing_path)
        with open(listing_path, "w") as listing_file:
            listing_file.writelines(image_paths)

    @staticmethod
    def build_db(coco: COCO) -> Dict[int, Img]:
        """
        Builds a datastructure of images. All annotations are grouped into their
        corresponding images to facilitate generating the Darknet formatted metadata.

        Args:
            coco: a pycocotools.coco COCO instance

        Returns: Dictionary whose keys are image id's, and values are Img instances that
            are loaded with all the image info and annotations from the coco-formatted
            json
        """
        anns = coco.dataset["annotations"]
        images: Dict[int, Img] = {}
        # Build images data structure:
        for i, ann in enumerate(anns):
            ann = CocoToDarknet.get_ann(ann)
            if ann.img_id not in images:
                coco_img = coco.dataset["images"][ann.img_id]
                images[ann.img_id] = Img(
                    ann.img_id,
                    coco_img["file_name"],
                    float(coco_img["width"]),
                    float(coco_img["height"]),
                )
            img = images[ann.img_id]
            img.add_ann(ann)
        return images

    @staticmethod
    def get_ann(ann):
        """
        Gets a bbox instance from an annotation element pulled from the coco-formatted
        json
        """
        box = ann["bbox"]
        return bbox(ann["image_id"], ann["category_id"], box[0], box[1], box[2], box[3])
