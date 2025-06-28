from .data_handler import DataHandler
import os
import numpy as np
from ..data.box import Box
from ..data.annotation_bundle import AnnotationBundle
from typing import List
import json
from src.containers.explicit_image_container import ExplicitImageContainer


class RTSDHandler(DataHandler):
    def __init__(self):
        super().__init__()
    
    def _handle_json(self, obj: dict, dataset_path: str, id_to_name: dict[int, str], is_val: bool) -> List[AnnotationBundle]:
        image_id_to_image_dicts = {image_dict["id"]: image_dict for image_dict in obj["images"]}

        image_id_to_bundles: dict[int, AnnotationBundle] = []
        for annotation_dict in obj["annotations"]:
            image_id = annotation_dict["image_id"]

            if image_id in image_id_to_bundles:
                bundle = image_id_to_bundles[image_id]
            else:
                image_path = os.path.join(dataset_path, "rtsd-frames", image_id_to_image_dicts[image_id]["file_name"])
                image_container = ExplicitImageContainer(image_path)

                bundle = AnnotationBundle([], image_container)
                image_id_to_bundles[image_id] = bundle
            
            label = id_to_name[annotation_dict["category_id"]]

            points = np.array(annotation_dict["bbox"]).reshape(-1, 2)
            points_n = points / np.ndarray(bundle.image_container.get_image_shape())

            annotation = Box(points, points_n, label, bundle.image_container, is_val, bundle)
            bundle.annotations.append(annotation)
        
        bundles = list(image_id_to_bundles.values())
        for bundle in bundles:
            bundle.bind_annotations()
        
        return bundles

    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)

        label_map_path = os.path.join(path, "label_map.json")

        rtsd_train_json_path = os.path.join(path, "train_anno.json")
        rtsd_val_json_path = os.path.join(path, "val_anno.json")

        with open(rtsd_train_json_path, "r", encoding="utf-8") as file:
            train_json = json.load(file)
        
        with open(rtsd_val_json_path, "r", encoding="utf-8") as file:
            val_json = json.load(file)

        with open(label_map_path) as f:
            name_to_id = json.load(f)

        label_names = list(name_to_id.keys())
        id_to_name = {v: k for k, v in name_to_id.items()}

        train_bundles = self._handle_json(train_json, path, id_to_name, is_val=False)
        val_bundles = self._handle_json(val_json, path, id_to_name, is_val=True)

        annotation_bundles = train_bundles + val_bundles

        return annotation_bundles, label_names

    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)