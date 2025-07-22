from .data_handler import DataHandler
from pathlib import Path
import yaml
import numpy as np
from ..data.box import Box
from ..data.annotation_bundle import AnnotationBundle
from typing import List
from src.containers.explicit_image_container import ExplicitImageContainer
import cv2


class BoschHandler(DataHandler):
    def __init__(self):
        super().__init__()
    
    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)
        annotation_bundles: List[AnnotationBundle] = []
        label_names = set()

        dataset_path = Path(path)
        yaml_path = dataset_path / "train.yaml"

        with open(yaml_path, "r") as f:
            entries = yaml.safe_load(f)

        for entry in entries:
            boxes = entry.get("boxes", [])
            if not boxes:
                continue
            image_rel_path = entry["path"]
            image_path = dataset_path / Path(image_rel_path)
            image_container = ExplicitImageContainer(str(image_path))

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_shape = np.array([image.shape[1], image.shape[0]])  # [width, height]

            annotations = []
            for row in boxes:
                label = row["label"]

                if label == "off":
                    continue

                if "Green" in label:
                    mapped_label = "go"
                elif "Red" in label:
                    mapped_label = "stop"
                elif "Yellow" in label:
                    mapped_label = "warning"
                else:
                    continue

                x1 = float(row["x_min"])
                y1 = float(row["y_min"])
                x2 = float(row["x_max"])
                y2 = float(row["y_max"])

                points = [[x1, y1], [x2, y2]]
                points_n = np.array(points) / image_shape

                label_names.add(mapped_label)

                box = Box(points, points_n.tolist(), mapped_label, image_container, False)
                annotations.append(box)


            annotation_bundle = AnnotationBundle(annotations, image_container)
            annotation_bundles.append(annotation_bundle)
        return annotation_bundles, list(label_names)

    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)