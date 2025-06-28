from .data_handler import DataHandler
import json
import os
from ..data.annotation_bundle import AnnotationBundle
from ..containers.explicit_image_container import ExplicitImageContainer
from ..data.annotation import Annotation
from ..data.box import Box
from typing import List
import numpy as np

class TrafficLightDetectionDatasetHandler(DataHandler):
    def __init__(self):
        super().__init__()
        self.min_side_size = -1
    
    def load(self, path: str) -> list:
        super().load(path)
        annotation_bundels: List[AnnotationBundle] = []

        json_path = os.path.join(path, "train.json")
        
        if not os.path.exists(json_path):
            raise Exception("Incorrect dataset structure: train.json file is missing!")

        with open(json_path, "r") as file:
            root = json.loads(file.read())
        
        image_container = None
        annotations: List[Annotation] = []
        root["annotations"].append({"filename":"", "stopkey":"stopkey"}) # Для корректного завершения цикла
        for annotation in root["annotations"]:
                image_filename = os.path.join(path, annotation["filename"])
                
                if image_container is not None and image_container.image_path != image_filename:
                    if len(annotations) > 0:
                        annotation_bundle = AnnotationBundle(annotations, image_container)
                        annotation_bundels.append(annotation_bundle)
                    if "stopkey" in annotation:
                        break
                
                if image_container is None or image_container.image_path != image_filename:            
                    image_container = ExplicitImageContainer(image_filename)
                    annotations: List[Annotation] = []
                
                if len(annotation["inbox"]) > 0:
                    points = np.array([[annotation["bndbox"]["xmin"], annotation["bndbox"]["ymin"]],
                                       [annotation["bndbox"]["xmax"], annotation["bndbox"]["ymax"]]])
                    points_n = points / image_container.get_image_shape()

                    width, height = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
                    if self.min_side_size < 0 or (width >= self.min_side_size or height >= self.min_side_size):
                        label = {"red":"redtrafficlight", "yellow":"yellowtrafficlight", "green":"greentrafficlight"}[annotation["inbox"][0]["color"]]
                        box = Box(points, points_n, label, image_container, False)
                        annotations.append(box)
        
        return annotation_bundels, ["redtrafficlight", "yellowtrafficlight", "greentrafficlight"]
                
        
    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)