from .data_handler import DataHandler
import os
import xml.etree.ElementTree as ET
from ..containers.explicit_image_container import ExplicitImageContainer
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..data.box import Box
from typing import List, Set
import numpy as np


class S2TLDHandler(DataHandler):
    def __init__(self):
        super().__init__()
    
    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)
        annotation_bundles: List[AnnotationBundle] = []
        label_set: Set[str] = set()

        for variant in sorted(os.listdir(path)):
            variant_path = os.path.join(path, variant)
            if not os.path.isdir(variant_path):
                continue

            # Функция-утилита для обработки одной папки с Annotations/JPEGImages
            def process_folder(folder_path: str):
                ann_dir = os.path.join(folder_path, "Annotations")
                img_dir = os.path.join(folder_path, "JPEGImages")
                if not (os.path.isdir(ann_dir) and os.path.isdir(img_dir)):
                    return

                for fname in sorted(os.listdir(ann_dir)):
                    if not fname.lower().endswith(".xml"):
                        continue
                    xml_path = os.path.join(ann_dir, fname)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # Размер изображения
                    size = root.find("size")
                    width = int(size.find("width").text)
                    height = int(size.find("height").text)
                    image_shape = np.array([width, height])

                    # Имя и путь к изображению
                    img_filename = root.find("filename").text
                    image_path = os.path.join(img_dir, img_filename)
                    image_container = ExplicitImageContainer(image_path)

                    annotations: List[Annotation] = []
                    for obj in root.findall("object"):
                        label_raw = obj.find("name").text
                        # Преобразование меток
                        if label_raw == "red":
                            label = "stop"
                        elif label_raw == "yellow":
                            label = "warning"
                        elif label_raw == "green":
                            label = "go"
                        else:
                            continue

                        bnd = obj.find("bndbox")
                        x1 = float(bnd.find("xmin").text)
                        y1 = float(bnd.find("ymin").text)
                        x2 = float(bnd.find("xmax").text)
                        y2 = float(bnd.find("ymax").text)

                        points = [[x1, y1], [x2, y2]]
                        points_n = (np.array(points) / image_shape).tolist()

                        box = Box(points, points_n, label, image_container, False)
                        annotations.append(box)
                        label_set.add(label)

                    bundle = AnnotationBundle(annotations, image_container)
                    annotation_bundles.append(bundle)

            # Если в variant_path сразу есть папки Annotations и JPEGImages
            if os.path.isdir(os.path.join(variant_path, "Annotations")) \
               and os.path.isdir(os.path.join(variant_path, "JPEGImages")):
                process_folder(variant_path)
            else:
                # Иначе ищем поддиректории normal_1, normal_2 и т.п.
                for split in sorted(os.listdir(variant_path)):
                    split_path = os.path.join(variant_path, split)
                    if not os.path.isdir(split_path):
                        continue
                    process_folder(split_path)

        label_names = sorted(label_set)
        return annotation_bundles, label_names
    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)
        return None
