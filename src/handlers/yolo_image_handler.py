from .data_handler import DataHandler
import os
import numpy as np
from ..data.mask import Mask
from ..containers.explicit_image_container import ExplicitImageContainer
from typing import List
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..data.box import Box
import yaml


class YoloImageHandler(DataHandler):
    def __init__(self, is_segmentation=False):
        super().__init__()
        self.is_segmentation = is_segmentation
    
    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)
        annotation_bundels: List[AnnotationBundle] = []
        
        path = os.path.normpath(path)
        
        config_path = os.path.join(path, "config.yaml")
        train_image_dir = os.path.join(path, "images", "train")
        valid_image_dir = os.path.join(path, "images", "valid")
        train_label_dir = os.path.join(path, "labels", "train")
        valid_label_dir = os.path.join(path, "labels", "valid")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        label_names = config["names"]

        count_train_images = len(os.listdir(train_image_dir))
        count_valid_images = len(os.listdir(valid_image_dir))
        count_images = count_train_images + count_valid_images
        self.logger.set_max_count(count_images)
        
        counter = 0
        for idx, (images_path, labels_path) in enumerate([(train_image_dir, train_label_dir), (valid_image_dir, valid_label_dir)]):
            data_filenames = os.listdir(images_path)

            for filename in data_filenames:
                image_path = os.path.join(images_path, filename)
                label_path = os.path.join(labels_path, os.path.splitext(filename)[0] + ".txt")

                image_container = ExplicitImageContainer(image_path)
                annotations: List[Annotation] = []

                with open(label_path, "r") as file:
                    label_lines = file.readlines()
                
                for line in label_lines:
                    values = line.strip('\n').split(' ')
                    # matches = re.findall(r'-?\d+\.\d+|-?\d+', line)
                    # values = np.array([float(match) for match in matches])
                    
                    
                    label_name = label_names[int(values[0])]
                    
                    # points_n = values[1:].reshape(-1, 2)
                    points_n = np.array([float(values[idx]) for idx in range(1, len(values))]).reshape((-1, 2))

                    if not self.is_segmentation:
                        points_n = np.array([points_n[0] - points_n[1] / 2, points_n[0] + points_n[1] / 2])
                    
                    # for point in points_n:
                    #     if point[0] < 0 or point[1] < 0:
                    #         print("AAAAAAAAAAAAAAAAAAAAAAAAA") # YOU HAVE TO DELETE THIS ROW! IT WAS MADE SOLELY FOR DEBUGING!!!
                    
                    points = points_n * image_container.get_image_shape()

                    args = dict(points=points, points_n=points_n, label=label_name, image_container=image_container, is_valid=idx==1)
                    annotation = Mask(**args) if self.is_segmentation else Box(**args)
                    
                    annotations.append(annotation)
                
                annotation_bundel = AnnotationBundle(annotations, image_container)
                annotation_bundels.append(annotation_bundel)
                
                self.logger.print_counter(counter)
                counter += 1
        
        return annotation_bundels, label_names
            
        
        
    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)
        label2id = {label_name: idx for idx, label_name in enumerate(label_names)}

        os.makedirs(path, exist_ok=True)
        
        config_path = os.path.join(path, "config.yaml")
        train_image_dir = os.path.join(path, "images", "train")
        valid_image_dir = os.path.join(path, "images", "valid")
        train_label_dir = os.path.join(path, "labels", "train")
        valid_label_dir = os.path.join(path, "labels", "valid")
        
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(valid_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(valid_label_dir, exist_ok=True)

        config = {}
        config["path"] = os.path.join(os.getcwd(), path)
        config["train"] = r"images\train"
        config["val"] = r"images\valid"
        config["nc"] = len(label_names)
        config["names"] = label_names

        with open(config_path, "w", encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
        count_images = len(annotation_bundels)
        self.logger.set_max_count(count_images)

        num_train = int(len(annotation_bundels) * (1.0 - validation_split))
        for bundle_id, annotation_bundel in enumerate(annotation_bundels):
            output_str = ""
            for annotation in annotation_bundel.annotations: 
                output_str += str(label2id[annotation.label])
                if isinstance(annotation, Box):
                    points_n = annotation.points_n
                    width, height = [points_n[1][0] - points_n[0][0], points_n[1][1] - points_n[0][1]]
                    x, y = points_n[0] + np.array([width, height]) / 2
                    output_str += f" {x:.10f} {y:.10f} {width:.10f} {height:.10f}"
                else:
                    for point in annotation.points_n:
                        output_str += f" {point[0]:.10f} {point[1]:.10f}"

                output_str += "\n"
                
            image_name = annotation_bundel.image_container.image_name
                
            if bundle_id < num_train:
                label_path = os.path.join(train_label_dir, image_name + ".txt")
            else:
                label_path = os.path.join(valid_label_dir, image_name + ".txt")
                
            with open(label_path, 'w') as file:
                file.write(output_str)
            
            if bundle_id < num_train:
                annotation_bundel.image_container.save_image(train_image_dir)
            else:
                annotation_bundel.image_container.save_image(valid_image_dir)
            
            self.logger.print_counter(bundle_id)
            