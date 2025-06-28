from .data_handler import DataHandler
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from ..data.mask import Mask
from ..data.box import Box
from ..containers.video_image_container import VideoImageContainer
from ..data.annotation_bundle import AnnotationBundle
from typing import List


class CvatVideoHandler(DataHandler):
    def __init__(self):
        super().__init__()
    
    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)
        annotation_bundels: List[AnnotationBundle] = []
        
        annotation_file_path = os.path.join(path, "annotations.xml")
        
        video_path = ""
        dataset_file_names = os.listdir(path)
        for file_name in dataset_file_names:
            file_name_base = os.path.splitext(file_name)[0]
            if file_name_base == "video":
                video_path = os.path.join(path, file_name)
                break
        
        if video_path == "":
            raise Exception("Incorrect dataset structure: video file is missing!")
        
        if not os.path.exists(annotation_file_path):
            raise Exception("Incorrect dataset structure: annotations.xml file is missing!")

        tree = ET.parse(annotation_file_path)
        root = tree.getroot()
        
        label_elements = root.find('.//labels').findall(".//label")
        label_names = [label_element.find('.//name').text for label_element in label_elements]
        # label2id = {label_name: idx  for idx, label_name in enumerate(label_names)}
    
        width = int(root.find('.//original_size').find('.//width').text)
        height = int(root.find('.//original_size').find('.//height').text)
        image_shape = np.array([width, height])

        frames_count = int(root.find('.//job').find('.//size').text)

        annotation_batches = [[] for frame_id in range(frames_count)]
        image_containers = [None for frame_id in range(frames_count)]

        track_elements = root.findall('.//track')
        for track_element in track_elements:
            label = track_element.attrib['label']
            
            polygon_elements = track_element.findall('.//polygon')
            for polygon_element in polygon_elements:
                points_str = polygon_element.attrib['points']
                frame_number = int(polygon_element.attrib['frame'])

                matches = re.findall(r'-?\d+\.\d+|-?\d+', points_str)
                points = np.array([float(match) for match in matches])
                points = points.reshape((-1, 2))
                
                points_n = points / image_shape

                if image_containers[frame_number] is None:
                    image_containers[frame_number] = VideoImageContainer(video_path, frame_number)
                
                mask = Mask(points, points_n, label, image_containers[frame_number], False) 
                annotation_batches[frame_number].append(mask)
            
            box_elements = track_element.findall('.//box')
            for box_element in box_elements:
                frame_number = int(box_element.attrib['frame'])

                points = [[float(box_element.attrib['xtl']), float(box_element.attrib['ytl'])],
                          [float(box_element.attrib['xbr']), float(box_element.attrib['ybr'])]]
                
                points_n = points.copy() / image_shape
                
                if image_containers[frame_number] is None:
                    image_containers[frame_number] = VideoImageContainer(video_path, frame_number)
                
                box = Box(points, points_n, label, image_containers[frame_number], False) 
                annotation_batches[frame_number].append(box)
    
        # It's better to fix this thing below (because it's kinda lame and I did it just to save some time for now)
        filtered_image_containers =  []
        filtered_annotation_batches = []
        for idx, image_container in enumerate(image_containers):
            if image_container is not None:
                filtered_image_containers.append(image_container)
                filtered_annotation_batches.append(annotation_batches[idx])
        ###################################################################
    
        annotation_bundels: List[AnnotationBundle] = [AnnotationBundle(filtered_annotation_batches[idx],
                                                                       filtered_image_containers[idx]) for idx in range(len(filtered_image_containers))]
        
                


            #     if annotation_bundels[frame_number] is None:
            #         annotations: List[Annotation] = []
            #         annotations.append(mask)

            #         image_container = VideoImageContainer(video_path, frame_number)

            #         annotation_bundels[frame_number] = AnnotationBundle(annotations, image_container)
            #     else:
            #         annotation_bundels[frame_number].annotations # You stopped here. (create seperate arrays - for annotations and for containers)
                
            # annotation_bundle = AnnotationBundle(annotations, image_container)
            # annotation_bundels.append(annotation_bundle)
        return annotation_bundels, label_names

    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)