from .data_handler import DataHandler
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from ..data.mask import Mask
from ..containers.explicit_image_container import ExplicitImageContainer
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..data.box import Box
from typing import List
import xml.dom.minidom
from datetime import datetime, timezone


class CvatImageHandler(DataHandler):
    def __init__(self):
        super().__init__()
    
    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)
        annotation_bundels: List[AnnotationBundle] = []
        
        annotation_file_path = os.path.join(path, "annotations.xml")
        images_path = os.path.join(path, "images")
        
        if not os.path.exists(annotation_file_path):
            raise Exception("Incorrect dataset structure: annotations.xml file is missing!")
        
        if not os.path.exists(images_path):
            raise Exception("Incorrect dataset structure: images folder is missing!")
        
        tree = ET.parse(annotation_file_path)
        root = tree.getroot()
        
        label_elements = root.find('.//labels').findall(".//label")
        label_names = [label_element.find('.//name').text for label_element in label_elements]
        label2id = {label_name: idx  for idx, label_name in enumerate(label_names)}
    
        image_elements = root.findall('.//image')

        for image_element in image_elements:
            width = int(image_element.attrib['width'])
            height = int(image_element.attrib['height'])
            
            image_shape = np.array([width, height])
            
            image_path = os.path.join(path, "images", image_element.attrib['name'])
            
            image_container = ExplicitImageContainer(image_path)

            if image_container.get_image() is None:
                raise Exception("Cannot load image")

            annotations: List[Annotation] = []

            polygon_elements = image_element.findall('.//polygon')
            for polygon_element in polygon_elements:
                points_str = polygon_element.attrib['points']
                
                label = polygon_element.attrib['label']
                
                matches = re.findall(r'-?\d+\.\d+|-?\d+', points_str)
                points = np.array([float(match) for match in matches])
                points = points.reshape((-1, 2))
                
                points_n = points / image_shape
                
                mask = Mask(points, points_n, label, image_container, False)
                annotations.append(mask)
            
            box_elements = image_element.findall('.//box')
            for box_element in box_elements:      
                label = box_element.attrib['label']
                points = [[float(box_element.attrib['xtl']), float(box_element.attrib['ytl'])],
                          [float(box_element.attrib['xbr']), float(box_element.attrib['ybr'])]]
                
                points_n = points.copy() / image_shape
                
                box = Box(points, points_n, label, image_container, False)
                annotations.append(box)

            annotation_bundle = AnnotationBundle(annotations, image_container)
            annotation_bundels.append(annotation_bundle)
        return annotation_bundels, label_names
    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)

        root = ET.Element("annotations")
        
        ET.SubElement(root, "version").text = str(1.1)

        # Метаданные
        meta = ET.SubElement(root, "meta")
        job = ET.SubElement(meta, "job")

        job_id = 1035178 + np.random.randint(0, 5000)
        ET.SubElement(job, "id").text = str(job_id)
        ET.SubElement(job, "size").text = str(len(annotation_bundels))
        ET.SubElement(job, "mode").text = "annotation"
        ET.SubElement(job, "overlap").text = str(0)
        ET.SubElement(job, "bugtracker")

        now_utc = datetime.now(timezone.utc)
        formatted_now = now_utc.strftime("%Y-%m-%d %H:%M:%S.%f%z")
        formatted_now = formatted_now[:-2] + ":" + formatted_now[-2:]

        ET.SubElement(job, "created").text = formatted_now
        ET.SubElement(job, "updated").text = formatted_now
        ET.SubElement(job, "subset").text = "default"
        ET.SubElement(job, "start_frame").text = str(0)
        ET.SubElement(job, "stop_frame").text = str(len(annotation_bundels) - 1)
        ET.SubElement(job, "frame_filter")

        segments = ET.SubElement(job, "segments")
        segment = ET.SubElement(segments, "segment")

        ET.SubElement(segment, "id").text = str(job_id)
        ET.SubElement(segment, "start").text = str(0)
        ET.SubElement(segment, "stop").text = str(len(annotation_bundels) - 1)
        ET.SubElement(segment, "url").text = f"https://app.cvat.ai/api/jobs/{job_id}"

        owner = ET.SubElement(job, "owner")
        ET.SubElement(owner, "username").text = "lit92"
        ET.SubElement(owner, "email").text = "emaillit8@gmail.com"

        ET.SubElement(job, "assignee")

        labels = ET.SubElement(job, "labels")
        for label_name in label_names:
            label_xml = ET.SubElement(labels, "label")

            ET.SubElement(label_xml, "name").text = label_name

            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            b = np.random.randint(0, 256)
            ET.SubElement(label_xml, "color").text = '#{:02x}{:02x}{:02x}'.format(r, g, b)

            ET.SubElement(label_xml, "type").text = "any"
            ET.SubElement(label_xml, "attributes")
        
        ET.SubElement(meta, "dumped").text = formatted_now

        for idx, bundle in enumerate(annotation_bundels):
            image_xml = ET.SubElement(root, "image")
            
            image_xml.set("id", str(idx))
            image_xml.set("name", bundle.image_container.image_name + ".jpg")

            image_shape = bundle.image_container.get_image_shape()
            image_xml.set("width", str(image_shape[0]))
            image_xml.set("height", str(image_shape[1]))

            for annotation in bundle.annotations:
                if isinstance(annotation, Box):
                    box = ET.SubElement(image_xml, "box")
                    box.set("label", annotation.label)
                    box.set("source", "manual")
                    box.set("occluded", str(0))

                    box.set("xtl", str(annotation.points[0][0]))
                    box.set("ytl", str(annotation.points[0][1]))

                    box.set("xbr", str(annotation.points[1][0]))
                    box.set("ybr", str(annotation.points[1][1]))

                    box.set("z_order", str(0))

        tree = ET.ElementTree(root)

        xml_str = ET.tostring(root, encoding='utf-8', method='xml')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml_as_string = dom.toprettyxml()

        os.makedirs(path, exist_ok=True)

        annotations_path = os.path.join(path, "annotations.xml")

        with open(annotations_path, "w", encoding='utf-8') as file:
            file.write(pretty_xml_as_string)
        
        images_path = os.path.join(path, "images")

        os.makedirs(images_path, exist_ok=True)

        for bundle in annotation_bundels:
            bundle.image_container.save_image(images_path)
        
