from .data_handler import DataHandler
import os
import xml.etree.ElementTree as ET
import re
from ..data.mask import Mask
from ..containers.explicit_image_container import ExplicitImageContainer
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..data.box import Box
from typing import List
import xml.dom.minidom
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd


class LisaHandler(DataHandler):
    def __init__(self):
        super().__init__()
    
    def load(self, path: str) -> List[AnnotationBundle]:
        super().load(path)
        annotation_bundels: List[AnnotationBundle] = []

        anotations_dir = os.path.join(path, "Annotations", "Annotations")
        weather_condition_folder_names = os.listdir(anotations_dir)

        annotation_csv_paths = []
        for folder_name in weather_condition_folder_names:
            folder_path = os.path.join(anotations_dir, folder_name)
            if os.path.isdir(folder_path):
                csv_file_path = os.path.join(folder_path, "frameAnnotationsBOX.csv")
                if os.path.exists(csv_file_path):
                    annotation_csv_paths.append(csv_file_path)
                else:
                    clip_names = os.listdir(folder_path)

                    for clip_name in clip_names:
                        
                        clip_path = os.path.join(folder_path, clip_name)
                        if os.path.isdir(clip_path):
                            csv_file_path = os.path.join(clip_path, "frameAnnotationsBOX.csv")
                            if os.path.exists(csv_file_path):
                                annotation_csv_paths.append(csv_file_path)
        
        annotation_dataframes = []
        for annotation_csv_path in annotation_csv_paths:
            df = pd.read_csv(annotation_csv_path, sep=";")
            df["folder_name"] = os.path.dirname(os.path.relpath(annotation_csv_path, anotations_dir))

            folder_name = os.path.dirname(os.path.relpath(annotation_csv_path, anotations_dir))
            df["folder_name"] = folder_name

            category_name = Path(folder_name).parts[0]

            df["Filename"] = df["Filename"].apply(os.path.basename)

            df["image_path"] = os.path.join(path, category_name, folder_name, "frames") + os.sep + df["Filename"]

            annotation_dataframes.append(df)

        annotation_dataframe = annotation_dataframes[0]
        for df in annotation_dataframes[1:]:
            annotation_dataframe = pd.concat([annotation_dataframe, df], ignore_index=True)

        label_names = annotation_dataframe["Annotation tag"].unique().tolist()

        image_path = annotation_dataframe.iloc[0]["image_path"]
        width, height = Image.open(image_path).size
        image_shape = np.array([width, height])

        annotation_bundels: List[AnnotationBundle] = []

        grouped = annotation_dataframe.groupby("image_path")
        
        for image_path_id, (image_path, annotation_rows) in enumerate(grouped):
            print(f"{image_path_id} / {len(grouped)}")

            if image_path_id == 5000:
                break

            image_container = ExplicitImageContainer(image_path)

            annotations: List[Annotation] = []

            for _, row in annotation_rows.iterrows():

                label = row["Annotation tag"]

                x1 = float(row["Upper left corner X"])
                y1 = float(row["Upper left corner Y"])
                x2 = float(row["Lower right corner X"])
                y2 = float(row["Lower right corner Y"])

                points = [[x1, y1], [x2, y2]]

                points_n = points.copy() / image_shape

                box = Box(points, points_n, label, image_container, False)

                annotations.append(box)
            
            # for row in annotation_rows.itertuples(index=False):
            #     row = row._asdict()

            #     label = row["_1"] # Annotation tag

            #     # x1 = float(row["Upper left corner X"])
            #     # y1 = float(row["Upper left corner Y"])
            #     # x2 = float(row["Lower right corner X"])
            #     # y2 = float(row["Lower right corner Y"])

            #     x1 = float(row["_2"]) # Upper left corner X
            #     y1 = float(row["_3"]) # Upper left corner Y
            #     x2 = float(row["_4"]) # Lower right corner X
            #     y2 = float(row["_5"]) # Lower right corner Y

            #     points = [[x1, y1], [x2, y2]]

            #     points_n = points.copy() / image_shape

            #     box = Box(points, points_n, label, image_container, False)

            #     annotations.append(box)
            
            annotation_bundle = AnnotationBundle(annotations, image_container)
            annotation_bundels.append(annotation_bundle)
        
        return annotation_bundels, label_names
    
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: List[str], path: str, validation_split: int):
        super().save(annotation_bundels, label_names, path, validation_split)
        return None

        # root = ET.Element("annotations")
        
        # ET.SubElement(root, "version").text = str(1.1)

        # # Метаданные
        # meta = ET.SubElement(root, "meta")
        # job = ET.SubElement(meta, "job")

        # job_id = 1035178 + np.random.randint(0, 5000)
        # ET.SubElement(job, "id").text = str(job_id)
        # ET.SubElement(job, "size").text = str(len(annotation_bundels))
        # ET.SubElement(job, "mode").text = "annotation"
        # ET.SubElement(job, "overlap").text = str(0)
        # ET.SubElement(job, "bugtracker")

        # now_utc = datetime.now(timezone.utc)
        # formatted_now = now_utc.strftime("%Y-%m-%d %H:%M:%S.%f%z")
        # formatted_now = formatted_now[:-2] + ":" + formatted_now[-2:]

        # ET.SubElement(job, "created").text = formatted_now
        # ET.SubElement(job, "updated").text = formatted_now
        # ET.SubElement(job, "subset").text = "default"
        # ET.SubElement(job, "start_frame").text = str(0)
        # ET.SubElement(job, "stop_frame").text = str(len(annotation_bundels) - 1)
        # ET.SubElement(job, "frame_filter")

        # segments = ET.SubElement(job, "segments")
        # segment = ET.SubElement(segments, "segment")

        # ET.SubElement(segment, "id").text = str(job_id)
        # ET.SubElement(segment, "start").text = str(0)
        # ET.SubElement(segment, "stop").text = str(len(annotation_bundels) - 1)
        # ET.SubElement(segment, "url").text = f"https://app.cvat.ai/api/jobs/{job_id}"

        # owner = ET.SubElement(job, "owner")
        # ET.SubElement(owner, "username").text = "lit92"
        # ET.SubElement(owner, "email").text = "emaillit8@gmail.com"

        # ET.SubElement(job, "assignee")

        # labels = ET.SubElement(job, "labels")
        # for label_name in label_names:
        #     label_xml = ET.SubElement(labels, "label")

        #     ET.SubElement(label_xml, "name").text = label_name

        #     r = np.random.randint(0, 256)
        #     g = np.random.randint(0, 256)
        #     b = np.random.randint(0, 256)
        #     ET.SubElement(label_xml, "color").text = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        #     ET.SubElement(label_xml, "type").text = "any"
        #     ET.SubElement(label_xml, "attributes")
        
        # ET.SubElement(meta, "dumped").text = formatted_now

        # for idx, bundle in enumerate(annotation_bundels):
        #     image_xml = ET.SubElement(root, "image")
            
        #     image_xml.set("id", str(idx))
        #     image_xml.set("name", bundle.image_container.image_name + ".jpg")

        #     image_shape = bundle.image_container.get_image_shape()
        #     image_xml.set("width", str(image_shape[0]))
        #     image_xml.set("height", str(image_shape[1]))

        #     for annotation in bundle.annotations:
        #         if isinstance(annotation, Box):
        #             box = ET.SubElement(image_xml, "box")
        #             box.set("label", annotation.label)
        #             box.set("source", "manual")
        #             box.set("occluded", str(0))

        #             box.set("xtl", str(annotation.points[0][0]))
        #             box.set("ytl", str(annotation.points[0][1]))

        #             box.set("xbr", str(annotation.points[1][0]))
        #             box.set("ybr", str(annotation.points[1][1]))

        #             box.set("z_order", str(0))

        # tree = ET.ElementTree(root)

        # xml_str = ET.tostring(root, encoding='utf-8', method='xml')
        # dom = xml.dom.minidom.parseString(xml_str)
        # pretty_xml_as_string = dom.toprettyxml()

        # os.makedirs(path, exist_ok=True)

        # annotations_path = os.path.join(path, "annotations.xml")

        # with open(annotations_path, "w", encoding='utf-8') as file:
        #     file.write(pretty_xml_as_string)
        
        # images_path = os.path.join(path, "images")

        # os.makedirs(images_path, exist_ok=True)

        # for bundle in annotation_bundels:
        #     bundle.image_container.save_image(images_path)
        
