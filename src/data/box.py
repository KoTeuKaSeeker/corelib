import numpy as np
import cv2
from ..visualizer.palette.abstract_palette import AbstractPalette
from ..containers.explicit_image_container import ExplicitImageContainer
from .annotation import Annotation
from .annotation_bundle import AnnotationBundle

box_draw_thickness = 3
box_text_font_scale = 1
box_text_font_thickness = 3
box_font_padding = 5
box_text_color_white = (255, 255, 255)
box_text_color_black = (25, 25, 25)

class Box(Annotation):
    def __init__(self,
                points: np.ndarray, 
                points_n: np.ndarray,
                label: str, 
                image_container: ExplicitImageContainer,
                is_valid: bool, 
                annotation_bundle: AnnotationBundle = None):
        super().__init__(points, points_n, label, image_container, is_valid, annotation_bundle)


    def draw(self, image: np.ndarray = None, mask_image: np.ndarray = None, palette: AbstractPalette = None):
        points = self.points_n * np.array(image.shape[:2])[::-1]
        color = palette.get_color(self.annotation_bundle._lableable.get_labels().index(self.label))[:3]
        cv2.rectangle(image, points[0].astype(int), points[1].astype(int), color, thickness=box_draw_thickness)

        text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, box_text_font_scale, thickness=box_text_font_thickness)[0]

        p1_text_element = np.array([points[0][0], points[0][1] - box_font_padding * 2 - text_size[1]])
        p2_text_element = p1_text_element + np.array([box_font_padding, box_font_padding]) * 2 + text_size

        cv2.rectangle(image, p1_text_element.astype(int), p2_text_element.astype(int), color, thickness=-1)

        p1_text = p1_text_element + np.array([box_font_padding, box_font_padding])

        box_text_color = box_text_color_white
        if np.array(color).max() > 255 / 2:
            box_text_color = box_text_color_black

        cv2.putText(image, self.label, (p1_text + np.array([0, text_size[1]])).astype(int), cv2.FONT_HERSHEY_SIMPLEX, fontScale=box_text_font_scale, color=box_text_color, thickness=box_text_font_thickness)

        return image