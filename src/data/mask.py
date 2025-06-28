from ..visualizer.palette.abstract_palette import AbstractPalette
from ..containers.explicit_image_container import ExplicitImageContainer
from .annotation import Annotation
from .annotation_bundle import AnnotationBundle
from ..data.box import Box
import numpy as np
import cv2

class Mask(Annotation):
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

        cv2.drawContours(mask_image, [points.astype(int)], contourIdx=-1, color=color, thickness=-1)

        p1 = np.array([self.points_n[:, 0].min(), self.points_n[:, 1].min()])
        p2 = np.array([self.points_n[:, 0].max(), self.points_n[:, 1].max()])
        box_points_n = np.array([p1, p2])
        box_points = box_points_n * np.array(image.shape[:2])[::-1]

        box = Box(box_points, box_points_n, self.label, self.image_container, self.is_valid, self.annotation_bundle)
        box.draw(image, mask_image, palette)

        return image