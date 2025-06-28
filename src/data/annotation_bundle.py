from ..visualizer.palette.abstract_palette import AbstractPalette
from ..containers.image_container import ImageContainer
from ..visualizer.drawable import IDrawable
from ..lable_interface import ILableable
import numpy as np
import cv2

mask_alpha = 0.6

class AnnotationBundle(IDrawable):
    def __init__(self, annotations, image_container: ImageContainer, lableable: ILableable = None):
        self._annotations = annotations
        self._image_container = image_container
        self._lableable = lableable

        for annotation in self._annotations:
            annotation._annotation_bundle = self
    

    @property
    def annotations(self):
        return self._annotations
    

    @annotations.setter
    def annotations(self, new_value):
        self._annotations = new_value
    

    @property
    def image_container(self):
        return self._image_container
    

    def draw(self, image: np.ndarray = None, mask_image: np.ndarray = None, palette: AbstractPalette = None):
        image = np.copy(self.image_container.get_image())
        draw_image = np.zeros_like(image)
        mask_image = np.zeros_like(image)
        
        for annotation in self.annotations:
            annotation.draw(draw_image, mask_image, palette)
        
        mask_indices = np.any(mask_image != np.array([0, 0, 0], dtype=np.uint8), axis=-1)
        image[mask_indices] = cv2.addWeighted(image, 1 - mask_alpha, mask_image, mask_alpha, 0)[mask_indices]
        
        draw_indices = np.any(draw_image != np.array([0, 0, 0], dtype=np.uint8), axis=-1)
        image[draw_indices] = draw_image[draw_indices]

        return image