from abc import ABC, abstractmethod
import numpy as np
from .palette.abstract_palette import AbstractPalette
import cv2

class IDrawable(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray = None, mask_image: np.ndarray = None, palette: AbstractPalette = None):
        return image

    def draw_pp(self, image: np.ndarray = None, mask_image: np.ndarray = None, palette: AbstractPalette = None, target_width: int = -1):
        image = self.draw(image, mask_image, palette)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if target_width >= 0:
            target_height = image.shape[0] / image.shape[1] * target_width
            image = cv2.resize(image, (int(target_width), int(target_height)))
        
        return image
        
        