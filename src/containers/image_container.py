from abc import abstractmethod
import numpy as np

class ImageContainer:
    def __init__(self, image_name: str):
        self._image_name = image_name
    
    @property
    def image_name(self) -> str:
        return self._image_name
    
    @abstractmethod
    def get_image(self) -> np.array:
        pass

    @abstractmethod
    def get_image_shape(self) -> np.array:
        pass
    
    @abstractmethod
    def save_image(self, path: str) -> np.array:
        pass