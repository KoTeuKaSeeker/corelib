from PIL import Image
import numpy as np
import os

from .image_container import ImageContainer
import uuid
import cv2

class ExplicitImageContainer(ImageContainer):
    def __init__(self, image_path: str):
        #image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_name = str(uuid.uuid4())
        super().__init__(image_name)
        
        self._image_path = os.path.normpath(image_path)
    
    def get_image(self):
        image = np.asarray(Image.open(self._image_path))[:, :, :3]
        return image
    
    def get_image_shape(self) -> tuple:
        with Image.open(self._image_path) as img:
            return img.size

    @property
    def image_path(self):
        return self._image_path
    
    def save_image(self, dir_path: str):
        image = cv2.imread(self._image_path)
        cv2.imwrite(os.path.join(dir_path, self._image_name + ".jpg"), image)
        
