import numpy as np
from ..data.annotation import Annotation
from ..data.mask import Mask
from ..data.box import Box
from ..data.annotation_bundle import AnnotationBundle
from .pallete import Pallete
import cv2
# from ..containers.image_container import ImageContainer


class Visualizer():
    def __init__(self):
        pass

    def show_image(self, image: np.ndarray, width: int = -1):
        rgb_image = image.copy()
        
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        if width != -1 and width != rgb_image.shape[1]:
            aspect_ratio = bgr_image.shape[0] / bgr_image.shape[1]
            height = int(width * aspect_ratio)

            bgr_image = cv2.resize(bgr_image, (width ,height))

        cv2.imshow("Image", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def draw_mask(self, image: np.ndarray, mask: Mask, pallete: Pallete):
        counter = mask.points.copy().reshape((-1, 1, 2)).astype(int)
        
        print(mask.points)
        print(counter.shape)
        
        color = pallete.get_color(mask.label)
        cv2.drawContours(image, [counter], contourIdx=-1, color=color, thickness=-1)
    
    def draw_box(self, image: np.ndarray, box: Box):
        pass

    def draw_annotation(self, image, annotation: Annotation, pallate: Pallete):
        if isinstance(annotation, Mask):
            self.draw_mask(image, annotation, pallate)
        elif isinstance(annotation, Box):
            self.draw_box(image, annotation)

    def show_annotation_bundle(self, annotation_bundle: AnnotationBundle, pallate: Pallete, width=-1):
        image = annotation_bundle.image_container.get_image().copy()

        for annotation in annotation_bundle.annotations:
            self.draw_mask(image, annotation, pallate)
        
        self.show_image(image, width=width)