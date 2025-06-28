from abc import ABC
import numpy as np
from ..containers.explicit_image_container import ExplicitImageContainer
from ..visualizer.drawable import IDrawable
from .annotation_bundle import AnnotationBundle

class Annotation(ABC, IDrawable):
    def __init__(self, 
                 points: np.ndarray,
                 points_n: np.ndarray,
                 label: str, 
                 image_container: ExplicitImageContainer,
                 is_valid: bool,
                 annotation_bundle: AnnotationBundle = None):
        self._points = points
        self._points_n = points_n
        self._label = label
        self._image_container = image_container
        self._is_valid = is_valid
        self._annotation_bundle = annotation_bundle
    

    @property
    def points(self):
        return self._points
    

    @property
    def points_n(self):
        return self._points_n
    

    @property
    def label(self):
        return self._label
    

    @property
    def image_container(self):
        return self._image_container
    

    @property
    def is_valid(self):
        return self._is_valid
    
    
    @property
    def annotation_bundle(self):
        return self._annotation_bundle