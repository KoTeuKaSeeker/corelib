from .abstract_yolo_model import AbstractYoloModel
from ..data.annotation import Annotation
from ..data.box import Box
from ..containers.image_container import ImageContainer
from typing import List
import numpy as np


class YoloDetectionModel(AbstractYoloModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)


    def handle_prediction_result(self, result, image_container: ImageContainer) -> List[Annotation]:
        annotations: List[Annotation] = []
        image_shape = image_container.get_image_shape()
                
        if result.boxes is not None:
            for idx, xyxyn in enumerate(result.boxes.xyxyn):
                points_n = np.reshape(xyxyn.cpu().numpy(), (-1, 2)) # Might be an error
                points = points_n * image_shape
                label = self._model.names[int(result.boxes[idx].cls)]
                
                box = Box(points, points_n, label, image_container, False)
                
                annotations.append(box)
        
        return annotations