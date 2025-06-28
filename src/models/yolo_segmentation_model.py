from .abstract_yolo_model import AbstractYoloModel
from ..data.annotation import Annotation
from ..containers.image_container import ImageContainer
from typing import List
from src.data.mask import Mask


class YoloSegmentationModel(AbstractYoloModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)


    def handle_prediction_result(self, result, image_container: ImageContainer) -> List[Annotation]:
        annotations: List[Annotation] = []
        image_shape = image_container.get_image_shape()
                
        if result.masks is not None:
            for idx, result_mask in enumerate(result.masks):
                points_n = result_mask.xyn[0] # Might be an error
                points = points_n * image_shape
                label = self._model.names[int(result.boxes[idx].cls)]
                
                mask = Mask(points, points_n, label, image_container, False)
                
                annotations.append(mask)
        
        return annotations