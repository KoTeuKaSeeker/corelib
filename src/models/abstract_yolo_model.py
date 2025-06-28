from abc import abstractmethod
from .abstract_model import AbstractModel
from ..containers.image_container import ImageContainer
from ..data.annotation_bundle import AnnotationBundle
from ..data.annotation import Annotation
from ultralytics import YOLO
import torch
from typing import List
import timeit

class AbstractYoloModel(AbstractModel):
    def __init__(self, model_path: str):
        super().__init__()
        self._model = YOLO(model_path)

        if torch.cuda.is_available():
            self._model.to('cuda')
    

    @abstractmethod
    def handle_prediction_result(self, result, image_container: ImageContainer) -> List[Annotation]:
        return []


    # В зависимости от формата предсказания конкретной модели, формирует унифицированный список объектов класса Annotation
    def predict(self, image_containers: List[ImageContainer]) -> List[AnnotationBundle]:
        annotation_bundles: List[AnnotationBundle] = []
        
        for image_container in image_containers:
            image = image_container.get_image()[:, :, :3]
            results = self._model.predict([image], verbose=False)
            
            image_shape = image_container.get_image_shape()   
            for result in results:
                annotations: List[Annotation] = self.handle_prediction_result(result, image_container)
                annotation_bundle = AnnotationBundle(annotations, image_container, self)
                annotation_bundles.append(annotation_bundle)
        
        return annotation_bundles

    # Возвращает унифицированный список AnnotationBundle в зависимости от результата работы метода predict конкретной модели
    def annotate(self, annotation_bundles: List[AnnotationBundle], verbose=True):
        bundles_count = len(annotation_bundles)
        annotated_bundles_count = 0
        if verbose:
            print(f"Annotation started. Total number of annotation bundles: {bundles_count}")
        start = timeit.default_timer()
        for annotation_bundle in annotation_bundles:
            annotation_bundle.annotations += self.predict([annotation_bundle.image_container])[0].annotations
            
            annotated_bundles_count += 1

            end = timeit.default_timer()

            elapsed_time = end - start
            eta = (elapsed_time / annotated_bundles_count) * bundles_count - annotated_bundles_count

            if verbose:
                print(f"{annotated_bundles_count}/{bundles_count}   elapsed time: {elapsed_time:.3f}    eta: {eta:.3f}")


    def get_labels(self) -> List:
        return list(map(lambda x: x[1], self._model.names.items()))