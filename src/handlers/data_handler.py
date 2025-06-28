from abc import ABC, abstractmethod
from ..data.annotation_bundle import AnnotationBundle
from typing import List
from ..data.Logger.colon_logger import ColonLogger

class DataHandler(ABC):
    def __init__(self):
        self.logger: ColonLogger = ColonLogger()

    @abstractmethod
    def load(self, path: str) -> List[AnnotationBundle]:
        self.logger.set_base("loading")
    
    @abstractmethod
    def save(self, annotation_bundels: List[AnnotationBundle], label_names: str, path: str, validation_split: int):
        self.logger.set_base("saving")