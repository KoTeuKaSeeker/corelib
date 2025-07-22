from ..models.abstract_model import AbstractModel
from ..handlers import DataHandler
from ..handlers.data_handler_factory import DataHandlerFactory
import cv2
from src.visualizer.palette.abstract_palette import AbstractPalette
from src.visualizer.palette.palette_register import palette_register
from ..lable_interface import ILableable
import copy
import random


class Core(ILableable):
    def __init__(self, dataset_path: str, dataset_format: str="yolo", handler: DataHandler=None):
        self._label_names = []
        self._validation_split = 0.0
        self._dataset_format = dataset_format
        self._dataset_path = dataset_path

        handler = DataHandlerFactory.create_handler(dataset_format) if handler is None else handler
        self._annotation_bundles, self._label_names = handler.load(dataset_path)

        for bundle in self._annotation_bundles:
            bundle._lableable = self

    def export(self, output_path: str, dataset_format: str, validation_split: float):
        handler = DataHandlerFactory.create_handler(dataset_format)
        handler.save(self._annotation_bundles, self._label_names, output_path, validation_split)
    
    def merge(self, core) -> "Core":
        self._annotation_bundles += core._annotation_bundles
        self._label_names = list(set(self._label_names + core._label_names))

        for bundle in core._annotation_bundles:
            bundle._lableable = self
        
        return self

    def shuffle(self) -> "Core":
        random.shuffle(self._annotation_bundles)
        return self

    def annotate(self, model: AbstractModel, verbose=True):
        model.annotate(self._annotation_bundles, verbose=verbose)
        self._label_names = list(set(self._label_names + model.get_label_names()))
    
    def set_label_names_from_annotations_labels(self):
        '''
            Делает, чтобы использовались только те лейблы, которые присутствуют
            в аннотациях.
        '''
        labels = set()
        for bundle in self._annotation_bundles:
            for annotation in bundle.annotations:
                labels = labels.union([annotation.label])
        
        self._label_names = list(labels)
    
    def get_labels(self):
        return self._label_names

    def filter_bundles(self, labels, max_bundles=-1):
        filtred_bundles = set()

        for label in labels:
            label_filtered_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == label, bundle.annotations)), self._annotation_bundles))
            
            if max_bundles >= 0:
                label_filtered_bundles = label_filtered_bundles = label_filtered_bundles[:max_bundles] if len(label_filtered_bundles) > max_bundles else label_filtered_bundles

            filtred_bundles = filtred_bundles.union(label_filtered_bundles)

        self._label_names = list(labels)
        labels = set(labels)
        for bundle in filtred_bundles:
            bundle.annotations = list(filter(lambda annotation: annotation.label in labels, bundle.annotations))

        self._annotation_bundles = list(filtred_bundles)

    def filter_bundles_with_losses(self, labels, max_bundles):
        counts = {label: 0 for label in labels}
        filtred_bundles = set()      
        for bundle in self._annotation_bundles:
            added_annotations = {}
            label_filtered_bundles = {}
            to_add = True
            for annotation in bundle.annotations:
                if annotation.label in labels:
                    if annotation.label not in added_annotations:
                        added_annotations[annotation.label] = counts[annotation.label]
                        label_filtered_bundles[annotation.label] = []
                    added_annotations[annotation.label] += 1
                    label_filtered_bundles[annotation.label].append(annotation)
                
                    if added_annotations[annotation.label] > max_bundles:
                        to_add = False
                        break
            
            if to_add:
                for label, annotation_count in added_annotations.items():
                    counts[label] = annotation_count
                
                filtred_bundles = filtred_bundles.union(label_filtered_bundles)
        
        self._label_names = list(labels)
        labels = set(labels)
        for bundle in filtred_bundles:
            bundle.annotations = list(filter(lambda annotation: annotation.label in labels, bundle.annotations))

        self._annotation_bundles = list(filtred_bundles)
    
    def filter_and_split(self, labels, max_bundles, filter_with_loses=False):
        annotation_bundles = set(self._annotation_bundles)
        
        if filter_with_loses:
            self.filter_bundles_with_losses(labels, max_bundles)    
        else:
            self.filter_bundles(labels, max_bundles)
        
        remaining_bundles = annotation_bundles - set(self._annotation_bundles)

        remaining_core = copy.copy(self)
        remaining_core._annotation_bundles = list(remaining_bundles)
        
        remaning_labels = set()
        for bundle in remaining_core._annotation_bundles:
            for annotation in bundle.annotations:
                remaning_labels = remaning_labels.union([annotation.label])
        
        remaining_core._label_names = list(remaning_labels)

        return remaining_core

    def count_annotations(self, verbose=0):
        counts = {label: 0 for label in self._label_names}
        for bundle in self._annotation_bundles:
            for annotation in bundle.annotations:
                counts[annotation.label] += 1
        
        if verbose > 0:
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1]) if verbose > 1 else counts.items()
            for label, count in sorted_counts:
                print(f"{label}: {count}")

        return counts

    def show_bundles(self, palette: AbstractPalette = palette_register.palettes["rainbow"], target_width: int = 1000):
        for bundle in self._annotation_bundles:
            image = bundle.draw_pp(palette=palette, target_width=target_width)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()