from .data_handler import DataHandler
from .cvat_image_handler import CvatImageHandler
from .cvat_video_handler import CvatVideoHandler
from .yolo_image_handler import YoloImageHandler
from .yolo_seg_image_handler import YoloSegImageHandler
from .traffic_light_detection_dataset_handler import TrafficLightDetectionDatasetHandler
from src.handlers.rtsd_handler import RTSDHandler

class DataHandlerFactory:
    @staticmethod
    def create_handler(format_name: str) -> DataHandler:
        switch = {
            "cvat-image": CvatImageHandler,
            "cvat-video": CvatVideoHandler,
            "yolo-seg": YoloSegImageHandler,
            "yolo": YoloImageHandler,
            "traffic-light-detection-dataset": TrafficLightDetectionDatasetHandler,
            "rtsd": RTSDHandler
        }
        
        if format_name not in switch:
            raise Exception("Invalid data handler format.")
        
        handler = switch.get(format_name)
        
        return handler()
