from .yolo_image_handler import YoloImageHandler

class YoloSegImageHandler(YoloImageHandler):
    def __init__(self, is_segmentation=True):
        super().__init__(is_segmentation)