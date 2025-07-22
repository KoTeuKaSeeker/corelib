from src.data.annotation_bundle import AnnotationBundle
from src.visualizer.palette.palette_register import PaletteRegister
from src.core.core import Core
import cv2

if  __name__ == "__main__":
    rtsd_core = Core(r"datasets\tsd-2", "yolo")
    rtsd_core.show_bundles()