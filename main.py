from src.core.core import Core

if  __name__ == "__main__":
    rtsd_core = Core(r"datasets\tsd-2", "yolo")

    rtsd_core.show_bundles()