from src.data.annotation_bundle import AnnotationBundle
from src.visualizer.palette.palette_register import PaletteRegister
from src.core.core import Core
import cv2

if  __name__ == "__main__":
    bosch_core = Core(dataset_path="dataset_train_rgb", dataset_format="bosch")

    palette = PaletteRegister().rainbow()
    print("I show something")
    for annotation_bundle in bosch_core._annotation_bundles:
        annotation_bundle: AnnotationBundle

        image = annotation_bundle.image_container.get_image()
        image = annotation_bundle.draw(image, None, palette)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image)
        cv2.waitKey(0)