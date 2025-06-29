from src.data.annotation_bundle import AnnotationBundle
from src.visualizer.palette.palette_register import PaletteRegister
from src.core.core import Core
import cv2

if  __name__ == "__main__":
    lisa_core = Core("D:/datasets/lisa", "lisa")

    palette = PaletteRegister().rainbow()

    for annotation_bundle in lisa_core._annotation_bundles[1980:]:
        annotation_bundle: AnnotationBundle

        image = annotation_bundle.image_container.get_image()
        image = annotation_bundle.draw(image, None, palette)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image)
        cv2.waitKey(0)