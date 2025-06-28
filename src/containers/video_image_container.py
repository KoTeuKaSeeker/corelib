import os

from .image_container import ImageContainer
import uuid
import cv2

class VideoImageContainer(ImageContainer):
    def __init__(self, video_path: str, frame: int):
        #image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_name = str(uuid.uuid4())
        self.__video_path = os.path.normpath(video_path)
        self.__frame = frame

        super().__init__(image_name)

    
    def get_image(self):
        cap = cv2.VideoCapture(self.__video_path)

        if not cap.isOpened():
            raise Exception("Error: Could not open video.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.__frame)

        ret, frame_image = cap.read()

        cap.release()

        if not ret:
            raise Exception(f"Error: Could not read frame {self.__frame}.")
        
        frame_image_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

        return frame_image_rgb
    
    def get_image_shape(self) -> tuple:
        cap = cv2.VideoCapture(self.__video_path)
    
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image_shape = (int(width), int(height))

        cap.release()

        return image_shape


    @property
    def video_path(self):
        return self.__video_path
    
    def save_image(self, dir_path: str):
        image = self.get_image()
        image_bgr_converted = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dir_path, self._image_name + ".jpg"), image_bgr_converted)
        
