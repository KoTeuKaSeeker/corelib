from .color_keypoint import ColorKeypoint
from typing import List

class ColorMap():
    def __init__(self):
        self.keypoints: List[ColorKeypoint] = None
    
    @staticmethod
    def from_path(path: str) -> 'ColorMap':
        raise Exception("This method is not yet implemented")
        return ColorMap()
        
    @staticmethod
    def from_keypoints(keypoints: List[ColorKeypoint]) -> 'ColorMap':
        color_map = ColorMap()
        color_map.keypoints = keypoints
        return color_map
    
    def get_color(self, value: float) -> list:
        left_keypoint = None
        right_keypoint = None

        if value == 0.9:
            print("kek")
        
        for keypoint in self.keypoints:
            if keypoint.point == value:
                return keypoint.color
            if keypoint.point < value:
                left_keypoint = keypoint
            if keypoint.point > value:
                right_keypoint = keypoint
                break
        
        if left_keypoint is None or right_keypoint is None:
            raise Exception("Incorrect keypoints.")

        # linear interpolation
        color = [-1, -1, -1]

        t = (value - left_keypoint.point) / (right_keypoint.point - left_keypoint.point)
        for channel in range(3):
            color[channel] = round(left_keypoint.color[channel] + (right_keypoint.color[channel] - left_keypoint.color[channel]) * t)
        
        return color

