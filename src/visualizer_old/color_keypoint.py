
class ColorKeypoint():
    def __init__(self, color: list, point: float):
        self.__color = color
        self.__point = point
    
    @property
    def color(self):
        return self.__color
    
    @property
    def point(self):
        return self.__point