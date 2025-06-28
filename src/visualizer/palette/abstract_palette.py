from abc import abstractmethod

class AbstractPalette():
    @abstractmethod
    def get_color(self, value):
        pass