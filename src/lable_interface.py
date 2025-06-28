from abc import abstractmethod

class ILableable():
    @abstractmethod
    def get_labels(self):
        pass