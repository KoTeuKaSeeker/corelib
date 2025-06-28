from abc import ABC, abstractmethod

class Logger():
    def __init__(self):
        self._base = "default logger base"
        self._max_count = 0

    def set_base(self, base):
        self._base = base
    
    def set_max_count(self, max_count):
        self._max_count = max_count

    @abstractmethod
    def print(self, _message):
        pass

    @abstractmethod
    def print_counter(self, counter):
        pass