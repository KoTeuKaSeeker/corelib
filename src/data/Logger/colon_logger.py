from .logger import Logger

class ColonLogger(Logger):
    def __init__(self):
        super().__init__()
    
    def print(self, message):
        print(f"{self._base}: {message}")
    
    def print_counter(self, counter):
        print(f"{self._base}: {counter}/{self._max_count}")