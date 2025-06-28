from .abstract_palette import AbstractPalette
from typing import List

class PaletteList(AbstractPalette):
    def __init__(self, colors: List[tuple]):
        super().__init__()
        self.colors = colors


    def get_color(self, value):
        idx = int(value) % len(self.colors)
        return self.colors[idx]