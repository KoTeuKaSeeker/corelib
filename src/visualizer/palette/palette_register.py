from .palette_list import PaletteList

class PaletteRegister():
    def __init__(self):
        self.palettes = {}

    
    def rainbow(self):
        palette = PaletteList(colors=[
            (255, 100, 100), # Красный
            (100, 255, 100), # Синий
            (100, 100, 255), # Зелёный
            (100, 255, 255), # Берюзовый
            (255, 100, 255), # Фиолетовый
            (255, 255, 100), # Жёлтый
            (255, 255, 255), # Белый
            (25, 25, 25),    # Чёрный
        ])

        return palette


    def register_palettes(self):
        self.palettes["rainbow"] = self.rainbow()


palette_register = PaletteRegister()
palette_register.register_palettes()