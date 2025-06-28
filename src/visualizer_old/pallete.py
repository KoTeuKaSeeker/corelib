from .color_map import ColorMap


class Pallete:
    def __init__(self):
        self.colors: list = None
        self.key_values: list = None
        self.key_to_color: dict = None
    
    @staticmethod
    def from_colormap(key_values: list, color_map: ColorMap) -> 'Pallete':
        # num_key_values = len()
        pallete = Pallete()
        pallete.key_values = key_values.copy()

        points = [keypoint.point for keypoint in color_map.keypoints]
        min_point = min(points)
        max_point = max(points)
        
        t_size = (max_point - min_point) / (len(pallete.key_values) - 1)
        pallete.colors = []
        for idx in range(len(pallete.key_values)):
            t = min_point + t_size * idx
            t = min(max(t, min_point), max_point)
            color = color_map.get_color(t)
            pallete.colors.append(color)
        
        pallete.key_to_color = {key: color for key, color in zip(pallete.key_values, pallete.colors)}

        return pallete

    def from_colors(self, key_values: list, colors: list):
        pass
    
    def get_color(self, key_value):
        return self.key_to_color[key_value]
    
    # @staticmethod
    # def generate_random_colors(num_colors, seed=42):
    #     random.seed(seed)
    #     colors = []

    #     for idx in range(num_colors):
    #         red = random.randint(0, 255)
    #         green = random.randint(0, 255)
    #         blue = random.randint(0, 255)

    #         color = [red, green, blue]

    #         colors.append(color)
        
    #     return colors


        