class rgb():
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __str__(self):
        return f"rgb({self.r}, {self.g}, {self.b})"

    def __repr__(self):
        return f"rgb({self.r}, {self.g}, {self.b})"

    def as_bgr(self) -> tuple:
        return (self.b, self.g, self.r)
    
    def invert(self):
        return rgb(255-self.r, 255-self.g, 255-self.b)
    
    def text_color(self):
        luma = 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

        if luma < 40:
            return rgb(255, 255, 255)
        else:
            return rgb(0, 0, 0)


colors: dict[str, rgb] = {
    "blue_buoy": rgb(33, 49, 255),
    "dock": rgb(132, 66, 0),
    "green_buoy": rgb(135, 255, 0),
    "green_pole_buoy": rgb(0, 255, 163),
    "misc_buoy": rgb(255, 0, 161),
    "red_buoy": rgb(255, 0, 0),
    "red_pole_buoy": rgb(255, 92, 0),
    "yellow_buoy": rgb(255, 255, 0),
    "black_buoy": rgb(0, 0, 0),
    "red_racquet_ball": rgb(255, 153, 0),
    "yellow_racquet_ball": rgb(204, 255, 0),
    "blue_racquet_ball": rgb(102, 20, 219),
}
