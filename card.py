
#  3 colors - red,green,purple
# 3 fills - stripes,empty,full
# 3 number 1,2,3
# 3 shapes - elipse,waves,Rhombus
class Card:
    Shapes = {
        0: "oval",
        1: "diamond",
        2: "wave"
    }

    Colors = {
        0: "green",
        1: "red",
        2: "purple"
    }

    Shades = {
        0: "full",
        1: "stripes",
        2: "empty"

    }

    def __init__(self,shape,shade,color,number,countres):
        self.shape =shape
        self.shade =shade
        self.color =color
        self.number =number
        self.countres = countres
        self.center = (self.get_center_of_card())



    def get_center_of_card(self):
        max_x, min_x = max([i[0][0] for i in self.countres]), min([i[0][0] for i in self.countres])
        max_y, min_y = max([i[0][1] for i in self.countres]), min([i[0][1] for i in self.countres])
        return int((max_x+min_x)/2),int((max_y+min_y)/2)

    def get_color(self):
        return self.color
    def get_center(self):
        return self.center

    def get_shade(self):
        return self.shade

    def get_shape(self):
        return self.shape

    def get_number(self):
        return self.number

    def get_countres(self):
        return self.countres

    def __str__(self):

        return str(("Number: ",self.number," Shape: ",self.Shapes[self.shape]," Color: ",self.Colors[self.color]," Shade: ",self.Shades[self.shade]))
