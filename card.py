
#  3 colors - red,green,purple
# 3 fills - stripes,empty,full
# 3 number 1,2,3
# 3 shapes - elipse,waves,Rhombus
class Card:


    def __init__(self,shape,shade,color,number,countres):
        self.shape =shape
        self.shade =shade
        self.color =color
        self.number =number
        self.countres = countres



    def get_color(self):
        return self.color

    def get_shade(self):
        return self.shade

    def get_shape(self):
        return self.shape

    def get_number(self):
        return self.number
