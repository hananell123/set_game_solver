import numpy as np
"""
   +------+
   |      | 0 - green  
   |COLORS| 1 - red    
   |      | 2 - purple 
   +------+

   +------+
   |      | 0 - empty
   |SHADE | 1 - stripes
   |      | 2 - full
   +------+

   +------+
   |      | 0 - oval
   |SHAPE | 1 - diamond
   |      | 2 - Squiggle
   +------+
"""

def find_sets(card1,card2,card3):
#     color
    color = card1.get_color() + card2.get_color() + card3.get_color()
    if color%3 != 0:
        return False
    shape = card1.get_shape() + card2.get_shape() + card3.get_shape()

    if shape % 3 != 0:
        return False

    shade = card1.get_shade() + card2.get_shade() + card3.get_shade()

    if shade % 3 != 0:
        return False

    number = card1.get_number() + card2.get_number() + card3.get_number()

    if number % 3 != 0:
        return False

    return True



