import numpy as np
from PIL import Image

im = Image.open('training_data_singles/entry0.jpg')

pixels = list(im.getdata())

row = 20*192
row1 = 19*192
row2 = 21*192
col = 64+64+35-1
'''
row = 57*192
row1 = 56*64
row2 = 58*64
row3 = 55*64
row4 = 59*64
'''

for i in range(3):

    pixels[row+col + i] = (0,255,0)
    pixels[row1+col + i] = (0,255,0)
    pixels[row2+col + i] = (0,255,0)


im.putdata(pixels)
im.save('figure2.jpg')
