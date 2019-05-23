import errno
import imageio
import numpy as np
import os
from PIL import Image
import skvideo.io # pip install sk-video

def triplecrop(frame, coord, directory, name=''):
    '''
    frame: numpy array shape (H,W)
    coord: (row,col) tuple ** note not x,y **
    directory: folder to put crops
    name: optional name for file
    '''

    # print ('image shape', frame.shape)
    # print ('centered around', coord)
    width = 50
    height = 50
    initial_width = width
    initial_height = height
    increment = 50
    row, col = coord
    crops = []

    if not os.path.exists(os.path.dirname(directory)):
        try:
            os.makedirs(os.path.dirname(directory))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for i in range(3):
        w = width // 2
        h = height // 2
        left = col-w
        right = col+w
        upper = row-h
        lower = row+h
        box = (left, upper, right, lower)
        img = Image.fromarray(frame).crop(box)
        path = directory + name + str(i) + '.jpg'
        img = img.resize((initial_width, initial_height))
        img.save(path)
        width += increment
        height += increment

def load_video_data(path='beach.m2t'):
    videodata = skvideo.io.vread(path) 
    print('data read in, shape:', videodata.shape)
    
    # examples of things that can be done with data
    first_frame = videodata[0]
    imageio.imsave('frame.jpg', first_frame)
    first_frame = imageio.imread('frame.jpg')
                            #row, col
    triplecrop(first_frame, (360, 640), 'triplecrop_middle3/', 'beach')
    triplecrop(first_frame, (100,50), 'triplecrop_edge3/', 'beach')