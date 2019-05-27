import errno
import imageio
import numpy as np
import os
from PIL import Image
import skvideo.io # pip install sk-video
import cv2 # pip install opencv-python or  pip3 install opencv-python

def triplecrop(frame, coord, directory, name=''):
    '''
    frame: numpy array shape (H,W)
    coord: (row,col) tuple ** note not x,y **
        center point around which the crops are made
    directory: folder to put crops
    name: optional name for file
    '''

    # print ('image shape', frame.shape)
    # print ('centered around', coord)
    sizes = [128, 256, 512]
    final_width = 64
    final_height = 64
    #increment = 50
    row, col = coord
    crops = []

    if not os.path.exists(os.path.dirname(directory)):
        try:
            os.makedirs(os.path.dirname(directory))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for i in range(3):
        w = sizes[i] // 2
        h = sizes[i] // 2
        left = col-w
        right = col+w
        upper = row-h
        lower = row+h
        box = (left, upper, right, lower)
        img = Image.fromarray(frame).crop(box)
        path = directory + name + '_' + str(i) + '.jpg'
        img = img.resize((final_width, final_height))
        img.save(path)
        #width += increment
        #height += increment

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

def get_frames_per_millisecond(filename):
    cap = cv2.VideoCapture(filename)
    #amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # get the frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps/1000

def get_frame_num(milli, fpms):
    if milli < 1: return 0
    return int(fpms*milli)-1

def get_training_images(time, x, y, videopath, directory, name):
    videodata = skvideo.io.vread(videopath)
    print('data read in, shape:', videodata.shape)


    fpms = get_frames_per_millisecond(videopath)
    frame_num = get_frame_num(time, fpms)
    frame = videodata[frame_num]
    #imageio.imsave('frame.jpg', first_frame)
                            #row, col

    H, W, D = frame.shape
    row = H - y - 1
    col = x
    triplecrop(frame, (row, col), directory, name)
