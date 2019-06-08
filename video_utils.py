import errno
import imageio
import numpy as np
import os
from PIL import Image
import skvideo.io # pip install sk-video
import cv2 # pip install opencv-python or  pip3 install opencv-python
from os.path import isfile, join

def create_dir(directory):
    '''
    creates a directory if the path given does not yet exist
    else does nothing
    '''
    if not os.path.exists(os.path.dirname(directory)):
        try:
            os.makedirs(os.path.dirname(directory))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

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
    create_dir(directory)
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

    '''
    images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save('test.jpg')
    '''

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
    #print('data read in, shape:', videodata.shape)

    fpms = get_frames_per_millisecond(videopath)
    frame_num = get_frame_num(time, fpms)
    frame = videodata[frame_num-1]
    #imageio.imsave('frame.jpg', first_frame)
                            #row, col

    H, W, D = frame.shape
    row = H - y - 1
    col = x
    # comment out this line (call to triple crop) to just create labels
    triplecrop(frame, (row, col), directory, name)
    return row, col, H     # used to create labels


def combine_triple_crops_in_dir(path):
    '''
    used to combine three separate crop images made before network input
    format change

    input: location of crops -  directory ending in a '/'
    should pass in a directory which has 3 * (number of entries)
    standard naming scheme is expected to be entry<entryid>_<crop number from 0 to 2>
    ex) entry1_0.jpg
    '''
    crop_paths = sorted([f for f in os.listdir(path) if isfile(join(path, f)) and f != '.DS_Store'])

    # print (crop_paths)
    for i in range(len(crop_paths)//3):
        crop_group = 3 * i
        p0 = path + crop_paths[crop_group]
        p1 = path + crop_paths[crop_group + 1]
        p2 = path + crop_paths[crop_group + 2]
        crop0 = imageio.imread(p0)
        crop1 = imageio.imread(p1)
        crop2 = imageio.imread(p2)
        H = 64
        W = 64*3
        C = 3
        triple = np.zeros((H,W,C), dtype=np.uint8)
        triple[:, :W//3, :] += crop0
        triple[:, W//3 : 2*W//3, :] += crop1
        triple[:, 2*W//3:, :] += crop2

        save_dir = path[:-1] + '_singles/'
        # can't use i because sorted doesn't keep crops in order, just together
        entry_id = crop_paths[crop_group][5:-6]
        trip_name = 'entry' + entry_id + '.jpg'
        create_dir(save_dir)
        imageio.imsave(save_dir + trip_name, triple)
        print(trip_name, 'composed from', p0)
