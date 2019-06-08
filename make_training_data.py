import numpy as np
import os
import saccade_parser as sp
import video_utils
import csv

if __name__ == "__main__":
    files = []
    for filename in os.listdir('gaze/natural_movies_gaze'):
        video_start_index = filename.find('_')+1
        video_end_index = filename.find('.coord')
        video_path = '../movies-m2t/' + filename[video_start_index: video_end_index] + '.m2t'
        files.append((os.path.join('gaze/natural_movies_gaze', filename), video_path))

    entry = 0
    video_num = 0
    for filepath, videopath in files:
        print('video num:', video_num)
        intervals = sp.make_intervals(filepath)
        velocities = sp.get_velocities(intervals)
        data = sp.find_saccades(velocities, intervals, videopath)

        w = csv.writer(open("labels.csv", "a"))
        for point, label, timeframe, videopath in data:
            print('entry:', entry)

            if entry >= 960: # good for picking up where you left off
                # saves a training image
                # comment out line that calls triple crop in get_training_images to only make labels (see comment in function)
                row, col, height = video_utils.get_training_images(timeframe/1000, point[0], point[1], videopath, 'training_data/', 'entry' + str(entry))

                label_x, label_y = label
                label_row = height - label_y - 1
                label_col = label_x

                # move label to be in the 512 x 512 box
                if label_row > row + 512//2:
                    print('change row')
                    label_row = row + 512//2
                elif label_row < row - 512//2:
                    print('change row')
                    label_row = row - 512//2
                if label_col > col + 512//2:
                    print('change col')
                    label_col = col + 512//2
                elif label_col < col - 512//2:
                    print('change col')
                    label_col = col - 512//2

                # scale down label
                top_left_big_crop = (row - 512//2, col - 512//2)
                label_row_in_big_crop = label_row - top_left_big_crop[0]
                label_col_in_big_crop = label_col - top_left_big_crop[1]
                label_row_in_big_crop /= 8
                label_col_in_big_crop /= 8
                label = (round(label_row_in_big_crop), round(label_col_in_big_crop))
                # labels written as (entry number, pathname to training entry, label row, label col)
                w.writerow([entry, 'training_data_singles/entry' + str(entry) + '.jpg', label[0], label[1]])
            entry += 1

        video_num += 1
        # ONCE DONE MAKING TRIPLE CROP IMAGES, CALL COMBINE_TRIPLE_CROPS_IN_DIR
