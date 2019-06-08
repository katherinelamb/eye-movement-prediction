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
            print(videopath)
            print(point[0])
            print(point[1])
            print(timeframe)
            if entry >= 0: # good for picking up where you left off
                # saves a training image
                row, col, height = video_utils.get_training_images(timeframe/1000, point[0], point[1], videopath, 'training_data/', 'entry' + str(entry))

                label_x, label_y = label
                label_row = height - label_y - 1
                label_col = label_x
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
                if label_row < row:
                    label_row += (row - label_row)/8
                else:
                    label_row -= (label_row - row)/8
                if label_col < col:
                    label_col += (col - label_col)/8
                else:
                    label_col -= (label_col - col)/8
                label = (round(label_row), round(label_col))
                # labels written as (entry, label row, label col)
                w.writerow([entry, label[0], label[1]])
            entry += 1


        video_num += 1
