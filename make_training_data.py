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
            if entry >= 608: # good for picking up where you left off
                frame = video_utils.get_training_images(timeframe/1000, point[0], point[1], videopath, 'training_data/', 'entry' + str(entry))

                label_x, label_y = label
                row, col = point
                if label_x > row + 512//2:
                    print('change x')
                    label_x = row + 512//2
                elif label_x < row - 512//2:
                    print('change x')
                    label_x = row - 512//2
                if label_y > col + 512//2:
                    print('change y')
                    label_y = col + 512//2
                elif label_y < col - 512//2:
                    print('change y')
                    label_y = col - 512//2
                label = (round(label_x), round(label_y))
                w.writerow([entry, label])
            entry += 1


        video_num += 1
