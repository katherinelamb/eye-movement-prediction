import numpy as np
import os
import saccade_parser as sp
import video_utils

if __name__ == "__main__":
    files = []
    for filename in os.listdir('gaze/natural_movies_gaze'):
        video_start_index = filename.find('_')+1
        video_end_index = filename.find('.coord')
        video_path = '../movies-m2t/' + filename[video_start_index: video_end_index] + '.m2t'
        files.append((os.path.join('gaze/natural_movies_gaze', filename), video_path))

    entry = 0
    for filepath, videopath in files:
        intervals = sp.make_intervals(filepath)
        velocities = sp.get_velocities(intervals)
        data = sp.find_saccades(velocities, intervals, videopath)

        # FIGURE OUT WHAT TO DO ABOUT LABEL
        for point, label, timeframe, videopath in data:
            frame = video_utils.get_training_images(timeframe/1000, point[0], point[1], videopath, 'training_data/', 'entry' + str(entry))
            entry += 1
        break