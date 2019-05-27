import numpy as np
import gaze_utils as utils
from matplotlib import pyplot as plt
import scipy.spatial.distance as distance

INT_SIZE = 8000
START_THRESH = .002
END_THRESH = .001

def make_intervals(filename):
    name, coords = utils.load_gazes(filename, 'test')
    # stored as tuples, (start location (tuple), end location (tuple), time difference between locs)
    intervals = []
    timestamps = coords.T[0]
    curr_start = timestamps[0]
    start_index = 0

    while start_index != len(timestamps)-1:
        start_loc = (coords[start_index][1], coords[start_index][2])
        end_index = start_index + 1
        curr_end = curr_start + INT_SIZE
        while (end_index != len(timestamps)-1) and (timestamps[end_index] < curr_end):
            end_index += 1
        # if not an exact interval
        if timestamps[end_index] != curr_end and end_index-1 != start_index:
            if abs(timestamps[end_index] - curr_end) > abs(timestamps[end_index-1] - curr_end):
                end_index = end_index-1
        end_loc = (coords[end_index][1], coords[end_index][2])
        time_diff = abs(timestamps[start_index] - timestamps[end_index])
        intervals.append((start_loc, end_loc, time_diff))
        print(start_loc, end_loc, time_diff)

        # find new start
        curr_start += int(INT_SIZE/2)
        new_start_index = start_index + 1
        while new_start_index != len(timestamps)-1 and timestamps[new_start_index] < curr_start:
            new_start_index += 1

        #if not an exact interval
        if timestamps[new_start_index] != curr_start and new_start_index-1 != start_index:
            if abs(timestamps[new_start_index] - curr_start) > abs(timestamps[new_start_index-1] - curr_start):
                new_start_index = new_start_index-1
        start_index = new_start_index
        curr_start = timestamps[start_index]
    intervals = np.array(intervals)
    return intervals

def get_velocities(intervals):
    vels = []
    for i in intervals:
        d = distance.euclidean(i[0], i[1])
        v = d/i[2]
        vels.append(v)
    vels = np.array(vels)
    return vels

def plot_velocity(intervals):
    vels = []
    for i in intervals:
        d = distance.euclidean(i[0], i[1])
        v = d/i[2]
        vels.append(v)
    x = np.arange(len(intervals))
    #plt.plot(x, vels)
    #plt.savefig('beach1_vels')

    diffs = []
    for i in range(len(vels)-1):
        diff = abs(vels[i] - vels[i+1])
        diffs.append(diff)
    x = np.arange(len(intervals)-1)
    plt.plot(x, diffs)
    #n, bins, patches = plt.hist(diffs, bins=100)
    #print(bins)
    #plt.ylim([0, 600])
    plt.hlines(.002, 0, x[len(x)-1], color='black', label='Start saccade threshold')
    plt.hlines(.001, 0, x[len(x)-1], color='red', label='End saccade threshold')
    plt.legend()
    plt.xlabel('Interval number (8ms)')
    plt.ylabel('Acceleration')
    plt.title('Beach Video Eye Accelerations Over Time')
    plt.savefig('AAF_beach_1_vels_diff')

def findSaccade(vels, intervals, vid_name):
    started_saccade = False
    data = []
    current_start_loc = None
    for i in range(len(vels)-1):
        diff = abs(vels[i] - vels[i+1])
        if started_saccade is False:
            if diff >= START_THRESH:
                started_saccade = True
                current_start_loc = intervals[i][0]
        else:
            if diff <= END_THRESH:
                started_saccade = False
                label = intervals[i][1]
                data.append([current_start_loc, label])
                current_start_loc = None
    print(data)


if __name__ == "__main__":
    intervals = make_intervals('gaze/natural_movies_gaze/AAF_beach.coord')
    #plot_velocity(intervals)
    velocities = get_velocities(intervals)
    findSaccade(velocities, intervals)
