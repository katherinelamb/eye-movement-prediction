import numpy as np
import matplotlib.pyplot as plt

def get_data_lists(filename):
    '''
    returns list of lists (timestamp, x, y)
    '''
    gazes = []
    counter = 0
    with open(filename, 'r') as f:
        for line in f:
            counter += 1
            line = line.rstrip('\n')
            contents = line.split(' ')
            if len(contents) == 4 and contents[0][0] != '#' and float(contents[3]) == 1:   
                    contents.pop() # don't need to keep certainty since it's binary
                    contents = [float(content) for content in contents]
                    gazes.append(contents)
    print ('stored', counter, 'gaze coordinates')
    return gazes

def visualize_gaze_data(coords, option=4, interval=None):
    '''
    input:
        Tx3 ndarray, 
        optional int option
        optional tuple interval (start, end) *assumes valid choices
    
    option == 1: plots axis 1 against time axis 0
    option == 2: plots axis 2 against time axis 0
    option == 3: plots all viewed x,y coords
    option == 4: plots magnitude of slope against time
    interval restricts axis 0 to given start and end values
    '''
    start = 0
    end = coords.shape[0]
    x = None
    y = None
    if interval != None and len(interval) == 2:
        start, end = interval
    if option == 1:
        x = coords[start:end, 0]
        y = coords[start:end, 1]
    elif option == 2:
        x = coords[start:end, 0]
        y = coords[start:end, 2]
    elif option == 3:
        x = coords[start:end, 1]
        y = coords[start:end, 2]
    else:
        x = coords[start:end-1, 0]
        dxdt = np.diff(coords[start:end, 1])/np.diff(coords[start:end, 0])
        dydt = np.diff(coords[start:end, 2])/np.diff(coords[start:end, 0])
        euc_slope = np.sqrt(np.square(dxdt) + np.square(dydt))
        y = euc_slope

    plt.scatter(x, y)
    plt.show()


def load_gazes(filename, gaze_data_name):
    coords = get_data_lists(filename) # note coords[0] is timestamp
    coord_matrix = np.array(coords)
    return gaze_data_name, coord_matrix