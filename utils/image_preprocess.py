import os
import numpy as np


def max_length_of_mask(image_array, mask, target_shape=(64, 64, 64)):
    # get the index of non-zero mask 
    nonzero_coords = np.argwhere(mask)
    min_coords = nonzero_coords.min(axis=0)
    max_coords = nonzero_coords.max(axis=0)

    # caculate center point
    center = (min_coords + max_coords) // 2    

    # caculate the max length from mask and ensure the new boundary
    max_length = (max_coords - min_coords).max() + 20  # max length + 10*2

    return max_length


# Normalize Window Level
def normalize(image, min_bound=-1000., max_bound=400.):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

# Slice a cube from 3D numpy array based on center coordinate
# Return 3D numpy array, size=(h, w, w)
def slice_3d_array(arr, center, h, w, padding=0., dtype=np.float32):
    con_x, con_y, con_z = 0, 0, 0
    buff_x = max(0, int(center[1]-w/2))
    buff_y = max(0, int(center[2]-w/2))
    buff_z = max(0, int(center[0]-h/2))

    # X exceed
    if buff_x == 0:
        con_x = abs(int(center[1]-w/2))
    # Y exceed
    if buff_y == 0:
        con_y = abs(int(center[2]-w/2))
    # Z exceed
    if buff_z == 0:
        con_z = abs(int(center[0]-h/2))

    # Slice nodule
    buff = arr[buff_z:buff_z+h-con_z, buff_x:buff_x+w-con_x, buff_y:buff_y+w-con_y]

    # Padding
    container = np.full((h, w, w), padding)
    container[con_z:con_z+buff.shape[0], con_x:con_x+buff.shape[1], con_y:con_y+buff.shape[2]] = buff

    return container.astype(dtype)
