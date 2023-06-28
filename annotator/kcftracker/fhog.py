import numpy as np
import cv2
import math
from .fhog_utils import map_points_to_bins, aggregate_to_hog_feature_map, create_normalized_features, hog_pca

# constant
NUM_SECTOR = 9
FLT_EPSILON = 0.0000001  # To not have division by zero

def compile_fhog():
    import glob
    import os
    try:
        from .fhog_utils import cc
        cc.compile()
        print('Done')
    except ImportError:
        path = os.path.dirname(__file__)
        files = [glob.glob(os.path.join(path, 'fhog_utils.*.' + x)) for x in ['pyd', 'so']]
        for file in files:
            if file:
                print('Tracker is ready.')
                print('To re-compile the tracker, remove "{}" and try again.'.format(file[0]))

# This is file involves functions used to compute histogram of oriented gradients
def get_feature_maps(box_image, cell_size, feature_map):
    """Here we compute the edge convolution in x and y direction, with interval k in x and y direction.
    By using the interval k we limit the accuracy of the edge detection but it saves time.
    In our current usage of the function the k is the cell size"""
    kernel = np.array([[-1., 0., 1.]], np.float32)

    height = box_image.shape[0]
    width = box_image.shape[1]
    #print("Height is: {}".format(height))
    #print("Width is: {}".format(width))
    assert (box_image.ndim == 3 and box_image.shape[2])
    num_channels = 3  # (1 if box_image.ndim==2 else box_image.shape[2])

    cells_amount_direction_x = int(width / cell_size)
    cells_amount_direction_y = int(height / cell_size)
    amount_of_orientation_bins = num_channels * NUM_SECTOR
    row_size = int(cells_amount_direction_x * amount_of_orientation_bins)
    feature_map['sizeX'] = cells_amount_direction_x
    feature_map['sizeY'] = cells_amount_direction_y
    feature_map['numFeatures'] = amount_of_orientation_bins
    feature_map['map'] = np.zeros(feature_map['sizeX'] * feature_map['sizeY'] * feature_map['numFeatures'], np.float32)

    # Computing the gradients
    dx = cv2.filter2D(np.float32(box_image), -1, kernel)  # np.float32(...) is necessary #Detecting edges in x-direction
    dy = cv2.filter2D(np.float32(box_image), -1, kernel.T)  # detecting edges in y-direction
    arg_vector = np.arange(NUM_SECTOR + 1).astype(np.float32) * np.pi / NUM_SECTOR
    boundary_x = np.cos(arg_vector)  # The orientations value in x-axis(as vectors)
    boundary_y = np.sin(arg_vector)  # The orientations value in y-axis(as vectors)

    """ Using the gradients in each channel to get the largest value of the channels and then using the resulting gradient
    to calculate the bin-value(for the histogram), where r is the radians """
    r = np.zeros((height, width), np.float32)  # The radians
    alpha = np.zeros((height, width, 2), np.int64)  # Will be the directions in which the maximum gradient was found
    map_points_to_bins(dx, dy, boundary_x, boundary_y, r, alpha, height, width, num_channels)  # with @jit
    # ~0.001s
    nearest_cell = np.ones(cell_size, np.int64)
    nearest_cell[0:math.floor(cell_size / 2)] = -1

    # Computing weights which are used to interpolate between cells
    cell_weights = np.zeros((cell_size, 2), np.float32)
    a_x = np.concatenate(
        (cell_size / 2 - np.arange(cell_size / 2) - 0.5, np.arange(cell_size / 2, cell_size) - cell_size / 2 + 0.5)).astype(
        np.float32)
    b_x = np.concatenate((cell_size / 2 + np.arange(cell_size / 2) + 0.5,
                          -np.arange(cell_size / 2, cell_size) + cell_size / 2 - 0.5 + cell_size)).astype(np.float32)
    cell_weights[:, 0] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
    cell_weights[:, 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))

    # Here we compute the actual HOG-features, by using the weights to sum up the values for each bin  for each cell.
    temporary_feature_map = np.zeros(cells_amount_direction_x * cells_amount_direction_y * amount_of_orientation_bins, np.float32)
    aggregate_to_hog_feature_map(temporary_feature_map, r, alpha, nearest_cell, cell_weights, cell_size, height, width,
                                 cells_amount_direction_x, cells_amount_direction_y, amount_of_orientation_bins, row_size)
    feature_map['map'] = temporary_feature_map
    # ~0.001s

    return feature_map


def normalize_and_truncate(feature_map, alpha):
    cells_amount_direction_x = feature_map['sizeX']
    cells_amount_direction_y = feature_map['sizeY']
    num_channels = 3
    normalization_features = 4
    amount_of_orientation_bins_per_channel = NUM_SECTOR
    amount_of_orientation_bins = amount_of_orientation_bins_per_channel * num_channels
    total_amount_of_features_per_cell = amount_of_orientation_bins * normalization_features
    # 50x speedup
    index = np.arange(0, cells_amount_direction_x * cells_amount_direction_y * feature_map['numFeatures'],
                      feature_map['numFeatures']).reshape(
        (cells_amount_direction_x * cells_amount_direction_y, 1)) + np.arange(amount_of_orientation_bins_per_channel)
    # The divisor-component used to normalize each cell
    part_of_norm = np.sum(feature_map['map'][index] ** 2, axis=1)  # ~0.0002s
    # Removes the cells at the borders
    cells_amount_direction_x, cells_amount_direction_y = cells_amount_direction_x - 2, cells_amount_direction_y - 2

    new_data = np.zeros(cells_amount_direction_y * cells_amount_direction_x * total_amount_of_features_per_cell, np.float32)
    create_normalized_features(new_data, part_of_norm, feature_map['map'], cells_amount_direction_x, cells_amount_direction_y,
                               amount_of_orientation_bins_per_channel, amount_of_orientation_bins,
                               total_amount_of_features_per_cell)  # with @jit

    # truncation
    new_data[new_data > alpha] = alpha

    feature_map['numFeatures'] = total_amount_of_features_per_cell
    feature_map['sizeX'] = cells_amount_direction_x
    feature_map['sizeY'] = cells_amount_direction_y
    feature_map['map'] = new_data

    return feature_map


def pca_feature_maps(feature_map):
    cells_amount_direction_x = feature_map['sizeX']
    cells_amount_direction_y = feature_map['sizeY']

    total_amount_of_features_per_cell = feature_map['numFeatures']
    num_channels = 3
    new_amount_of_features = NUM_SECTOR * num_channels + 4
    normalization_features = 4
    amount_of_bins_per_channel = NUM_SECTOR

    nx = 1.0 / np.sqrt(amount_of_bins_per_channel * 2)
    ny = 1.0 / np.sqrt(normalization_features)

    new_data = np.zeros(cells_amount_direction_x * cells_amount_direction_y * new_amount_of_features, np.float32)
    hog_pca(new_data, feature_map['map'], total_amount_of_features_per_cell, cells_amount_direction_x, cells_amount_direction_y,
            new_amount_of_features, normalization_features, amount_of_bins_per_channel, nx, ny)  # with @jit

    feature_map['numFeatures'] = new_amount_of_features
    feature_map['map'] = new_data

    return feature_map
