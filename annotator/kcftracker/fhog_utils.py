import numpy as np
from numba.pycc import CC
from numba import jit
import math

# constant
NUM_SECTOR = 9
FLT_EPSILON = 0.0000001  # To not have division by zero

cc = CC('fhog_utils')

@cc.export('map_points_to_bins', '(f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:], i8[:,:,:], i8, i8, i8)')
def map_points_to_bins(dx, dy, bin_boundary_x, bin_boundary_y, r, chosen_bins, height, width, num_channels):
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            channel = 0
            x = dx[j, i, channel]
            y = dy[j, i, channel]
            r[j, i] = math.sqrt(x * x + y * y)

            for ch in range(1, num_channels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = math.sqrt(tx * tx + ty * ty)
                if magnitude > r[j, i]:
                    r[j, i] = magnitude
                    x = tx
                    y = ty

            max_projection_magnitude = bin_boundary_x[0] * x + bin_boundary_y[0] * y

            max_index = 0

            for kk in range(0, NUM_SECTOR):
                dot_prod = bin_boundary_x[kk] * x + bin_boundary_y[kk] * y
                if dot_prod > max_projection_magnitude:
                    max_projection_magnitude = dot_prod
                    max_index = kk
                elif -dot_prod > max_projection_magnitude:
                    max_projection_magnitude = -dot_prod
                    max_index = kk + NUM_SECTOR

            chosen_bins[j, i, 0] = max_index % NUM_SECTOR
            chosen_bins[j, i, 1] = max_index


@cc.export('aggregate_to_hog_feature_map', '(f4[:], f4[:,:], i8[:,:,:], i8[:], f4[:,:], i8, i8, i8, i8, i8, i8, i8)')
def aggregate_to_hog_feature_map(feature_map, r, alpha, nearest_cell, cell_weights, cell_size, height, width,
                                 cells_amount_direction_x, cells_amount_direction_y, amount_of_orientation_bins, row_size):
    """The algorithm goes through each cell one by one and each pixel one by one
      and allocates the computed magnitude to the right histogram bins in the right
      cells. It also performs interpolation between cells as can be seen by the nearest_cell and cell_weights."""
    for i in range(cells_amount_direction_y):
        for j in range(cells_amount_direction_x):
            for ii in range(cell_size):
                for jj in range(cell_size):
                    if (i * cell_size + ii > 0) and (i * cell_size + ii < (height - 1)) and (j * cell_size + jj > 0) and (j * cell_size + jj < (width - 1)):

                        feature_map[i * row_size + j * amount_of_orientation_bins
                                    + alpha[cell_size * i + ii, j * cell_size + jj, 0]] += r[cell_size * i + ii,
                                                                                             j * cell_size + jj] * \
                                                                                           cell_weights[ii, 0] * \
                                                                                           cell_weights[jj, 0]
                        feature_map[i * row_size + j * amount_of_orientation_bins
                                    + alpha[cell_size * i + ii, j * cell_size + jj, 1] + NUM_SECTOR] += r[cell_size * i + ii,
                                                                                                          j * cell_size + jj] * \
                                                                                                        cell_weights[ii, 0] * \
                                                                                                        cell_weights[jj, 0]

                        if (i + nearest_cell[ii] >= 0) and (i + nearest_cell[ii] <= (cells_amount_direction_y - 1)):
                            feature_map[(i + nearest_cell[ii]) * row_size + j * amount_of_orientation_bins
                                        + alpha[cell_size * i + ii, j * cell_size + jj, 0]] += r[cell_size * i + ii,
                                                                                                 j * cell_size + jj] * \
                                                                                               cell_weights[ii, 1] * \
                                                                                               cell_weights[jj, 0]
                            feature_map[(i + nearest_cell[ii]) * row_size + j * amount_of_orientation_bins
                                        + alpha[cell_size * i + ii, j * cell_size + jj, 1] + NUM_SECTOR] += r[cell_size * i + ii,
                                                                                                              j * cell_size + jj] * \
                                                                                                            cell_weights[ii, 1] * \
                                                                                                            cell_weights[jj, 0]

                        if (j + nearest_cell[jj] >= 0) and (j + nearest_cell[jj] <= (cells_amount_direction_x - 1)):
                            feature_map[i * row_size + (j + nearest_cell[jj]) *
                                        + alpha[cell_size * i + ii, j * cell_size + jj, 0]] += r[cell_size * i + ii,
                                                                                                 j * cell_size + jj] * \
                                                                                               cell_weights[ii, 0] * \
                                                                                               cell_weights[jj, 1]
                            feature_map[i * row_size + (j + nearest_cell[jj]) * amount_of_orientation_bins
                                        + alpha[cell_size * i + ii, j * cell_size + jj, 1] + NUM_SECTOR] += r[cell_size * i + ii,
                                                                                                              j * cell_size + jj] * \
                                                                                                            cell_weights[ii, 0] * \
                                                                                                            cell_weights[jj, 1]
                        if (i + nearest_cell[ii] >= 0) and (i + nearest_cell[ii] <= (cells_amount_direction_y - 1)
                                                            and (j + nearest_cell[jj] >= 0)) and (j + nearest_cell[jj] <= (cells_amount_direction_x - 1)):
                            feature_map[(i + nearest_cell[ii]) * row_size + (j + nearest_cell[jj]) * amount_of_orientation_bins
                                        + alpha[cell_size * i + ii, j * cell_size + jj, 0]] += r[cell_size * i + ii,
                                                                                                 j * cell_size + jj] * \
                                                                                               cell_weights[ii, 1] * \
                                                                                               cell_weights[jj, 1]

                            feature_map[(i + nearest_cell[ii]) * row_size + (j + nearest_cell[jj]) * amount_of_orientation_bins
                                        + alpha[cell_size * i + ii, j * cell_size + jj, 1] + NUM_SECTOR] += r[cell_size * i + ii,
                                                                                                              j * cell_size + jj] * \
                                                                                                            cell_weights[ii, 1] * \
                                                                                                            cell_weights[jj, 1]


@cc.export('create_normalized_features', '(f4[:], f4[:], f4[:], i8, i8, i8, i8, i8)')
def create_normalized_features(new_data, part_of_norm, feature_map, cells_amount_direction_x, cells_amount_direction_y, amount_of_orientation_bins_per_channel, amount_of_orientation_bins, amount_of_features_per_cell):
    """Creates 4 normalized features for each cell, creates the features by normalizing the targeted cell
       with the norm of the 4 cells in that direction(incl. itself) - e.g. southwest, northwest, northeast, southeast"""
    for i in range(1, int(cells_amount_direction_y + 1)):
        for j in range(1, int(cells_amount_direction_x + 1)):
            pos1 = i * (cells_amount_direction_x + 2) * amount_of_orientation_bins + j * amount_of_orientation_bins
            pos2 = (i - 1) * cells_amount_direction_x * amount_of_features_per_cell + (j - 1) * amount_of_features_per_cell
            val_of_norm = math.sqrt(part_of_norm[i * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[i * (cells_amount_direction_x + 2) + (j + 1)] +
                                    part_of_norm[(i + 1) * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[(i + 1) * (cells_amount_direction_x + 2) + (j + 1)]) + FLT_EPSILON
            new_data[pos2:pos2 + amount_of_orientation_bins_per_channel] = feature_map[pos1:pos1 + amount_of_orientation_bins_per_channel] / val_of_norm
            new_data[pos2 + 4 * amount_of_orientation_bins_per_channel:pos2 + 6 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1 + amount_of_orientation_bins_per_channel:pos1 + 3 * amount_of_orientation_bins_per_channel] / val_of_norm

            val_of_norm = math.sqrt(part_of_norm[i * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[i * (cells_amount_direction_x + 2) + (j + 1)] +
                                    part_of_norm[(i - 1) * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[(i - 1) * (cells_amount_direction_x + 2) + j + 1]) + FLT_EPSILON
            new_data[pos2 + amount_of_orientation_bins_per_channel:pos2 + 2 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1:pos1 + amount_of_orientation_bins_per_channel] / val_of_norm
            new_data[pos2 + 6 * amount_of_orientation_bins_per_channel:pos2 + 8 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1 + amount_of_orientation_bins_per_channel:pos1 + 3 * amount_of_orientation_bins_per_channel] / val_of_norm

            val_of_norm = math.sqrt(part_of_norm[i * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[i * (cells_amount_direction_x + 2) + (j - 1)] +
                                    part_of_norm[(i + 1) * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[(i + 1) * (cells_amount_direction_x + 2) + (j - 1)]) + FLT_EPSILON
            new_data[pos2 + 2 * amount_of_orientation_bins_per_channel:pos2 + 3 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1:pos1 + amount_of_orientation_bins_per_channel] / val_of_norm
            new_data[pos2 + 8 * amount_of_orientation_bins_per_channel:pos2 + 10 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1 + amount_of_orientation_bins_per_channel:pos1 + 3 * amount_of_orientation_bins_per_channel] / val_of_norm

            val_of_norm = math.sqrt(part_of_norm[i * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[i * (cells_amount_direction_x + 2) + (j - 1)] +
                                    part_of_norm[(i - 1) * (cells_amount_direction_x + 2) + j] +
                                    part_of_norm[(i - 1) * (cells_amount_direction_x + 2) + (j - 1)]) + FLT_EPSILON
            new_data[pos2 + 3 * amount_of_orientation_bins_per_channel:pos2 + 4 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1:pos1 + amount_of_orientation_bins_per_channel] / val_of_norm

            new_data[pos2 + 10 * amount_of_orientation_bins_per_channel:pos2 + 12 * amount_of_orientation_bins_per_channel] = \
                feature_map[pos1 + amount_of_orientation_bins_per_channel:pos1 + 3 * amount_of_orientation_bins_per_channel] / val_of_norm


@cc.export('hog_pca', '(f4[:], f4[:], i8, i8, i8, i8, i8, i8, f8, f8)')
def hog_pca(new_data, feature_map, total_amount_of_features_per_cell, cells_amount_direction_x, cells_amount_direction_y,
           new_amount_of_features, normalization_features, amount_of_bins_per_channel, nx, ny):
    for i in range(cells_amount_direction_y):
        for j in range(cells_amount_direction_x):
            pos1 = (i * cells_amount_direction_x + j) * total_amount_of_features_per_cell
            pos2 = (i * cells_amount_direction_x + j) * new_amount_of_features
            # Does not seem to be PCA, he just creates his own kind of base based on channel and histogram bin
            # And then also creates vector based on each normalization features
            for jj in range(2 * amount_of_bins_per_channel):  # 2*9
                new_data[pos2 + jj] = np.sum(feature_map[
                                             pos1 + normalization_features * amount_of_bins_per_channel + jj:
                                             pos1 + 3 * normalization_features * amount_of_bins_per_channel + jj: 2 * amount_of_bins_per_channel]) * ny
            for jj in range(amount_of_bins_per_channel):  # 9
                new_data[int(pos2 + 2 * amount_of_bins_per_channel + jj)] = np.sum(feature_map[
                                                                                   pos1 + jj: pos1 + jj + normalization_features * amount_of_bins_per_channel:
                                                                                   amount_of_bins_per_channel]) * ny
            for ii in range(normalization_features):  # 4
                new_data[int(pos2 + 3 * amount_of_bins_per_channel + ii)] = \
                    np.sum(feature_map[pos1 + normalization_features * amount_of_bins_per_channel + ii * amount_of_bins_per_channel * 2:
                                       pos1 + normalization_features * amount_of_bins_per_channel + ii * amount_of_bins_per_channel * 2 +
                                       2 * amount_of_bins_per_channel]) * nx
