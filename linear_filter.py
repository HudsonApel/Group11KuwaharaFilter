import cv2
import numpy as np
import math
import sys

# linear local filter
def linear_local_filtering(
    img: np.uint8,
    filter_weights: np.ndarray,
) -> np.uint8:

    img = img / 255
    img = img.astype("float32") # input image
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    kernel_size = filter_weights.shape[0] # filter kernel size
    sizeX, sizeY = img.shape

    padded_image = np.pad(img, kernel_size // 2, mode='constant', constant_values=0)

    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):
            starting_i_index = i - kernel_size // 2
            ending_i_index = i + 1 + kernel_size // 2
            starting_j_index = j - kernel_size // 2
            ending_j_index = j + 1 + kernel_size // 2

            result = padded_image[starting_i_index: ending_i_index, starting_j_index: ending_j_index]
            img_filtered[i, j] = np.sum(result * filter_weights)

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

# gaussian kernel generator
def gauss_kernel_generator(kernel_size: int, spatial_variance: float) -> np.ndarray:
    kernel_weights = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            adjusted_i = i - kernel_size // 2
            adjusted_j = j - kernel_size // 2

            squared_distance = adjusted_i ** 2 + adjusted_j ** 2
            kernel_weights[i, j] = np.exp(-squared_distance / (2 * spatial_variance))

    return kernel_weights