import cv2
import numpy as np
import math

# bilateral filter
def bilateral_filtering(
        img: np.uint8,
        spatial_variance: float,
        intensity_variance: float,
        kernel_size: int,
) -> np.uint8:
    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape)  # Placeholder of the filtered image

    padded_image = np.pad(img, kernel_size // 2, mode='constant', constant_values=0)
    x, y = img.shape

    for i in range(kernel_size // 2, x + kernel_size // 2):
        for j in range(kernel_size // 2, y + kernel_size // 2):
            i_start = i - kernel_size // 2
            i_end = i + 1 + kernel_size // 2
            j_start = j - kernel_size // 2
            j_end = j + 1 + kernel_size // 2

            k = np.arange(i_start, i_end)
            l = np.arange(j_start, j_end)

            k_meshed, l_meshed = np.meshgrid(k, l)
            sw_num = -((k_meshed - i) ** 2 + (l_meshed - j) ** 2)

            sw = np.exp(sw_num / (2 * spatial_variance))

            iw_num = -(padded_image[k, l] - padded_image[i, j]) ** 2
            iw = np.exp(iw_num) / (2 * intensity_variance)

            bw = sw * iw

            img_filtered[i - kernel_size // 2, j - kernel_size // 2] = np.sum(
                bw * padded_image[k_meshed, l_meshed]) / np.sum(bw)

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered