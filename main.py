import cv2
import numpy as np
import matplotlib.pyplot as plt
import linear_filter
import math
import random
import os

def add_noise(img, probability):
    output_img = np.zeros(img.shape, np.uint8)
    threshold = 1 - probability

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_number = random.random()

            if random_number < probability:
                output_img[i][j] = 0  # set pixel to black
            elif random_number > threshold:
                output_img[i][j] = 255  # set pixel to white
            else:
                output_img[i][j] = img[i][j] # keep pixel same

    return output_img


def Kuwahara_filter( # from section 3
    img: np.uint8, # image
    N: int, # number of sectors
    sigma: float, #gaussian strength, best results seem between r and r/2,
    r: float, #radius of the regions,
    q: float, # parameter, controls the sharpness of transition
) -> np.uint8:
    # convert into grayscale into float for computation
    img = img / 255
    img = img.astype("float32")
    diameter = int(r*2)

    half_size = int(diameter / 2)
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    #padding
    img = cv2.copyMakeBorder(img, half_size, half_size, half_size, half_size, cv2.BORDER_REPLICATE);

    sizeX, sizeY = img.shape
    g_kernel = linear_filter.gauss_kernel_generator(diameter, sigma) / (2 * math.pi * sigma)
    g_kernel_over4 = linear_filter.gauss_kernel_generator(diameter, sigma / 16.0) / (2 * math.pi * sigma / 16.0)

    # precompute V[i] (7) and w[i] (8) according to N and r
    w = np.ndarray((N, diameter, diameter))
    U = np.zeros((N, diameter, diameter))

    # cut the circle into N chunks, U[i]
    for x in range(diameter):
        for y in range(diameter):
            cx = x - half_size
            cy = (y - half_size) * -1
            radians = ((math.atan2(cy, cx) % (2 * math.pi)) + (math.pi / N)) % (2 * math.pi) # radians + shift
            i = int(N * (radians / (2 * math.pi)))
            U[i, y, x] = N

    arr = np.zeros((N))
    middle = U[0, diameter // 2, diameter // 2]
    for i in range(0, N):
        U[i, diameter // 2, diameter // 2] = middle / N
        Vi = U[i] * g_kernel_over4
        w[i] = g_kernel * U[i]
        arr[i] = np.sum(w[i]) / N

    """
    #check, shows the cutting and weighting functions based on the Parameters
    for i in range(0, N):
        imgcheck = U[i] * 255
        imgcheck = np.uint8(imgcheck)
        cv2.imwrite(("Ui_check" + str(i) + ".png"), imgcheck)
        imgcheck = w[i] * 255 / np.max(w)
        imgcheck = np.uint8(imgcheck)
        cv2.imwrite(("wi_check" + str(i) + ".png"), imgcheck)
        """

    #loop through image, and for each,
    for x in range(half_size, sizeX - half_size):
        for y in range(half_size, sizeY - half_size):
            m = np.zeros(N, dtype=float)
            s2 = np.zeros(N, dtype=float)
            top = 0.0
            bottom = 0.0
            for i in range(0, N):
                # compute weighted local average mi, (5)
                img_window = img[x-half_size:x+half_size, y-half_size:y+half_size]
                m[i] = np.sum(img_window * w[i])
                # compute si^2 (5)
                #s2[i] = np.sum(np.matmul(img_window, img_window) * w[i]) - (m[i] * m[i])
                s2[i] = np.sum(np.square(img_window) * w[i]) - (m[i] * m[i])
                s2[i] = math.sqrt(max(s2[i], 0.0000000000000001)) # sqrt because we use si not s^2i
                top += m[i] * math.pow(s2[i], -q)
                bottom += math.pow(s2[i], -q)
            # compute final output (11)
            if bottom != 0:
                img_filtered[x - half_size, y - half_size] = max(min((top / bottom), 1.0), 0.0)
            else:
                img_filtered[x - half_size, y - half_size] = 0.0

    # convert back into grayscale
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered


def run(filename, filetype, N, sigma, r, q, addnoise):
    """
    # Kuwahara Parameters
    N = 8 # number of sectors
    sigma = 6.0 # sigma, ussually good between r and r/2
    r = 9.0 # radius of the filter
    q = 5 # sharpness, between 3 and 8 for best effect, 0 is essentially guassian blur
    """
    # read the rgb image
    rgb_filename = 'data/' + filename + "." + filetype
    im = cv2.imread(rgb_filename)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # add salt and pepper noise to dataset
    if addnoise:
        gray = add_noise(gray, 0.02)

    # Kuwahara filter
    output = Kuwahara_filter(gray, N, sigma, r, q)

    # Gaussian blur
    spatial_var = sigma * sigma  # sigma_s^2
    kernel_size = max(int(r), 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian = cv2.GaussianBlur(gray, (int(kernel_size), int(kernel_size)), sigma)

    # For comparing Bilateral filtering
    intensity_variance = 10.0
    # Bilateral filtering
    bilateral = cv2.bilateralFilter(gray, kernel_size, intensity_variance, spatial_var)

    os.makedirs('results/' + filename, exist_ok=True)
    if addnoise:
        cv2.imwrite('results/' + filename + '/grayscaled_with_noise.png', gray)
        cv2.imwrite('results/' + filename + '/Kuwahara_with_noise.png', output)
        cv2.imwrite('results/' + filename + '/gaussian_with_noise.png', gaussian)
        cv2.imwrite('results/' + filename + '/bilateral_with_noise.png', bilateral)
        print('results/' + filename + " with noise")
    else:
        cv2.imwrite('results/' + filename + '/grayscaled.png', gray)
        cv2.imwrite('results/' + filename + '/Kuwahara.png', output)
        cv2.imwrite('results/' + filename + '/gaussian.png', gaussian)
        cv2.imwrite('results/' + filename + '/bilateral.png', bilateral)
        print('results/' + filename)

"""
    # visualization for debugging
    fig = plt.figure()
    # show input image
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    ax.set_title('input image')
    # show output image
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(output, cmap="gray")
    ax.set_title('Kuwahara filter')
    # show gaussian image
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(gaussian, cmap="gray")
    ax.set_title('gaussian blur')
    # show bilateral image
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(bilateral, cmap="gray")
    ax.set_title('bilateral filter')
    plt.show()
"""

# main function
if __name__ == '__main__':
    # parameters holds data about the settings to be used for the filters
    with open("data/parameters.txt") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        numbers = line.split()
        if numbers[5] == 'False':
            numbers[5] = False
        run(numbers[6], numbers[4], int(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3]), bool(numbers[5]))
