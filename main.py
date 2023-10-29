import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def Kuwahara_filter( # from section 3
    img: np.uint8, # image
    N: int, # number of sectors
    sigma: float, #gaussian strength
    r: float, #radius of the regions, perhaps also kernel_size = 2r?
    q: float, # parameter, controls the sharpness of transition
) -> np.uint8:


    # convert into grayscale into float for computation
    img = img / 255
    img = img.astype("float32")
    
    """
    #old code that zero padded border, don't know if we will do zero padding
    half_size = r
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    #zero-padding
    img = cv2.copyMakeBorder(img, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT, 0);
    """
    sizeX, sizeY = img.shape
    
    g_kernel = gauss_kernel_generator(int(r*2), sigma)
    
    #loop through image, and for each,
    #compute cutting function Vi, (7)
    #compute weighting function wi, (8)
    #compute weighted local average mi, (5)
    #compute si^2 (5)
    #compute final output (11)
    
    
    """
    #old code to iterate through the zero padded matrix
    for i in range(half_size, sizeX - half_size):
        for j in range(half_size, sizeY - half_size):
            img_filtered[i - half_size, j - half_size] = output based on filtering at img[i, j]
    """
    
    # convert back into grayscale
    img_filtered = img # temporary code
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered
    
# pulled this from previous homework and modified it a bit, but might need more work for our purposes
def gauss_kernel_generator(kernel_size: int, sigma: float) -> np.ndarray:
    # Todo: given sigma(spacial variance) and kernel size, you need to create a kernel_sizexkernel_size gaussian kernel
    # Please check out the formula in slide 15 of lecture 6 to learn how to compute the gaussian kernel weight: g[k, l] at each position [k, l].
    kernel_weights = np.zeros((kernel_size, kernel_size))
    for k in range(kernel_size):
        for l in range(kernel_size):
            kernel_weights[k][l] = math.exp(-((k-kernel_size//2)**2 + (l-kernel_size//2)**2) / (2 * (sigma**2))) # might need to add the 1/(2pisigma^2

    return kernel_weights


# main function
if __name__ == '__main__':
    # read the rgb image
    rgb_filename = '1.jpg'
    im = cv2.imread(rgb_filename)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # for display purposes
    original = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    
    output = Kuwahara_filter(gray, 8, 3, 5.0, 4.0) # all parameters will need to be tweaked
    
    # visualization for debugging
    fig = plt.figure()
    
    # show input image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(original)
    ax.set_title('input image')
    
    # show output image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap="gray")
    ax.set_title('resulting image')

    plt.show()
    cv2.imwrite('results.png', output)