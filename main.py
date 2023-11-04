import cv2
import numpy as np
import matplotlib.pyplot as plt
import linear_filter
import bilateral_filter

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
    
    g_kernel = linear_filter.gauss_kernel_generator(int(r*2), sigma)
    
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

# main function
if __name__ == '__main__':
    # read the rgb image
    rgb_filename = 'image.jpg'
    im = cv2.imread(rgb_filename)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # for display purposes
    original = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # TODO: all parameters will need to be tweaked
    output = Kuwahara_filter(gray, 8, 3, 5.0, 4.0)
    
    # visualization for debugging
    fig = plt.figure()
    
    # show input image
    ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(original)
    ax.set_title('input image')
    
    # show output image
    ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(output, cmap="gray")
    ax.set_title('resulting image')

    # plt.show()
    cv2.imwrite('results_kuwahara.png', output)

    # Gaussian filtering
    reduced_image = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    kernel_size = 7
    spatial_var = 15  # sigma_s^2
    gaussian_filter = linear_filter.gauss_kernel_generator(kernel_size, spatial_var)
    # normalization term
    gaussian_filter_normalized = gaussian_filter / (np.sum(gaussian_filter) + 1e-16)
    # apply the filter to process the image: im
    gaussian_image = linear_filter.linear_local_filtering(reduced_image, gaussian_filter_normalized)
    cv2.imwrite('result_gaussian.png', gaussian_image)

    # Bilateral filtering
    spatial_variance = 30
    intensity_variance = 0.5
    kernel_size = 7
    bilateral_image = bilateral_filter.bilateral_filtering(gray, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results_bilateral.png', bilateral_image)