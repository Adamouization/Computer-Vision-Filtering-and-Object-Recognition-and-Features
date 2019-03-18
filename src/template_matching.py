import cv2
import numpy as np
from scipy import signal

from src.convolution import gaussian_kernel, reduce_size


def subsample_image(img, pyramid_depth):
    """
    Subsamples the image multiple times by half.
    :param img: the image to subsample
    :param pyramid_depth: the number of times to subsample by half
    :return: the subsampled image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # no downsampling: return same size image
    if pyramid_depth is 0:
        return gray

    # blur the image using the gaussian kernel
    gaussian_filter = gaussian_kernel(5, 5, 15)
    blurred_img = signal.convolve2d(gray, gaussian_filter, mode="full", boundary="fill", fillvalue=255)
    blurred_img = blurred_img.astype(np.uint8)
    blurred_img_reduced = reduce_size(blurred_img, gaussian_filter)

    # initialise loop variables
    img_height = blurred_img_reduced.shape[0]
    img_width = blurred_img_reduced.shape[1]
    scaled_img = np.zeros((int(img_height / 2), int(img_width / 2)), dtype=np.uint8)  # initialise final scaled image
    prev_scaled_img = blurred_img_reduced

    # recursively downsample image by half
    for _ in range(0, pyramid_depth):
        # assume that height=width for all images
        scaled_img = np.zeros((int(img_height / 2), int(img_width / 2)), dtype=np.uint8)
        i = 0
        for m in range(0, img_height, 2):
            j = 0
            for n in range(0, img_width, 2):
                scaled_img[i, j] = prev_scaled_img[m, n]
                j += 1
            i += 1
        prev_scaled_img = scaled_img
        prev_scaled_img = signal.convolve2d(prev_scaled_img, gaussian_filter, mode="full", boundary="fill", fillvalue=255)
        prev_scaled_img = reduce_size(prev_scaled_img, gaussian_filter)
        img_height = prev_scaled_img.shape[0]
        img_width = prev_scaled_img.shape[1]

    return scaled_img
