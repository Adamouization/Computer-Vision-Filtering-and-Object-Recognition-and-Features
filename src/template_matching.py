import cv2
import numpy as np
from scipy import signal

from src.convolution import gaussian_kernel, reduce_size


def fill_black(img):
    """
    Replace white pixels (value 255) with black pixels (value 0) for a gray scale image.
    :param img: the gray scale image
    :return: the gray scale image with black pixels rather than white pixels
    """
    # loop through each pixel in the image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 255:  # look for white pixels
                img[i, j] = 0  # replace with black pixels
    return img


def normalize_image(img):
    """
    Normalizes an image.
    :param img: the image to normalize
    :return: the normalized image
    """
    normalized_img = np.zeros((img.shape[0], img.shape[1]))
    normalized_img = cv2.normalize(img, normalized_img, 0, 255, cv2.NORM_MINMAX)
    return normalized_img


def subsample_image(img, pyramid_depth):
    """
    Subsamples the gray scale image multiple times by half.
    :param img: the gray scale image to subsample
    :param pyramid_depth: the number of times to subsample by half
    :return: the subsampled image
    """
    # no downsampling: return same size image
    if pyramid_depth is 0:
        return img

    # blur the image using the gaussian kernel
    gaussian_filter = gaussian_kernel(5, 5, 15)
    blurred_img = signal.convolve2d(img, gaussian_filter, mode="full", boundary="fill", fillvalue=0)
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

        downsizing_factor = 2
        i = 0
        for m in range(0, img_height, downsizing_factor):
            j = 0
            for n in range(0, img_width, downsizing_factor):
                scaled_img[i, j] = prev_scaled_img[m, n]
                j += 1
            i += 1

        # convolute img
        prev_scaled_img = scaled_img
        prev_scaled_img = signal.convolve2d(prev_scaled_img, gaussian_filter, mode="full", boundary="fill", fillvalue=0)
        prev_scaled_img = reduce_size(prev_scaled_img, gaussian_filter)
        img_height = prev_scaled_img.shape[0]
        img_width = prev_scaled_img.shape[1]

    return scaled_img
