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


def find_rect_corners_with_trigonometry(angle, img_height):
    if angle < 90:
        alpha = np.deg2rad(angle)
    else:
        if 90 <= angle < 180:
            alpha = np.deg2rad(angle-90)
        else:
            if 180 <= angle < 270:
                alpha = np.deg2rad(angle-180)
            else:
                if 270 <= angle < 360:
                    alpha = np.deg2rad(angle-270)
                else:
                    alpha = np.deg2rad(0)

    k = (np.sin(alpha)) * (np.sin(alpha))
    h = img_height

    d = (np.sqrt(h*h*k*k-h*h*k*k*k*k)+(h*k*k-h))/(2*k*k-1)
    # THIS IS THE OTHER POSSIBLE SOLUTION WITH MINUS d = -(np.sqrt(h*h*k*k-h*h*k*k*k*k)+(h*k*k-h))/(2*k*k-1)

    c = img_height - d

    # the following rotate counter-clockwise
    #
    #    0 - 90           90 - 180         180 - 270        270 - 360
    #
    #          P3               P4               P1               P2
    # P4 .-----.       P1 .-----.       P2 .-----.       P3 .-----.
    #    |     |          |     |          |     |          |     |
    #    |     |          |     |          |     |          |     |
    #    .-----. P2       .-----. P3       .-----. P4       .-----. P1
    #    P1               P2               P3               P4

    P1x = 0
    P1y = 0

    P2x = 0
    P2y = 0

    P3x = 0
    P3y = 0

    P4x = 0
    P4y = 0

    if angle < 90:
        P1x = c
        P1y = 0

        P2x = h
        P2y = c

        P3x = h - c
        P3y = h

        P4x = 0
        P4y = h - c
    else:
        if 90 <= angle < 180:
            P1x = h
            P1y = c

            P2x = h - c
            P2y = h

            P3x = 0
            P3y = h - c

            P4x = c
            P4y = 0
        else:
            if 180 <= angle < 270:
                P1x = h - c
                P1y = h

                P2x = 0
                P2y = h - c

                P3x = c
                P3y = 0

                P4x = h
                P4y = c
            else:
                if 270 <= angle < 360:
                    P1x = h - c
                    P1y = h

                    P2x = 0
                    P2y = h - c

                    P3x = c
                    P3y = 0

                    P4x = h
                    P4y = c
                else:
                    pass

    coord_dict = {
        "P1": [P1x, P1y],
        "P2": [P2x, P2y],
        "P3": [P3x, P3y],
        "P4": [P4x, P4y]
    }
    return coord_dict
