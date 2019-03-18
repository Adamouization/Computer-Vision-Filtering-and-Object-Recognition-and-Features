import math

import numpy as np
from scipy import signal


def convolution(image, kernel):
    """
    Outputs an image of the same size as the original. Treats areas outside of the image as zero. Uses images from the
    Training dataset to test the function with a range of kernels: blurs of different sizes and edge detectors in
    different directions.

    i i       i
    = =       =
    0 1 - - - img_width - 1
    j=0    n=0      o o o . . .
    j=1    n=1      o o o . . .
    .               o o o . . .
    .               . . . . . .
    .               . . . . . .
    j=img_height-1  . . . . . .
    loop through each pixel in the image

    :param image: original image
    :param kernel: the kernel used for convolution
    :return: the convoluted image
    """
    kn = kernel

    img_height = image.shape[0]
    img_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    r = int((kernel_width - 1) / 2)
    c = int((kernel_height - 1) / 2)

    extended_img = np.pad(image, [r, c], mode="constant")
    img_height = extended_img.shape[0]
    img_width = extended_img.shape[1]

    output_img = np.zeros((img_width, img_height), dtype=np.uint8)

    for i in range(r, img_width - r):
        for j in range(c, img_height - c):
            accumulator = 0
            # DEBUG
            # print("kernel[{}, {}] = {}".format(m, n, kernel[m, n]))
            # print("extended_img[{}, {}] = {}".format(i + m, j + n, extended_img[i + m, j + n]))
            # print("accumulator = {}".format(accumulator))
            # print("\n-----\n")
            for m in range(-r-1, r):
                for n in range(-c-1, c):
                    accumulator += kernel[m + r + 1, n + c + 1] * extended_img[i + m + 1, j + n + 1]

            output_img[i, j] = accumulator

    return output_img


def perform_custom_convolution(img, kernel):
    """
    Performs convolution using the custom convolution function written in this module.
    :param img: the image to perform convolution on
    :param kernel: the kernel to convolute with
    :return: the convoluted image
    """
    conv_img = convolution(img, kernel)
    conv_img_smaller = reduce_size(conv_img, kernel)
    conv_img_smaller = reduce_size(conv_img_smaller, kernel)
    return conv_img_smaller


def perform_library_convolution(img, kernel):
    """
    Performs convolution using the library conv2 function from Numpy.
    :param img: the image to perform convolution on
    :param kernel: the kernel to convolute with
    :return: the convoluted image
    """
    library_conv = signal.convolve2d(img, kernel, mode="full", boundary="fill", fillvalue=0)
    library_conv = library_conv.astype(np.uint8)
    library_conv_smaller = reduce_size(library_conv, kernel)
    library_conv_smaller = reduce_size(library_conv_smaller, kernel)
    return library_conv_smaller


def reduce_size(image, kernel):
    """
    Reduces the image size back to its original size by deleting the added borders by the extended convolution.
    :param image: image to reduce
    :param kernel: the kernel initially used on the convoluted image
    :return: reduced image
    """
    new_im = image

    img_height = image.shape[0]
    img_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    r = int((kernel_width - 1) / 2)
    c = int((kernel_height - 1) / 2)

    # delete the right columns
    for m in range(0, r):
        new_im = np.delete(new_im, img_width - m - 1, 1)
    # delete bottom rows
    for m in range(0, c):
        new_im = np.delete(new_im, img_height - m - 1, 0)
    # delete left columns
    for m in range(0, r):
        new_im = np.delete(new_im, 0, 1)
    # delete top rows
    for m in range(0, c):
        new_im = np.delete(new_im, 0, 0)

    return new_im


def gaussian_kernel(rows, columns, dev=1):
    """
    Generates a gaussian matrix to be used to blur an image when used as a convolution filter.
    :param rows: the width of the kernel
    :param columns: the height of the kernel
    :param dev: the standard deviation
    :return: the kernel as a numpy.ndarray
    """
    output_matrix = np.zeros((rows, columns))  # initialise output kernel

    matrix_sum = 0
    r = int((rows - 1) / 2)  # used for the loops to leave out borders and to center kernel loop
    c = int((columns - 1) / 2)  # used for the loops to leave out borders and to center kernel loop

    # loop through each row of image then each column (pixel) of that row
    for i in range(-r, r + 1, 1):
        for j in range(-c, c + 1, 1):
            gaussian_value = (1 / (2 * math.pi * (dev ** 2))) * math.exp(((i ** 2) + (j ** 2)) / (2 * (dev ** 2)))
            output_matrix[i + r, j + c] = gaussian_value
            matrix_sum += gaussian_value

    return output_matrix / matrix_sum
