import math

import cv2
import numpy as np


def main():
    """
    Program entry point.
    :return: None
    """
    image_directory = "dataset/Training/png/"
    image_name = "001-lighthouse.png"

    gauss_kernel = gaussian_kernel(10, 10, 5)
    blur_kernel = np.ones((5, 5)) / 25

    img = cv2.imread(image_directory + image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    conv_img = convolution(gray, gauss_kernel)

    cv2.imshow("Original image", gray)
    cv2.imshow("Convoluted image", conv_img)
    cv2.waitKey(0)


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
    img_height = image.shape[0]
    img_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    output_img = np.zeros((img_width, img_height), dtype=np.uint8)

    r = int((kernel_width - 1) / 2)
    c = int((kernel_height - 1) / 2)

    for i in range(r + 1, img_width - r):
        for j in range(c + 1, img_height - c):
            accumulator = 0

            for m in range(-r, r):
                for n in range(-c, c):
                    accumulator += kernel[m + r + 1, n + c + 1] * image[i + m, j + n]

            output_img[i, j] = accumulator

    return output_img


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


if __name__ == "__main__":
    main()
