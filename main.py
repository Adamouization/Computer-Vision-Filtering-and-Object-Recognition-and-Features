import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Program entry point.
    :return: None
    """
    image_directory = "dataset/Training/png/"
    image_name = "001-lighthouse.png"

    gauss = gaussian_kernel(5, 5, 1)
    print(gauss)

    img = cv2.imread(image_directory + image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Gray image", gray)
    # cv2.waitKey(0)

    kernel = np.ones((3, 3))/9  # blur kernel
    conv_img = convolution(gray, kernel)

    cv2.imshow("Convoluted image", conv_img)
    cv2.waitKey(0)


def convolution(image, kernel):
    """
    Outputs an image of the same size as the original. Treats areas outside of the image as zero. Uses images from the
    Training dataset to test the function with a range of kernels: blurs of different sizes and edge detectors in
    different directions.
    :param image: original image
    :param kernel: the kernel used for convolution
    :return: the convoluted image
    """
    img_height = image.shape[0]
    img_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    print("img_height:{}\nimg_width: {}\nkernel_height: {}\nkernel_width: {}".format(
        img_height, img_width, kernel_height, kernel_width)
    )

    convoluted_img = np.zeros((img_height, img_width))

    # loop through each pixel in the image
    for i in range(0, img_width - 1):
        for j in range(0, img_height - 1):
            accumulator = 0
            # loop through each cell in the kernel
            for m in range(kernel_width):
                for n in range(kernel_height):
                    # print("i+m={}, j+n={}".format(i+m, j+n))
                    # print("img_height-1 = {}, img_width-1={}".format(img_height-1, img_width-1))
                    if m+i <= img_height - 1 and n+j <= img_width - 1:
                        accumulator += kernel[m][n] * image[i+m][j+n]
            convoluted_img[i][j] = accumulator
    return convoluted_img


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
