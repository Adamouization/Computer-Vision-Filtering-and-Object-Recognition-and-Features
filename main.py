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


if __name__ == "__main__":
    main()
