import cv2
import numpy as np

from convolution import convolution, gaussian_kernel


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


if __name__ == "__main__":
    main()
