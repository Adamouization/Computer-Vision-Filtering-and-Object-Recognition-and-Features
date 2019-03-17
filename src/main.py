import os
import pickle

import cv2
import numpy as np
from scipy import ndimage, signal

from src.convolution import convolution, gaussian_kernel, reduce_size
from src.helpers import get_class_name_from_file, get_video_filenames


scaling = [512 * s / 20 for s in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
rotations = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0]


def main():
    """
    Program entry point.
    :return: None
    """
    training_dataset_directory = "dataset/Training/"
    testing_dataset_directory = "dataset/Test/"

    # Blur an image
    blur_image(training_dataset_directory, "001-lighthouse.png")

    # Train the intensity-based template matching model
    #intensity_based_template_matching_training(training_dataset_directory)

    # Test the intensity-based template matching model
    # intensity_based_template_matching_testing(testing_dataset_directory)


def blur_image(directory, image):
    blur_kernel = gaussian_kernel(9, 9, 1)  # gauss_kernel
    # blur_kernel = np.ones((9, 9)) / 81
    img = cv2.imread("/Users/andrealissak/COSE/UNI/SEMESTER 2/Computer-Vision-Coursework/dataset/Training/png/001-lighthouse.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    conv_img = convolution(gray, blur_kernel)

    cv2.imshow("Original image", gray)

    library_conv = signal.convolve2d(gray, blur_kernel, mode="full", boundary="fill", fillvalue=0)
    library_conv = library_conv.astype(np.uint8)

    conv_img_smaller = reduce_size(conv_img, blur_kernel)
    library_conv_smaller = reduce_size(library_conv, blur_kernel)
    conv_img_smaller = reduce_size(conv_img_smaller, blur_kernel)
    library_conv_smaller = reduce_size(library_conv_smaller, blur_kernel)

    diff = library_conv_smaller - conv_img_smaller

    diff_ori = conv_img - library_conv

    for m in range(0,500,10):
        myarr = list()
        for n in range(0,500,10):
            myarr.append(diff[m,n])
        print(myarr)

    cv2.imshow("Convoluted image", conv_img_smaller)
    cv2.imshow("Library convolution", library_conv_smaller)
    cv2.waitKey(0)


def intensity_based_template_matching_training(directory):
    images = get_video_filenames(directory + "png/")
    print("\nTraining image file names:")
    print(images)

    for image in images:
        classname = get_class_name_from_file(image)
        img = cv2.imread(directory + "png/" + image)  # read image from file
        # for s in scaling:
        for r in rotations:
            # rotate image by r degrees
            rotated_img = ndimage.rotate(img, r)

            # check if directory exists, if it doesn't create it
            if not os.path.exists("{}/templates/{}".format(directory, classname)):
                os.makedirs("{}/templates/{}".format(directory, classname))

            # write to binary file
            file = open("{}/templates/{}/rot{}-sca{}.dat".format(directory, classname, r, "1"), 'wb')
            pickle.dump(rotated_img, file)
            file.close()

    # read an image from one of the binary files
    from_binary = np.load("dataset/Training/templates/telephone-r90.0-s1.dat")
    cv2.imshow("from_binary", from_binary)
    cv2.waitKey(0)


def intensity_based_template_matching_testing(directory):
    images = get_video_filenames(directory)
    print("\nTesting image file names:")
    print(images)


if __name__ == "__main__":
    main()
