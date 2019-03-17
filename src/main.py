import os
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage, signal

from src.convolution import convolution, gaussian_kernel, reduce_size
from src.helpers import get_class_name_from_file, get_scale_in_percentage, get_video_filenames
from src.template_matching import subsample_image


scaling_pyramid_depths = [1, 2, 3, 4]
rotations = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0]


def main():
    """
    Program entry point.
    :return: None
    """
    training_dataset_directory = "dataset/Training/"
    testing_dataset_directory = "dataset/Test/"

    # Blur an image
    # blur_image(training_dataset_directory, "001-lighthouse.png")

    # Train the intensity-based template matching model
    # intensity_based_template_matching_training(training_dataset_directory)

    # Test the intensity-based template matching model
    intensity_based_template_matching_testing(testing_dataset_directory)


def blur_image(directory, image):
    blur_kernel = gaussian_kernel(5, 5, 1)  # gauss_kernel
    # blur_kernel = np.ones((7, 7)) / 49
    # blur_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # blur_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img = cv2.imread(directory + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    conv_img = convolution(gray, blur_kernel)

    cv2.imshow("Original image", gray)

    library_conv = signal.convolve2d(gray, blur_kernel, mode="full", boundary="fill", fillvalue=0)
    library_conv = library_conv.astype(np.uint8)

    conv_img_smaller = reduce_size(conv_img, blur_kernel)
    library_conv_smaller = reduce_size(library_conv, blur_kernel)
    conv_img_smaller = reduce_size(conv_img_smaller, blur_kernel)
    library_conv_smaller = reduce_size(library_conv_smaller, blur_kernel)

    # calculate difference between library convolution and our convolution function
    difference = library_conv_smaller - conv_img_smaller
    # DEBUG
    # for m in range(5,500,10):
    #     myarr = list()
    #     for n in range(5,500,10):
    #         myarr.append(diff[m,n])
    #     print(myarr)

    cv2.imshow("Convoluted image", conv_img_smaller)
    cv2.imshow("Library convolution", library_conv_smaller)
    cv2.imshow("Difference", difference)
    cv2.waitKey(0)


def intensity_based_template_matching_training(directory):
    for image in get_video_filenames(directory + "png/"):
        classname = get_class_name_from_file(image)
        img = cv2.imread(directory + "png/" + image)  # read image from file
        for p in scaling_pyramid_depths:
            # scale the image by subsampling it p times
            scaled_img = subsample_image(img, p)

            # rotate the scaled image by r degrees
            for r in rotations:
                rotated_scaled_img = ndimage.rotate(scaled_img, r)

                # check if directory exists, if it doesn't create it
                if not os.path.exists("{}/templates/{}".format(directory, classname)):
                    os.makedirs("{}/templates/{}".format(directory, classname))

                # write to binary file
                file = open(
                    "{}/templates/{}/rot{}-sca{}.dat".format(directory, classname, r, get_scale_in_percentage(p)), 'wb'
                )
                pickle.dump(rotated_scaled_img, file)
                file.close()

    # read an image from one of the binary files
    # from_binary = np.load("dataset/Training/templates/airport/rot300.0-sca6.25%.dat")
    # cv2.imshow("from_binary", from_binary)
    # cv2.waitKey(0)


def intensity_based_template_matching_testing(directory):
    images = get_video_filenames(directory)
    print("\nTraining image file names:")
    print(images)

    img = cv2.imread("dataset/Test/test_10.png", 0)
    img2 = img.copy()

    template = np.load("dataset/Training/templates/airport/rot270.0-sca50%.dat")
    w, h = template.shape[::-1]
    img = img2.copy()

    # Apply template Matching
    method = eval('cv2.TM_SQDIFF')
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('cv2.TM_SQDIFF')

    plt.show()


if __name__ == "__main__":
    main()
