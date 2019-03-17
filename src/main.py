import cv2
import numpy as np
from scipy import ndimage, signal

from src.convolution import convolution, gaussian_kernel
from src.helpers import get_class_name_from_file, get_video_filenames


scaling = [512 * s / 20 for s in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
rotations = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0]


def main():
    """
    Program entry point.
    :return: None
    """
    training_dataset_directory = "dataset/Training/png/"
    testing_dataset_directory = "dataset/Test/"

    # Blur an image
    # blur_image(training_dataset_directory, "001-lighthouse.png")

    # Train the intensity-based template matching model
    intensity_based_template_matching_training(training_dataset_directory)

    # Test the intensity-based template matching model
    # intensity_based_template_matching_testing(testing_dataset_directory)


def blur_image(directory, image):
    gauss_kernel = gaussian_kernel(10, 10, 5)
    blur_kernel = np.ones((5, 5)) / 25

    img = cv2.imread(directory + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    conv_img = convolution(gray, blur_kernel)

    cv2.imshow("Original image", gray)
    cv2.imshow("Convoluted image", conv_img)

    library_conv = signal.convolve2d(gray, blur_kernel, mode="full", boundary="wrap", fillvalue=0)
    library_conv = library_conv.astype(np.uint8)

    # diff = library_conv - conv_img
    # print(diff)

    cv2.imshow("Library convolution", library_conv)
    cv2.waitKey(0)


def intensity_based_template_matching_training(directory):
    images = get_video_filenames(directory)
    print("\nTraining image file names:")
    print(images)

    for im in images:
        classname = get_class_name_from_file(im)
        print(classname)

    for s in scaling:
        for r in rotations:
            print("scaling: {}, rotation: {}".format(s, r))

    img = cv2.imread("/Users/ajaamour/Projects/Computer-Vision-Coursework/dataset/Training/png/001-lighthouse.png")
    img30 = ndimage.rotate(img, 30.0)

    img30.dump("dataset/Training/templates/test.dat")
    from_binary = np.load("dataset/Training/templates/test.dat")
    cv2.imshow("from_binary", from_binary)
    cv2.waitKey(0)


def intensity_based_template_matching_testing(directory):
    images = get_video_filenames(directory)
    print("\nTesting image file names:")
    print(images)


if __name__ == "__main__":
    main()
