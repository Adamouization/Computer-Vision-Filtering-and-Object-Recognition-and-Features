import argparse
import pickle

from matplotlib import pyplot as plt
from scipy import ndimage

import src.config as settings
from src.convolution import *
from src.helpers import *
from src.template_matching import *


scaling_pyramid_depths = [1, 2, 3, 4]
rotations = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]


def main():
    """
    Program entry point. Parses command line arguments.
    :return: None
    """
    training_dataset_directory = "dataset/Training/"
    testing_dataset_directory = "dataset/Test/"

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model",
                        required=True,
                        help="The histogram model to use. Choose from the following options: 'convolution', "
                             "'intensity' or 'sift."
                        )
    parser.add_argument("--mode",
                        required=False,
                        help="Training or testing the model. Choose from 'train' or 'test'"
                        )
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Specify whether you want to print additional logs for debugging purposes.")
    args = parser.parse_args()
    settings.model_type = args.model
    settings.mode = args.mode
    settings.debug = args.debug

    if settings.model_type == 'convolution':
        # Blur an image
        blur_image(training_dataset_directory, "png/001-lighthouse.png")

    elif settings.model_type == 'intensity':
        if settings.mode == 'train':
            # Train the intensity-based template matching model
            intensity_based_template_matching_training(training_dataset_directory)
        elif settings.mode == 'test':
            # Test the intensity-based template matching model
            intensity_based_template_matching_testing(testing_dataset_directory)
        else:
            print("Invalid mode chosen. Choose from 'train' or 'test'")
            exit(0)

    elif settings.model_type == 'sift':
        if settings.mode == 'train':
            # todo Train the sift-based template matching model
            pass
        elif settings.mode == 'test':
            # todo Test the sift-based template matching model
            pass
        else:
            print("Invalid mode chosen. Choose from 'train' or 'test'")
            exit(0)

    else:
        print("Invalid model chosen. Choose from 'blur', 'intensity' or 'sift'")
        exit(0)


def blur_image(directory, image):
    """
    Takes care of performing convolution over an image.
    :param directory:
    :param image:
    :return: None
    """
    img = cv2.imread(directory + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    gaussian_filter = gaussian_kernel(9, 9, 15)
    custom_conv = perform_custom_convolution(gray, gaussian_filter)
    library_conv = perform_library_convolution(gray, gaussian_filter)
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 1), plt.imshow(custom_conv, cmap='gray')
    plt.title("Custom gaussian blur"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 2), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 3), plt.imshow(difference, cmap='gray')
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # Uniform Blur
    uniform_blur_kernel = np.ones((7, 7)) / 49
    custom_conv = perform_custom_convolution(gray, uniform_blur_kernel)
    library_conv = perform_library_convolution(gray, uniform_blur_kernel)
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 4), plt.imshow(custom_conv, cmap='gray')
    plt.title("Custom uniform blur"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 5), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 6), plt.imshow(difference, cmap='gray')
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # Vertical Edge
    vertical_edge_detector = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    custom_conv = perform_custom_convolution(gray, vertical_edge_detector)
    library_conv = perform_library_convolution(gray, vertical_edge_detector)
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 7), plt.imshow(custom_conv, cmap='gray')
    plt.title("Custom vertical edge"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 8), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 9), plt.imshow(difference, cmap='gray')
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # Horizontal Edge
    horizontal_edge_detector = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    custom_conv = perform_custom_convolution(gray, horizontal_edge_detector)
    library_conv = perform_library_convolution(gray, horizontal_edge_detector)
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 10), plt.imshow(custom_conv, cmap='gray')
    plt.title("Custom horizontal edge"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 11), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 12), plt.imshow(difference, cmap='gray')
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    plt.show()


def intensity_based_template_matching_training(directory):
    """
    Trains the intensity-based model with the 50 images from the dataset.
    :param directory: path to the training dataset
    :return: None
    """
    for image in get_image_filenames(directory + "png/"):
        classname = get_class_name_from_file(image)

        # read the image from file, convert it to gray scale, and replace white pixels with black pixels
        img = cv2.imread(directory + "png/" + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalized = normalize_image(gray)
        img = fill_black(normalized)

        # start generating the pyramid of templates
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
                    "{}/templates/{}/rot{}-sca{}.dat".format(directory, classname, int(r), get_scale_in_percentage(p)),
                    'wb'
                )
                pickle.dump(rotated_scaled_img, file)
                file.close()
        print("Generated templates for {}".format(classname))

    # read an image from one of the binary files
    # from_binary = np.load("dataset/Training/templates/sign/rot60-sca25%.dat")
    # cv2.imshow("from_binary", from_binary)
    # cv2.waitKey(0)


def intensity_based_template_matching_testing(directory):
    """
    Tests the trained model with 20 test images.
    :param directory: path to the testing dataset
    :return: None
    """
    # todo get class name, image rotation and image height
    templates_dir = "dataset/Training/templates"
    for classname in os.listdir(templates_dir):
        print(classname)
        for template in os.listdir(templates_dir + "/" + classname + "/"):
            print(template)
            template_rotation = get_rotation_from_template(template)
            print(template_rotation)
        print("\n")

    img = cv2.imread("dataset/Test/test_10.png", 0)
    img2 = img.copy()

    template = np.load("dataset/Training/templates/airport/rot90-sca25%.dat")
    w, h = template.shape[::-1]
    img = img2.copy()

    # Apply template Matching
    method = eval('cv2.TM_SQDIFF')
    res = cv2.matchTemplate(img, template, method)  # todo implement
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # todo implement

    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    second_corner_height = find_rect_corners_with_trigonometry(90, h)

    cv2.line(img, (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), 250, 6)
    cv2.line(img, (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), 200, 6)
    cv2.line(img, (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), 150, 6)
    cv2.line(img, (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), 100, 6)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('cv2.TM_SQDIFF')
    plt.show()


if __name__ == "__main__":
    main()
