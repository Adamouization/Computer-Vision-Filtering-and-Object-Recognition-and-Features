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
            intensity_based_template_matching_testing(testing_dataset_directory, "dataset/Training/templates")
        else:
            print("Invalid mode chosen. Choose from 'train' or 'test'")
            exit(0)

    elif settings.model_type == 'sift':
        if settings.mode == 'train':
            sift_training(training_dataset_directory)
        elif settings.mode == 'test':
            sift_testing(testing_dataset_directory)
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

    filters = {
        'gaussian_filter': gaussian_kernel(5, 5, 3),
        'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'horizontal_edge_detector': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    }

    # Gaussian Blur
    custom_conv = perform_custom_convolution(gray, filters['gaussian_filter'])
    library_conv = perform_library_convolution(gray, filters['gaussian_filter'])
    print("custom_conv:{} - library_conv:{}".format(custom_conv.shape, library_conv.shape))
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 1), plt.imshow(custom_conv)
    plt.title("Gaussian blur"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 2), plt.imshow(library_conv)
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 3), plt.imshow(difference)
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # Uniform Blur
    custom_conv = perform_custom_convolution(gray, filters['sharpen'])
    library_conv = perform_library_convolution(gray, filters['sharpen'])
    print("custom_conv:{} - library_conv:{}".format(custom_conv.shape, library_conv.shape))
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 4), plt.imshow(custom_conv)
    plt.title("Sharpen blur"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 5), plt.imshow(library_conv)
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 6), plt.imshow(difference)
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # Vertical Edge
    custom_conv = perform_custom_convolution(gray, filters['horizontal_edge_detector'])
    library_conv = perform_library_convolution(gray, filters['horizontal_edge_detector'])
    print("custom_conv:{} - library_conv:{}".format(custom_conv.shape, library_conv.shape))
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 7), plt.imshow(custom_conv)
    plt.title("Horizontal gradient"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 8), plt.imshow(library_conv)
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 9), plt.imshow(difference)
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # Identity
    custom_conv = perform_custom_convolution(gray, filters['identity'])
    library_conv = perform_library_convolution(gray, filters['identity'])
    print("custom_conv:{} - library_conv:{}".format(custom_conv.shape, library_conv.shape))
    difference = library_conv - custom_conv
    plt.subplot(4, 3, 10), plt.imshow(custom_conv)
    plt.title("Identity"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 11), plt.imshow(library_conv)
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 12), plt.imshow(difference)
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    # print("Difference={}".format(sum(library_conv) - sum(custom_conv)))
    # cv2.imshow("custom_conv", custom_conv)
    # cv2.imshow("library_conv", library_conv)
    # cv2.imshow("difference", difference)
    # cv2.waitKey(0)

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
        #normalized = normalize_image(gray)
        img = fill_black(gray)

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

    if settings.debug:
        display_template_from_binary("dataset/Training/templates/sign/rot60-sca25%.dat")


def intensity_based_template_matching_testing(directory, templates_dir):
    """
    Tests the trained model with one of the 20 test images.
    :param templates_dir: path to the templates
    :param directory: path to the testing dataset
    :return: None
    """
    testing_img = cv2.imread("dataset/Test/test_5.png", 0)
    # testing_img = normalize_image(testing_img)

    for classname in os.listdir(templates_dir):
        print(classname)

        max_rot_for_scale = {
            '6.25': {
                'rot': "",
                'val': 0
            },
            '12.5': {
                'rot': "",
                'val': 0
            },
            '25': {
                'rot': "",
                'val': 0
            },
            '50': {
                'rot': "",
                'val': 0
            }
        }

        cur_vals = {
            'class_name': " ",
            'template_name': " ",
            'min_val': 0.0,
            'max_val': 0.0,
            'min_loc': 0.0,
            'max_loc': 0.0
        }

        for t in os.listdir(templates_dir + "/" + classname + "/"):
            # print(templates_dir + template)
            template = np.load(templates_dir + "/" + classname + "/" + t)
            w, h = template.shape[::-1]

            # show images
            if settings.debug:
                cv2.imshow("from_binary", template)
                cv2.imshow("testing_img", testing_img)
                cv2.waitKey(0)

            # Apply manual template matching
            # correlation_for_loops(testing_img, template)
            # correlation_fft(testing_img, template)

            # Apply library template matching
            method = eval('cv2.TM_CCORR_NORMED')  # cv2.TM_SQDIFF_NORMED
            res = cv2.matchTemplate(testing_img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if settings.debug:
                scale = get_scaling_from_template(t)
                if scale == '6' and max_val > max_rot_for_scale['6.25']['val']:
                    max_rot_for_scale['6.25']['rot'] = get_rotation_from_template(t)
                    max_rot_for_scale['6.25']['val'] = max_val * 0.5
                elif scale == '12' and max_val > max_rot_for_scale['12.5']['val']:
                    max_rot_for_scale['12.5']['rot'] = get_rotation_from_template(t)
                    max_rot_for_scale['12.5']['val'] = max_val * 0.6
                elif scale == '25' and max_val > max_rot_for_scale['25']['val']:
                    max_rot_for_scale['25']['rot'] = get_rotation_from_template(t)
                    max_rot_for_scale['25']['val'] = max_val * 0.8
                elif scale == '50' and max_val > max_rot_for_scale['50']['val']:
                    max_rot_for_scale['50']['rot'] = get_rotation_from_template(t)
                    max_rot_for_scale['50']['val'] = max_val * 1

            # penalties for smaller images
            scale = get_scaling_from_template(t)
            if scale == '6':
                max_val = max_val * 0.6
            elif scale == '12':
                max_val = max_val * 0.7
            elif scale == '25':
                max_val = max_val * 0.7
            elif scale == '50':
                max_val = max_val * 0.7

            if max_val > cur_vals['max_val']:
                cur_vals = {
                    'class_name': classname,
                    'template_name': t,
                    'min_val': min_val,
                    'max_val': max_val,
                    'min_loc': min_loc,
                    'max_loc': max_loc,
                    'h': h,
                    'w': w,
                    'rotation': get_rotation_from_template(t)
                }

            if settings.debug:
                print("max_val={}".format(max_val))
                print("min_val={}".format(min_val))
                print("max_loc={}".format(max_loc))
                print("min_loc={}".format(min_loc))
                cv2.rectangle(testing_img, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, 2)
                plt.subplot(121), plt.imshow(res, cmap='gray')
                plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(testing_img, cmap='gray')
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                plt.suptitle('res')
                plt.show()

        print(cur_vals['max_val'])
        if cur_vals['max_val'] > 0.54:  # threshold to ignore low values
            max_loc = cur_vals['max_loc']
            width = cur_vals['w']
            height = cur_vals['h']
            rotation = cur_vals['rotation']

            top_left = max_loc
            bottom_right = (top_left[0] + width, top_left[1] + height)
            second_corner_height = find_rect_corners_with_trigonometry(rotation, height)

            # draw the rectangle
            cv2.line(
                img=testing_img,
                pt1=(second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]),
                pt2=(second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]),
                color=250,
                thickness=2
            )
            cv2.line(
                img=testing_img,
                pt1=(second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]),
                pt2=(second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]),
                color=200,
                thickness=2
            )
            cv2.line(
                img=testing_img,
                pt1=(second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]),
                pt2=(second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]),
                color=150,
                thickness=2
            )
            cv2.line(
                img=testing_img,
                pt1=(second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]),
                pt2=(second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]),
                color=100,
                thickness=2
            )

            cv2.putText(
                img=testing_img,
                text=classname,
                org=top_left,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=1,
                color=(255, 255, 255),
                lineType=2
            )

    cv2.imshow("img", testing_img)
    cv2.waitKey(0)


def sift_training(directory):
    pass
    # todo sift training


def sift_testing(directory):
    pass
    # todo sift testing


if __name__ == "__main__":
    main()
