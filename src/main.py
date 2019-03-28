import argparse
import pickle
import time

from matplotlib import pyplot as plt

import src.config as settings
from src.convolution import *
from src.helpers import *
from src.sift import *
from src.sift import *
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
    print("Gaussian Blur")
    custom_conv = perform_custom_convolution(gray, filters['gaussian_filter'])
    library_conv = perform_library_convolution(gray, filters['gaussian_filter'])
    difference = library_conv - custom_conv
    is_different = np.all(difference == 0)
    plt.subplot(4, 3, 1), plt.imshow(custom_conv, cmap='gray')
    plt.title("Gaussian blur"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 2), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 3), plt.imshow(difference, cmap='gray')
    plt.title('is identical: {}'.format(is_different)), plt.xticks([]), plt.yticks([])

    # Uniform Blur
    print("Uniform Blur")
    custom_conv = perform_custom_convolution(gray, filters['sharpen'])
    library_conv = perform_library_convolution(gray, filters['sharpen'])
    difference = library_conv - custom_conv
    is_different = np.all(difference == 0)
    plt.subplot(4, 3, 4), plt.imshow(custom_conv, cmap='gray')
    plt.title("Sharpen blur"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 5), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 6), plt.imshow(difference, cmap='gray')
    plt.title('is identical: {}'.format(is_different)), plt.xticks([]), plt.yticks([])

    # Horizontal Edge
    print("Horizontal Gradient")
    custom_conv = perform_custom_convolution(gray, filters['horizontal_edge_detector'])
    library_conv = perform_library_convolution(gray, filters['horizontal_edge_detector'])
    difference = library_conv - custom_conv
    is_different = np.all(difference == 0)
    plt.subplot(4, 3, 7), plt.imshow(custom_conv, cmap='gray')
    plt.title("Horizontal gradient"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 8), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 9), plt.imshow(difference, cmap='gray')
    plt.title('is identical: {}'.format(is_different)), plt.xticks([]), plt.yticks([])

    # Identity
    print("Identity")
    custom_conv = perform_custom_convolution(gray, filters['identity'])
    library_conv = perform_library_convolution(gray, filters['identity'])
    difference = library_conv - custom_conv
    is_different = np.all(difference == 0)
    plt.subplot(4, 3, 10), plt.imshow(custom_conv, cmap='gray')
    plt.title("Identity"), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 11), plt.imshow(library_conv, cmap='gray')
    plt.title('Library conv2d'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 12), plt.imshow(difference, cmap='gray')
    plt.title('is identical: {}'.format(is_different)), plt.xticks([]), plt.yticks([])

    if settings.debug:
        print("Difference={}".format(sum(library_conv) - sum(custom_conv)))
        cv2.imshow("custom_conv", custom_conv)
        cv2.imshow("library_conv", library_conv)
        cv2.imshow("difference", difference)
        cv2.waitKey(0)

    plt.show()


def intensity_based_template_matching_training(directory):
    """
    Trains the intensity-based model with the 50 images from the dataset.
    :param directory: path to the training dataset
    :return: None
    """
    # calculate gaussian filter once
    gaussian_filter = gaussian_kernel(5, 5, 15)

    for image in get_image_filenames(directory + "png/"):
        classname = get_class_name_from_file(image)

        # read the image from file and replace white background pixels with black pixels
        img = cv2.imread(directory + "png/" + image)
        img = fill_black(img)

        # split image into 3 images for R-G-B
        img_b, img_g, img_r = cv2.split(img)

        # start generating the pyramid of templates
        for p in scaling_pyramid_depths:
            # scale the image by subsampling it p times
            scaled_img_b = subsample_image(img_b, p, gaussian_filter)
            scaled_img_g = subsample_image(img_g, p, gaussian_filter)
            scaled_img_r = subsample_image(img_r, p, gaussian_filter)

            # merge back the 3 R-G-B channels into one image
            scaled_img = cv2.merge((scaled_img_b, scaled_img_g, scaled_img_r))

            # rotate the scaled image by r degrees and normalize it
            for r in rotations:
                rotated_scaled_img = ndimage.rotate(scaled_img, r)
                rotated_scaled_norm_img = normalize_image(rotated_scaled_img)

                # check if directory exists, if it doesn't create it
                if not os.path.exists("{}/templates/{}".format(directory, classname)):
                    os.makedirs("{}/templates/{}".format(directory, classname))

                # write to binary file
                file = open(
                    "{}/templates/{}/rot{}-sca{}.dat".format(directory, classname, int(r), get_scale_in_percentage(p)),
                    'wb'
                )
                pickle.dump(rotated_scaled_norm_img, file)
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
    template_tracker = 0
    number_of_boxes_drawn = 50  # used to calculate the number of false positives and false negatives

    # read and normalise test image
    testing_img = cv2.imread("{}test_5.png".format(directory))
    testing_img = normalize_image(testing_img)

    # start measuring runtime
    start_time = time.time()

    for classname in os.listdir(templates_dir):
        template_tracker += 1
        print("\n{}/50".format(template_tracker))
        print("{}...".format(classname))

        cur_best_template = {
            'class_name': " ",
            'template_name': " ",
            'min_val': 0.0,
            'max_val': 0.0,
            'min_loc': 0.0,
            'max_loc': 0.0
        }

        for t in os.listdir(templates_dir + "/" + classname + "/"):
            template = np.load(templates_dir + "/" + classname + "/" + t)

            # show images
            if settings.debug:
                cv2.imshow("from_binary", template)
                cv2.imshow("testing_img", testing_img)
                cv2.waitKey(0)

            w = template.shape[0]
            h = template.shape[1]

            # Apply manual template matching
            # correlation_for_loops(testing_img, template)
            # correlation_fft(testing_img, template)

            # Apply library template matching
            method = eval('cv2.TM_CCORR_NORMED')  # cv2.TM_SQDIFF_NORMED
            correlation_results = cv2.matchTemplate(testing_img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(correlation_results)

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

            if max_val > cur_best_template['max_val']:
                cur_best_template = {
                    'class_name': classname,
                    'template_name': t,
                    'max_val': max_val,
                    'max_loc': max_loc,
                    'h': h,
                    'rotation': get_rotation_from_template(t)
                }

            # debugging
            if settings.debug:
                print("max_val={}".format(max_val))
                print("max_loc={}".format(max_loc))
                cv2.rectangle(testing_img, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, 2)
                plt.subplot(121), plt.imshow(correlation_results, cmap='gray')
                plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(testing_img, cmap='gray')
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                plt.suptitle('correlation_results')
                plt.show()

        print(cur_best_template['max_val'])
        if cur_best_template['max_val'] > 0.50:  # threshold to ignore low values
            top_left = cur_best_template['max_loc']
            rotation = cur_best_template['rotation']
            height = cur_best_template['h']
            box_corners = find_box_corners(rotation, height)

            # draw the rectangle by drawing 4 lines between the 4 corners
            cv2.line(
                img=testing_img,
                pt1=(box_corners["P1"][0] + top_left[0], box_corners["P1"][1] + top_left[1]),
                pt2=(box_corners["P2"][0] + top_left[0], box_corners["P2"][1] + top_left[1]),
                color=250,
                thickness=2
            )
            cv2.line(
                img=testing_img,
                pt1=(box_corners["P2"][0] + top_left[0], box_corners["P2"][1] + top_left[1]),
                pt2=(box_corners["P3"][0] + top_left[0], box_corners["P3"][1] + top_left[1]),
                color=200,
                thickness=2
            )
            cv2.line(
                img=testing_img,
                pt1=(box_corners["P3"][0] + top_left[0], box_corners["P3"][1] + top_left[1]),
                pt2=(box_corners["P4"][0] + top_left[0], box_corners["P4"][1] + top_left[1]),
                color=150,
                thickness=2
            )
            cv2.line(
                img=testing_img,
                pt1=(box_corners["P4"][0] + top_left[0], box_corners["P4"][1] + top_left[1]),
                pt2=(box_corners["P1"][0] + top_left[0], box_corners["P1"][1] + top_left[1]),
                color=100,
                thickness=2
            )

            # print classname above the box
            cv2.putText(
                img=testing_img,
                text=classname,
                org=top_left,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=1,
                color=(255, 255, 255),
                lineType=2
            )
        else:
            number_of_boxes_drawn -= 1

    # print runtime
    print("\n--- Runtime: {} seconds ---".format(time.time() - start_time))
    print("Number of boxes drawn = {}".format(number_of_boxes_drawn))

    # show the final image with the boxes
    cv2.imshow("img", testing_img)
    cv2.waitKey(0)


def sift_training(directory):
    (keypoints, descriptors) = get_features("/Users/andrealissak/COSE/UNI/SEMESTER 2/Computer-Vision-Coursework/src/lighthouse.png", 10)
    print(keypoints.shape[0])
    print(keypoints)

    img = cv2.imread("/Users/andrealissak/COSE/UNI/SEMESTER 2/Computer-Vision-Coursework/src/lighthouse.png")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    for i in range(0, keypoints.shape[0]):
        radius = int(keypoints[i][2])
        cv2.circle(img, (int(keypoints[i][1]), int(keypoints[i][0])), radius, (0, 255, 0), thickness=1, lineType=8, shift=0)
        #cv2.line(img=img,pt1=,pt2=,color=100,thickness=2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    '''
    '''



def sift_testing(directory):
    match_template("/Users/andrealissak/COSE/UNI/SEMESTER 2/Computer-Vision-Coursework/src/lighthouseaaa.png","/Users/andrealissak/COSE/UNI/SEMESTER 2/Computer-Vision-Coursework/src/lighthouse.png",1,10)
    # todo sift testing


if __name__ == "__main__":
    main()
