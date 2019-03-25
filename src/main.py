import argparse
import pickle

from matplotlib import pyplot as plt
from scipy import ndimage

import src.config as settings
from src.convolution import *
from src.helpers import *
from src.template_matching import *

from numpy.linalg import inv


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
    # templates_dir = "dataset/Training/templates"
    # for classname in os.listdir(templates_dir):
    #     print(classname)
    #     for template in os.listdir(templates_dir + "/" + classname + "/"):
    #         print(template)
    #         template_rotation = get_rotation_from_template(template)
    #         print(template_rotation)
    #     print("\n")

    img = cv2.imread("dataset/Test/test_4.png", 0)
    #img = normalize_image(img)

    #template = cv2.imread("dataset/Training/png/001-lighthouseTEST.png", 0)
    template = np.load("dataset/Training/templates/lighthouse/rot60-sca25%.dat")
    w, h = template.shape[::-1]





    cv2.imshow("from_binary", template)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    # Apply template Matching
    # Apply template Matching
    # Apply template Matching
    # Apply template Matching
    slide_template_over_image(img, template)
    # Apply template Matching
    # Apply template Matching
    # Apply template Matching
    # Apply template Matching

    method = eval('cv2.TM_CCORR_NORMED')

    #method = eval('cv2.TM_SQDIFF_NORMED')
    res = cv2.matchTemplate(img, template, method)  # todo implement
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # todo implement, need function that returns minlock

    print(max_val)
    print(min_val)
    print(max_loc)
    print(min_loc)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    #
    second_corner_height = find_rect_corners_with_trigonometry(60, h)
    #
    cv2.line(img, (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), 250, 6)
    cv2.line(img, (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), 200, 6)
    cv2.line(img, (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), 150, 6)
    cv2.line(img, (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), 100, 6)
    #

    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('res')
    plt.show()


def slide_template_over_image(image, template):
    img_height = image.shape[1]
    img_width = image.shape[0]
    template_height = template.shape[1]
    template_width = template.shape[0]

    r = int((template_width - 1) / 2)
    c = int((template_height - 1) / 2)

    corr_x = img_width - template_width + 1
    corr_y = img_height - template_height + 1

    end_range_x = img_width - r + 1
    end_range_y = img_height - r + 1

    hello = 0
    skip = 8
    skipT = 16
    corr = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.uint8)
    corr_float = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.float)


    a = [[8, 2], [2, 1]]
    b = [[4, 1], [2, 2]]

    c=np.divide(a, b)
    print("c")
    print(c)


    fft="yes"
    if fft == "no":
        ###  old method  ###
        ###  old method  ###
        ###  old method  ###
        ###  old method  ###
        ###  old method  ###

        corr = np.zeros((int((end_range_x-r)/skip+1), int((end_range_y-c)/skip+1)), dtype=np.uint8)
        corr_float = np.zeros((int((end_range_x-r)/skip+1), int((end_range_y-c)/skip+1)), dtype=np.float)

        for i in range(r + 1, end_range_x - 1, skip):
            for j in range(c + 1, end_range_y - 1, skip):
                numerator = 0
                denominator_left = 0
                denominator_right = 0
                for m in range(-r, r, skipT):
                    for n in range(-c, c, skipT):
                        numerator += float(image[i + m, j + n]) * float(template[m + r, n + c])
                        denominator_left += (image[i + m, j + n]) ** 2
                        denominator_right += (template[m + r, n + c]) ** 2

                denominator = np.sqrt(float(denominator_left) * float(denominator_right))

                if denominator == 0:
                    numerator = 0
                    denominator = 1
                    hello = hello + 1
                corr_float[int((i - r) / skip), int((j - c) / skip)] = float(float(numerator) / float(denominator))

        max_val = 0
        for i in range(r + 1, end_range_x - 1, skip):
            for j in range(c + 1, end_range_y - 1, skip):
                if corr_float[int((i - r) / skip), int((j - c) / skip)] > max_val:
                    max_val = corr_float[int((i - r) / skip), int((j - c) / skip)]

        for i in range(r + 1, end_range_x - 1, skip):
            for j in range(c + 1, end_range_y - 1, skip):
                corr[int((i - r) / skip), int((j - c) / skip)] = int(
                    corr_float[int((i - r) / skip), int((j - c) / skip)] / max_val * 255)

                # corr_float[int(float(i-r)/skip), int(float(j-c)/skip)] = image[i, j]

        x_max = 0
        y_max = 0
        for i in range(r + 1, end_range_x - 1, skip):
            for j in range(c + 1, end_range_y - 1, skip):
                if corr_float[int((i - r) / skip), int((j - c) / skip)] == max_val:
                    x_max = int((i - r) / skip)
                    y_max = int((j - c) / skip)

        print(x_max * skip)
        print(y_max * skip)
        w, h = template.shape[::-1]
        top_left = (int(y_max * skip), int(x_max * skip))
        ###  old method  ###
        ###  old method  ###
        ###  old method  ###
        ###  old method  ###
        ###  old method  ###

    else:
        ###     FFT    method     ###
        ###     FFT    method     ###
        ###     FFT    method     ###
        ###     FFT    method     ###
        ###     FFT    method     ###
        corr = np.zeros((int(img_width), int(img_height)), dtype=np.uint8)
        pad_x = image.shape[0] - template.shape[0]
        pad_y = image.shape[1] - template.shape[1]
        print("template {}, {}".format(template.shape[0], template.shape[1]))
        print("pad_x{}, pad_y{}".format(pad_x, pad_y))
        pad_right = np.zeros((template.shape[0], pad_y))
        pad_bot = np.zeros((pad_x, pad_y+template.shape[0]))
        template_mod=template
        template_mod = np.append(template, pad_right, axis=1)
        print("fft_template1:{}, {}".format(template_mod.shape[0], template_mod.shape[1]))
        print("pad_right.shape[0]:{}".format(pad_right.shape[0]))
        template_mod = np.append(template_mod, pad_bot, axis=0)
        print("fft_template: {}, {}".format(template_mod.shape[0], template_mod.shape[1]))
        fft_image = np.fft.fft2(image)
        fft_template = np.fft.fft2(template_mod)

        corr_float = fft_image * fft_template
        denominator_left = fft_image * fft_image
        denominator_right = fft_template * fft_template

        denominator = np.multiply(denominator_left, denominator_right)
        corr_float = np.fft.ifft2(corr_float)
        #denominator_left = np.fft.ifft2(denominator_left)
        #denominator_right = np.fft.ifft2(denominator_right)
        denominator = np.fft.ifft2(denominator)

        denominator = np.sqrt(denominator)

        #denominator =denominator+1
        corr_float = np.multiply(corr_float, 1/denominator)
        corr_float = 1/denominator

        print("FFFFFFFFF: {}, {}".format(corr_float.shape[0], corr_float.shape[1]))
        max_val = 0
        for i in range(1, corr_float.shape[0]):
            for j in range(1, corr_float.shape[1]):
                if corr_float[i, j] > max_val:
                    max_val = corr_float[i, j]


        for i in range(1, corr_float.shape[0]):
            for j in range(1, corr_float.shape[1]):
                corr[i, j] = int((corr_float[i, j]) / max_val * 255)
        x_max = 0
        y_max = 0
                #corr_float[int(float(i-r)/skip), int(float(j-c)/skip)] = image[i, j]
        for i in range(1, corr_float.shape[0]):
            for j in range(1, corr_float.shape[1]):
                if corr_float[i, j] == max_val:
                    x_max = i
                    y_max = j
        print(x_max)
        print(y_max)
        w, h = template.shape[::-1]
        top_left = (int(y_max), int(x_max))

        ###     FFT    method     ###
        ###     FFT    method     ###
        ###     FFT    method     ###
        ###     FFT    method     ###
        ###     FFT    method     ###
    








    #
    second_corner_height = find_rect_corners_with_trigonometry(60, h)
    #
    cv2.line(image, (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), 250, 6)
    cv2.line(image, (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), 200, 6)
    cv2.line(image, (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), 150, 6)
    cv2.line(image, (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), 100, 6)

    cv2.imshow("img", image)


    print("times denominator")
    print(hello)
    cv2.imshow("corr", corr)
    cv2.waitKey(0)

    '''
    
    
            print(int((i) / skip))
            print(int((j) / skip))
            print(int((i)))
            print(int((j)))
            print(image[i, j])


            print("-----")
            
            
    '''


def sift_training(directory):
    pass
    # todo sift training


def sift_testing(directory):
    pass
    # todo sift testing


if __name__ == "__main__":
    main()
