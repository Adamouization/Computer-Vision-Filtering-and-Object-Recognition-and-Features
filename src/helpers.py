import os
import re

import cv2
import numpy as np


def get_image_filenames(directory):
    """
    Returns a list containing all the mp4 files in a directory
    :param directory: the directory containing mp4 files
    :return: list of strings
    """
    list_of_images = list()
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            list_of_images.append(filename)
        else:
            print("no png files found in directory '{}'".format(directory))
    return list_of_images


def get_class_name_from_file(image):
    """
    Get the filename from the training dataset
    :param image: full filename (including extension)
    :return: string with only the filename to use as
    """
    return image[4:][:-4]


def get_rotation_from_template(template):
    """
    Retrieves the rotation value used for the specified template.
    :param template: the template
    :return: the rotation
    """
    return template.split('-')[0][3:]


def get_scaling_from_template(template):
    """
    Retrieves the scaling value used for the specified template.
    :param template: the template
    :return: the scale
    """
    return re.search("(?<=sca)[^.|%]*", template).group(0)


def get_scale_in_percentage(pyramid_depth):
    """
    Returns a string representing the percentage the image was downsampled to.
    :param pyramid_depth: the number of times to subsample by half
    :return: string in percentage
    """
    if pyramid_depth == 0:
        return "100%"
    elif pyramid_depth == 1:
        return "50%"
    elif pyramid_depth == 2:
        return "25%"
    elif pyramid_depth == 3:
        return "12.5%"
    elif pyramid_depth == 4:
        return "6.25%"
    elif pyramid_depth == 5:
        return "3.125%"
    return "0%"


def display_template_from_binary(path):
    """
    Read an image from one of the binary files
    :param template:
    :return:
    """
    from_binary = np.load(path)
    cv2.imshow("from_binary", from_binary)
    cv2.waitKey(0)
