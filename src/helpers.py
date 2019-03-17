import os


def get_video_filenames(directory):
    """
    Returns a list containing all the mp4 files in a directory
    :param directory: the directory containing mp4 files
    :return: list of strings
    """
    list_of_videos = list()
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            list_of_videos.append(filename)
        else:
            print("no png files found in directory '{}'".format(directory))
    return list_of_videos


def get_class_name_from_file(image):
    """
    Get the filename from the training dataset
    :param image: full filename (including extension)
    :return: string with only the filename to use as
    """
    return os.path.splitext(image)[0].split("-")[1]
