import cv2
import numpy as np
from scipy import signal

from src.convolution import gaussian_kernel, reduce_size


def fill_black(img):
    """
    Replace white pixels (value 255) with black pixels (value 0) for a gray scale image.
    :param img: the gray scale image
    :return: the gray scale image with black pixels rather than white pixels
    """
    # loop through each pixel in the image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 255:  # look for white pixels
                img[i, j] = 0  # replace with black pixels
    return img


def normalize_image(img):
    """
    Normalizes an image.
    :param img: the image to normalize
    :return: the normalized image
    """
    normalized_img = np.zeros((img.shape[0], img.shape[1]))
    normalized_img = cv2.normalize(img, normalized_img, 0, 255, cv2.NORM_MINMAX)
    return normalized_img


def subsample_image(img, pyramid_depth):
    """
    Subsamples the gray scale image multiple times by half.
    :param img: the gray scale image to subsample
    :param pyramid_depth: the number of times to subsample by half
    :return: the subsampled image
    """
    # no downsampling: return same size image
    if pyramid_depth is 0:
        return img

    # blur the image using the gaussian kernel
    gaussian_filter = gaussian_kernel(5, 5, 15)
    blurred_img = signal.convolve2d(img, gaussian_filter, mode="full", boundary="fill", fillvalue=0)
    blurred_img = blurred_img.astype(np.uint8)
    blurred_img_reduced = reduce_size(blurred_img, gaussian_filter)

    # initialise loop variables
    img_height = blurred_img_reduced.shape[0]
    img_width = blurred_img_reduced.shape[1]
    scaled_img = np.zeros((int(img_height / 2), int(img_width / 2)), dtype=np.uint8)  # initialise final scaled image
    prev_scaled_img = blurred_img_reduced

    # recursively downsample image by half
    for _ in range(0, pyramid_depth):
        # assume that height=width for all images
        scaled_img = np.zeros((int(img_height / 2), int(img_width / 2)), dtype=np.uint8)

        downsizing_factor = 2
        i = 0
        for m in range(0, img_height, downsizing_factor):
            j = 0
            for n in range(0, img_width, downsizing_factor):
                scaled_img[i, j] = prev_scaled_img[m, n]
                j += 1
            i += 1

        # convolute img
        prev_scaled_img = scaled_img
        prev_scaled_img = signal.convolve2d(prev_scaled_img, gaussian_filter, mode="full", boundary="fill", fillvalue=0)
        prev_scaled_img = reduce_size(prev_scaled_img, gaussian_filter)
        img_height = prev_scaled_img.shape[0]
        img_width = prev_scaled_img.shape[1]

    return scaled_img


def find_rect_corners_with_trigonometry(angle, img_height):
    """
    Calculates the coordinates of a rotated rectangle.
    :param angle: the angle to rotate the rectangle's coordinates to
    :param img_height: the height of the image that is going to be surrounded by the rectangle
    :return: coord_dict: dictionnary with the 4 (x,y) coordinates required to draw a rectangle around an object
    """
    angle = int(angle)

    if angle < 90:
        alpha = np.deg2rad(angle)
    else:
        if 90 <= angle < 180:
            alpha = np.deg2rad(angle-90)
        else:
            if 180 <= angle < 270:
                alpha = np.deg2rad(angle-180)
            else:
                if 270 <= angle < 360:
                    alpha = np.deg2rad(angle-270)
                else:
                    alpha = np.deg2rad(0)

    k = (np.sin(alpha)) * (np.sin(alpha))
    h = img_height

    d = (np.sqrt(h*h*k*k-h*h*k*k*k*k)+(h*k*k-h))/(2*k*k-1)
    # other possible solution with -d = -(np.sqrt(h*h*k*k-h*h*k*k*k*k)+(h*k*k-h))/(2*k*k-1)

    c = img_height - d

    # the following rotate counter-clockwise
    #
    #    0 - 90           90 - 180         180 - 270        270 - 360
    #
    #          P3               P4               P1               P2
    # P4 .-----.       P1 .-----.       P2 .-----.       P3 .-----.
    #    |     |          |     |          |     |          |     |
    #    |     |          |     |          |     |          |     |
    #    .-----. P2       .-----. P3       .-----. P4       .-----. P1
    #    P1               P2               P3               P4

    p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = (0,) * 8
    if angle < 90:
        # P1
        p1x = c
        p1y = h
        # P2
        p2x = h
        p2y = h - c
        # P3
        p3x = h - c
        p3y = 0
        # P4
        p4x = 0
        p4y = c
    else:
        if 90 <= angle < 180:
            # P1
            p1x = h
            p1y = h - c
            # P2
            p2x = h - c
            p2y = 0
            # P3
            p3x = 0
            p3y = c
            # P4
            p4x = c
            p4y = h
        else:
            if 180 <= angle < 270:
                # P1
                p1x = h - c
                p1y = 0
                # P2
                p2x = 0
                p2y = c
                # P3
                p3x = c
                p3y = h
                # P4
                p4x = h
                p4y = h - c
            else:
                if 270 <= angle < 360:
                    # P1
                    p1x = 0
                    p1y = c
                    # P2
                    p2x = c
                    p2y = h
                    # P3
                    p3x = h
                    p3y = h - c
                    # P4
                    p4x = h - c
                    p4y = 0
                else:
                    pass

    coord_dict = {
        "P1": [int(p1x), int(p1y)],
        "P2": [int(p2x), int(p2y)],
        "P3": [int(p3x), int(p3y)],
        "P4": [int(p4x), int(p4y)]
    }
    return coord_dict


def correlation_for_loops(image, template):
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

    times_denom = 0
    skip = 8
    skipT = 16
    corr = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.uint8)
    corr_float = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.float)

    a = [[8, 2], [2, 1]]
    b = [[4, 1], [2, 2]]

    c = np.divide(a, b)
    print("c={}", c)

    corr = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.uint8)
    corr_float = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.float)

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
                times_denom = times_denom + 1
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

    # draw rectangles
    second_corner_height = find_rect_corners_with_trigonometry(60, h)
    cv2.line(image, (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]),
             (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), 250, 6)
    cv2.line(image, (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]),
             (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), 200, 6)
    cv2.line(image, (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]),
             (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), 150, 6)
    cv2.line(image, (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]),
             (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), 100, 6)

    print("times denominator = {}".format(times_denom))
    cv2.imshow("img", image)
    cv2.imshow("corr", corr)
    cv2.waitKey(0)


def correlation_fft(image, template):
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

    times_denom = 0
    skip = 8
    skipT = 16
    corr = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.uint8)
    corr_float = np.zeros((int((end_range_x - r) / skip + 1), int((end_range_y - c) / skip + 1)), dtype=np.float)

    a = [[8, 2], [2, 1]]
    b = [[4, 1], [2, 2]]

    c = np.divide(a, b)
    print("c={}", c)

    corr = np.zeros((int(img_width), int(img_height)), dtype=np.uint8)
    pad_x = image.shape[0] - template.shape[0]
    pad_y = image.shape[1] - template.shape[1]
    print("template {}, {}".format(template.shape[0], template.shape[1]))
    print("pad_x{}, pad_y{}".format(pad_x, pad_y))
    pad_right = np.zeros((template.shape[0], pad_y))
    pad_bot = np.zeros((pad_x, pad_y + template.shape[0]))
    template_mod = template
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

    corr_float = np.fft.ifft2(corr_float)
    denominator_left = np.fft.ifft2(denominator_left)
    denominator_right = np.fft.ifft2(denominator_right)
    # denominator = np.fft.ifft2(denominator)
    denominator = np.multiply(denominator_left, denominator_right)
    denominator = np.sqrt(denominator)

    # denominator = denominator + 1
    corr_float = np.multiply(corr_float, 1 / denominator)

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
    # corr_float[int(float(i-r)/skip), int(float(j-c)/skip)] = image[i, j]
    for i in range(1, corr_float.shape[0]):
        for j in range(1, corr_float.shape[1]):
            if corr_float[i, j] == max_val:
                x_max = i
                y_max = j
    print(x_max)
    print(y_max)
    w, h = template.shape[::-1]
    top_left = (int(y_max), int(x_max))

    # draw rectangles
    second_corner_height = find_rect_corners_with_trigonometry(60, h)
    cv2.line(image, (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]),
             (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]), 250, 6)
    cv2.line(image, (second_corner_height["P2"][0] + top_left[0], second_corner_height["P2"][1] + top_left[1]),
             (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]), 200, 6)
    cv2.line(image, (second_corner_height["P3"][0] + top_left[0], second_corner_height["P3"][1] + top_left[1]),
             (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]), 150, 6)
    cv2.line(image, (second_corner_height["P4"][0] + top_left[0], second_corner_height["P4"][1] + top_left[1]),
             (second_corner_height["P1"][0] + top_left[0], second_corner_height["P1"][1] + top_left[1]), 100, 6)

    print("times denominator = {}".format(times_denom))
    cv2.imshow("img", image)
    cv2.imshow("corr", corr)
    cv2.waitKey(0)
