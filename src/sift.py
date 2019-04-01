import cv2
import numpy as np
from scipy import misc, ndimage
from scipy.stats import multivariate_normal


def get_features(img, thr):
    start_img = ndimage.imread(img, flatten=True)

    # SIFT Parameters
    k = np.sqrt(2)  # base of the exponential which represents the scale coefficient
    sig = 1.6  # sigma value suggested by Lowe (2004)
    levels = 5  # number of different-size kernels
    n_octaves = 3  # number of sampled scales
    # threshold variable is the contrast threshold. Set to at least 1

    # assign different kernel sizes for different image scales and different levels. More specifically, start from sigma and k and increase k's exponent by 1 for progression within a level and increase k's exponent by 2 between scales
    all_scales = np.zeros((levels + ((n_octaves - 1) * 2)))
    octave_at_scale = np.zeros((n_octaves, levels))
    for n in range(0, n_octaves):  # all octaves
        for m in range(0, levels):  # all levels
            octave_at_scale[n][m] = sig * (k ** (2 * n + m))
            all_scales[(n_octaves - 1) * n + m] = sig * (k ** (2 * n + m))

    # create multiple image scales
    img_at_scale = []
    for n in range(0, n_octaves):
        if n == 0:
            # double image size as recommended by Lowe (2004)
            double = misc.imresize(start_img, 200, 'bilinear').astype(int)
            img_at_scale.append(double)
        else:
            img_sc = misc.imresize(img_at_scale[n - 1], 50, 'bilinear').astype(int)
            img_at_scale.append(img_sc)

    # initialise Gaussian pyramids
    gauss_at_scale = []
    for n in range(0, n_octaves):
        gauss_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], levels)))

    # calcultate Gaussian pyramids
    for n in range(0, n_octaves):
        for i in range(0, levels):
            gauss_at_scale[n][:, :, i] = misc.imresize(
                ndimage.filters.gaussian_filter(img_at_scale[0], octave_at_scale[1][i]), 1 / (2 ** n), 'bilinear')
    # initialise DoG pyramids
    diff_at_scale = []
    for n in range(0, n_octaves):
        diff_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], levels)))

    # calculate DoG pyramids
    for n in range(0, n_octaves):
        for i in range(0, levels - 1):
            diff_at_scale[n][:, :, i] = gauss_at_scale[n][:, :, i + 1] - gauss_at_scale[n][:, :, i]

    # min max
    max_min_at_scale = []
    for n in range(0, n_octaves):
        max_min_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))

    # the below loop carries out three main tasks:
    # 1) ignores pixels where the difference of Gaussians is low
    # 2) computes maxima and minima points in the DoG
    # 3) rejects edge cases and keeps corner cases

    for n in range(0, 3):  # all octaves
        print("Computing octave n={}".format(n))
        for i in range(1, 4):  # skip 0 and 4 because you dont want top and bottom layer because they only have 17 neighbours, not 26
            for j in range(1, img_at_scale[n].shape[0] - 1):
                for k in range(1, img_at_scale[n].shape[1] - 1):
                    if np.absolute(diff_at_scale[n][j, k, i]) < thr:  # first filtering strategy explained in section 4 of Lowe's paper
                        continue
                    # below, elements of each pyramids that are larger or smaller than its 26 immediate neighbors in space and scale are labeled as extrema
                    max = (diff_at_scale[n][j, k, i] > 0)
                    min = (diff_at_scale[n][j, k, i] < 0)  # one among min and max will be true and one will be false
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            for dk in range(-1, 2):
                                if di == 0 and dj == 0 and dk == 0:  # skip itself
                                    continue
                                max = max and (diff_at_scale[n][j, k, i] > diff_at_scale[n][j + dj, k + dk, i + di])
                                min = min and (diff_at_scale[n][j, k, i] < diff_at_scale[n][j + dj, k + dk, i + di])
                                if not max and not min:
                                    break
                            if not max and not min:
                                break
                        if not max and not min:
                            break
                    if max or min:  # at this point we have all minima and maxima, the below will filter them further by using a threshold based on the eigenvalues of the Hessian matrix

                        # the below are direct application of second derivatives with help from the 2-tap filter [-1, 1] technique
                        dxx = (diff_at_scale[n][j, k, i] + diff_at_scale[n][j - 2, k, i] - 2 * diff_at_scale[n][
                            j - 1, k, i])
                        dyy = (diff_at_scale[n][j, k, i] + diff_at_scale[n][j, k - 2, i] - 2 * diff_at_scale[n][
                            j, k - 1, i])
                        dxy = (diff_at_scale[n][j, k, i] - diff_at_scale[n][j, k - 1, i] - diff_at_scale[n][
                            j - 1, k, i] + diff_at_scale[n][j - 1, k - 1, i])

                        # computation of the trace following relative formulas
                        tr_d = dxx + dyy
                        # computation of the determinant following relative formulas
                        det_d = dxx * dyy - (dxy ** 2)

                        r = 10.0 # value for r suggested by Lowe (2004)
                        # inequality used for separation of interest points referring to edges in contrast to the ones referring to corners
                        thr_ratio = (tr_d ** 2) / det_d
                        r_ratio = ((r + 1) ** 2) / r

                        if thr_ratio > r_ratio:
                            # simply mark the matrix with a value just to know which are the coordinates of the interest points
                            max_min_at_scale[n][j, k, i - 1] = 1

    # list result count from each octave
    for n in range(0, n_octaves):
        print("Number of extrema in octave n={}: {}".format(n, np.sum(max_min_at_scale[n])))


    # initialisation of gradient magnitude and orientation matrices
    mag_at_scale = []
    for n in range(0, n_octaves):
        mag_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))
    ori_at_scale = []
    for n in range(0, n_octaves):
        ori_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))

    # calculate all magnitudes and orientations using simple trigonometry (i.e. Pythagoras' theorem and inverse tangent)
    for n in range(0, 3):
        for i in range(0, 3):
            for j in range(1, img_at_scale[n].shape[0] - 1):
                for k in range(1, img_at_scale[n].shape[1] - 1):
                    mag_at_scale[n][j, k, i] = (((img_at_scale[n][j + 1, k] - img_at_scale[n][j - 1, k]) ** 2) + (
                                (img_at_scale[n][j, k + 1] - img_at_scale[n][j, k - 1]) ** 2)) ** 0.5
                    ori_at_scale[n][j, k, i] = (36 / (2 * np.pi)) * (
                                np.pi + np.arctan2((img_at_scale[n][j, k + 1] - img_at_scale[n][j, k - 1]),
                                                   (img_at_scale[n][j + 1, k] - img_at_scale[n][j - 1, k])))  # converted from radians to degrees, but divided by 10 (will be useful later)

    # getting the length of the keypoint matrix
    key_sum = int(np.sum(max_min_at_scale[0]) + np.sum(max_min_at_scale[1]) + np.sum(max_min_at_scale[2]))
    # initialising keypoint matrix
    keypoints = np.zeros((key_sum, 4))

    # the below loop assigns orientations and magnitudes to the selected keypoints
    key_count = 0
    for n in range(0, 3):  # all octaves
        for i in range(0, 3):  # all levels
            # through every pixel of the image
            for j in range(1, img_at_scale[n].shape[0] - 1):
                for k in range(1, img_at_scale[n].shape[1] - 1):
                    if max_min_at_scale[n][j, k, i] == 1:  # pick only pre-selected and filtered maxima and minima
                        window = multivariate_normal(mean=[j, k], cov=(1.5 * all_scales[i + n * 2]))  # 1.5 times all_scales[i+n*2] as suggested by Lowe (2004)
                        win_radius = np.floor(1.5 * all_scales[i + n * 2])
                        buckets = np.zeros([36, 1])  #this is the reason earlier we converted radians to this different type of degree
                        for x in range(int(-1 * win_radius * 2), int(win_radius * 2) + 1):  # for each bin, compute weight inside it, keeping in mind that the magnitudes close to the kypoint of interest are more important
                            square_to_circle_shape = int((((win_radius * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)  # circle equation
                            for y in range(-1 * square_to_circle_shape, square_to_circle_shape + 1):  # from 0 when x=-18, r
                                if j + x < 0 or j + x > img_at_scale[n].shape[0] - 1 or k + y < 0 or k + y > img_at_scale[n].shape[1] - 1:  # if window out of image
                                    continue
                                # if center of circle, i.e. j and k are reached, i.e. when x and y are 0, then maximum weight is reached, for borders, weight is smaller
                                weight = mag_at_scale[n][j + x, k + y, i] * window.pdf([j + x, k + y])  # probability density function
                                bucket_to_choose = int(np.clip(np.floor(ori_at_scale[n][j + x, k + y, i]) - 1, 0, 35))  # refactor float to an int within bounded range
                                buckets[bucket_to_choose] += weight  # add the weight to the selected bin

                        loc_of_max = np.argmax(buckets)
                        scale_factor = (2 ** (n - 1))  # matches scale of image (will be normalised for calculation of descriptors)
                        keypoints[key_count, :] = np.array([int(j * scale_factor), int(k * scale_factor), all_scales[i + n * 2], loc_of_max])  #add orientation and scale
                        key_count += 1

    descriptors = np.zeros([keypoints.shape[0], 128])

    # bin conversion follows proportion -> 8:36=?:ori
    for i in range(0, keypoints.shape[0]):
        window = multivariate_normal(mean=[keypoints[i, 0], keypoints[i, 1]], cov=(8))
        # the below are the blocks hosting the 16 histograms
        for j in range(-2, 2):
            for k in range(-2, 2):
                if keypoints[i, 0] + j * 4 < 0 or keypoints[i, 0] + (j + 1) * 4 > img_at_scale[1].shape[0] - 1 or keypoints[i, 1] + k * 4 < 0 or keypoints[i, 1] + (k + 1) * 4 > img_at_scale[1].shape[1] - 1:  # if window out of image
                    continue
                buckets = np.zeros([8, 1])  # 8 buckets for the 8 directions (8 was also a value suggested by Lowe (2004))
                # the below goes through the 16 cells which make up each block and fits its direction in the the relative bin of the histogram
                for m in range(0, 4):
                    for n in range(0, 4):
                        x = j * 4 + m
                        y = k * 4 + n
                        # weight depends on the magnitude of each cell and on its distance from the centre of the Gaussian
                        weight = mag_at_scale[1][int(keypoints[i, 0] + x), int(keypoints[i, 1] + y), 0] * window.pdf([keypoints[i, 0] + x, keypoints[i, 1] + y])  # probability density function
                        # if center of circle, i.e. j and k are reached, i.e. when x and y are 0, then maximum weight (the same way as before)
                        bucket_to_choose = int(np.clip(np.floor(ori_at_scale[1][int(keypoints[i, 0] + x), int(keypoints[i, 1] + y), 0] * 8 / 36) - 1, 0, 7))  # fit into buckets
                        buckets[bucket_to_choose] += weight
                        # setting up relative coordinate environment to simplify indices
                        j_helper = j + 2
                        k_helper = k + 2
                        current_4by4_chunck = j_helper * 4 + k_helper  # histogram number
                        current_location_in_megaarray = (current_4by4_chunck) * 8  # range for 1D interpretation of histograms
                        descriptors[i, current_location_in_megaarray + bucket_to_choose] += weight  # location of exact bin (among the 128 possible bins) in the 128-dimensional descriptor array

    return [keypoints, descriptors]


def draw_squares(img, keypoints, descriptors):
    testing_img = img
    for i in range(0, keypoints.shape[0]):
        x = int(keypoints[i][1])
        y = int(keypoints[i][0])
        scale = int(keypoints[i][2])
        rotation = keypoints[i][3] * 10
        alpha = np.deg2rad(rotation)
        cv2.circle(testing_img, (x, y), scale, (0, 255, 0), thickness=6, lineType=8, shift=0)
        cv2.line(img=testing_img, pt1=(x, y), pt2=(int(x + (np.cos(alpha) * scale * 3)), int(y + (np.sin(alpha) * scale * 3))), color=(0, 0, 255), thickness=4)
    cv2.imshow("img", testing_img)
    cv2.waitKey(0)
    print("hello")
