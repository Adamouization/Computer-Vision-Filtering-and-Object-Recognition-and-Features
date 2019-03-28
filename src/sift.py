import itertools

import cv2
import numpy as np
from numpy.linalg import norm
import numpy.linalg
from scipy import misc, ndimage, signal
from scipy.stats import multivariate_normal


def get_features(img, thr):
    start_img = ndimage.imread(img, flatten=True)

    # SIFT Parameters
    s = 2
    k = 2 ** (1.0 / s)
    sig = 1.6
    n_of_different_sig = 5
    n_octaves = 3
    # threshold variable is the contrast threshold. Set to at least 1

    all_scales      = np.zeros((n_of_different_sig + ((n_octaves - 1) * 2)))
    octave_at_scale = np.zeros((n_octaves, n_of_different_sig))
    for n in range(0, n_octaves):#all octaves
        for m in range(0, n_of_different_sig):  # all octaves
            octave_at_scale[n][m] = sig * (k ** (2 * n + m))
            all_scales[(n_octaves - 1) * n + m] = sig * (k ** (2 * n + m))
    # Standard deviations for Gaussian smoothing
    #octave_1_scales = np.array([sig * (k ** 0), sig * (k ** 1), sig * (k ** 2), sig * (k ** 3), sig * (k ** 4)])
    #octave_2_scales = np.array([sig * (k ** 2), sig * (k ** 3), sig * (k ** 4), sig * (k ** 5), sig * (k ** 6)])
    #octave_3_scales = np.array([sig * (k ** 4), sig * (k ** 5), sig * (k ** 6), sig * (k ** 7), sig * (k ** 8)])
    #all_scales      = np.array([sig * (k ** 0), sig * (k ** 1), sig * (k ** 2), sig * (k ** 3), sig * (k ** 4), sig * (k ** 5), sig * (k ** 6), sig * (k ** 7), sig * (k ** 8)])

    print("original.shape[0]")
    print(start_img.shape[0])
    '''
    img_at_scale = np.zeros((n_octaves, ))
    for n in range(0, n_octaves):#all octaves
        if n == 0:
            double = misc.imresize(start_img, 200, 'bilinear').astype(int)
            np.append(img_at_scale, double)
            #np.append(img_at_scale,         misc.imresize(start_img, 200, 'bilinear').astype(int))#Lowe (2004) p.10 sec.3.3: image doubling leads to a more efficient implementation
        else:
            np.append(img_at_scale,
                      misc.imresize(img_at_scale[n-1], 50, 'bilinear').astype(int))#Lowe (2004) p.10 sec.3.3: image doubling leads to a more efficient implementation
    
    img_at_scale[0] = img_scale_2
    img_at_scale[1] = img_scale_1
    img_at_scale[2] = img_scale_05
    '''


    #resize
    #img_scale_2 = misc.imresize(start_img, 200, 'bilinear').astype(int)#Lowe (2004) p.10 sec.3.3: image doubling leads to a more efficient implementation
    #img_scale_1 = misc.imresize(img_scale_2, 50, 'bilinear').astype(int)
    #img_scale_05 = misc.imresize(img_scale_1, 50, 'bilinear').astype(int)

    #img_at_scale = np.array([[0,1],[0], [[0],[0]], [[0],[0]]], dtype=object)
    img_at_scale = []
    for n in range(0, n_octaves):
        if n == 0:
            double = misc.imresize(start_img, 200, 'bilinear').astype(int)
            img_at_scale.append(double)
        else:
            img_sc = misc.imresize(img_at_scale[n-1], 50, 'bilinear').astype(int)
            img_at_scale.append(img_sc)

    gauss_at_scale = []
    for n in range(0, n_octaves):
        gauss_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], n_of_different_sig)))


    # Initialize Gaussian pyramids
    #gauss_scale_2 = np.zeros((img_scale_2.shape[0], img_scale_2.shape[1], n_of_different_sig))
    #gauss_scale_1 = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], n_of_different_sig))
    #gauss_scale_05 = np.zeros((img_scale_05.shape[0], img_scale_05.shape[1], n_of_different_sig))

    print("Constructing pyramids...")

    # Construct Gaussian pyramids
    for n in range(0, n_octaves):
        for i in range(0, n_of_different_sig):
            gauss_at_scale[n][:, :, i] = misc.imresize(ndimage.filters.gaussian_filter(img_at_scale[0], octave_at_scale[1][i]), 1/(2 ** n), 'bilinear')

    #for i in range(0, n_of_different_sig):
     #   gauss_scale_2[:, :, i] = ndimage.filters.gaussian_filter              (img_scale_2, octave_at_scale[0][i])
     #   gauss_scale_1[:, :, i] = misc.imresize(ndimage.filters.gaussian_filter(img_scale_2, octave_at_scale[1][i]), 0.5, 'bilinear')
     #   gauss_scale_05[:, :, i] = misc.imresize(ndimage.filters.gaussian_filter(img_scale_2, octave_at_scale[2][i]), 0.25, 'bilinear')

    print(gauss_at_scale[0].shape[0])
    print(gauss_at_scale[1].shape[0])
    print(gauss_at_scale[2].shape[0])
    #cv2.imshow("img", gauss_at_scale[0][:,:,0])
    #cv2.waitKey(0)

    diff_at_scale = []
    for n in range(0, n_octaves):
        diff_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], n_of_different_sig)))

    # Initialize Difference-of-Gaussians (DoG) pyramids
    #diff_scale_2 = np.zeros((img_scale_2.shape[0], img_scale_2.shape[1], n_of_different_sig))
    #diff_scale_1 = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], n_of_different_sig))
    #diff_scale_05 = np.zeros((img_scale_05.shape[0], img_scale_05.shape[1], n_of_different_sig))

    for n in range(0, n_octaves):
        for i in range(0, n_of_different_sig - 1):
            diff_at_scale[n][:, :, i] = gauss_at_scale[n][:, :, i + 1] - gauss_at_scale[n][:, :, i]
    # Construct DoG pyramids
    #for i in range(0, n_of_different_sig-1):
    #    diff_scale_2[:, :, i] = gauss_scale_2[:, :, i + 1] - gauss_scale_2[:, :, i]
    #    diff_scale_1[:, :, i] = gauss_scale_1[:, :, i + 1] - gauss_scale_1[:, :, i]
    #    diff_scale_05[:, :, i] = gauss_scale_05[:, :, i + 1] - gauss_scale_05[:, :, i]

    max_min_at_scale = []
    for n in range(0, n_octaves):
        max_min_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))

    # Initialize pyramids to store extrema locations
    #max_min_scale_2 = np.zeros((img_scale_2.shape[0], img_scale_2.shape[1], 3))
    #max_min_scale_1 = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], 3))
    #max_min_scale_05 = np.zeros((img_scale_05.shape[0], img_scale_05.shape[1], 3))

    print("Starting extrema detection...")
    print("First octave")

    # In each of the following for loops, elements of each pyramids that are larger or smaller than its 26 immediate neighbors in space and scale are labeled as extrema. As explained in section 4 of Lowe's paper, these initial extrema are pruned by checking that their contrast and curvature are above certain thresholds. The thresholds used here are those suggested by Lowe.
    for n in range(0, 3):#all octaves
        print("Computing octave n={}".format(n))
        for i in range(1, 4):#skip 0 and 4 because you dont want top and bottom layer because they only have 17 neighbours, not 26
            for j in range(1, img_at_scale[n].shape[0] - 1):
                for k in range(1, img_at_scale[n].shape[1] - 1):
                    if np.absolute(diff_at_scale[n][j, k, i]) < thr:#overall threshold to skip irrelevant keypoints (e.g. black areas where all zero)
                        continue
                    maxbool = (diff_at_scale[n][j, k, i] > 0)
                    minbool = (diff_at_scale[n][j, k, i] < 0)
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            for dk in range(-1, 2):
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                maxbool = maxbool and (diff_at_scale[n][j, k, i] > diff_at_scale[n][j + dj, k + dk, i + di])
                                minbool = minbool and (diff_at_scale[n][j, k, i] < diff_at_scale[n][j + dj, k + dk, i + di])
                                if not maxbool and not minbool:
                                    break
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break
                    if maxbool or minbool:
                        max_min_at_scale[n][j, k, i - 1] = 1

    for n in range(0, n_octaves):
        print("Number of extrema in octave n={}: {}".format(n, np.sum(max_min_at_scale[n])))

    #cv2.imshow("img", max_min_at_scale[0])
    #cv2.waitKey(0)
    # Gradient magnitude and orientation for each image sample point at each scale

    mag_at_scale = []
    for n in range(0, n_octaves):
        mag_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))
    ori_at_scale = []
    for n in range(0, n_octaves):
        ori_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))

    #magpyrlvl1 = np.zeros((img_scale_2.shape[0], img_scale_2.shape[1], 3))
    #magpyrlvl2 = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], 3))
    #magpyrlvl3 = np.zeros((img_scale_05.shape[0], img_scale_05.shape[1], 3))

    #oripyrlvl1 = np.zeros((img_scale_2.shape[0], img_scale_2.shape[1], 3))
    #oripyrlvl2 = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], 3))
    #oripyrlvl3 = np.zeros((img_scale_05.shape[0], img_scale_05.shape[1], 3))

    for n in range(0, 3):#all octaves
        for i in range(0, 3):
            for j in range(1, img_at_scale[n].shape[0] - 1):
                for k in range(1, img_at_scale[n].shape[1] - 1):
                    mag_at_scale[n][j, k, i] = (((img_at_scale[n][j + 1, k] - img_at_scale[n][j - 1, k]) ** 2) + ((img_at_scale[n][j, k + 1] - img_at_scale[n][j, k - 1]) ** 2)) ** 0.5
                    ori_at_scale[n][j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((img_at_scale[n][j, k + 1] - img_at_scale[n][j, k - 1]),(img_at_scale[n][j + 1, k] - img_at_scale[n][j - 1, k])))



    key_sum = int(np.sum(max_min_at_scale[0]) + np.sum(max_min_at_scale[1]) + np.sum(max_min_at_scale[2]))
    keypoints = np.zeros((key_sum, 4))


    print("Orientations...")

    key_count = 0
    for n in range(0, 3):#all octaves
        for i in range(0, 3):
            escape = False
            for j in range(1, img_at_scale[n].shape[0] - 1):
                for k in range(1, img_at_scale[n].shape[1] - 1):
                    if max_min_at_scale[n][j, k, i] == 1:
                        window = multivariate_normal(mean=[j, k], cov=10)
                        win_dimensions = 8#np.floor(2 * 1.5 * all_scales[i])#height of matrix window
                                                                                    #
                                                                                    #  9 9  or  4 4 the latter (8x8) is example by Lowe
                                                                                    #  9 9      4 4
                                                                                    #
                        buckets = np.zeros([36, 1])
                        for x in range(int(-1 * win_dimensions * 2), int(win_dimensions * 2) + 1):#for each bin, compute weight inside it, keeping in mind that the magnitudes close to the kypoint of interest are more important
                            square_to_circle_shape = int((((win_dimensions * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
                            for y in range(-1 * square_to_circle_shape, square_to_circle_shape + 1):#from 0 when x=-18, r
                                if j + x < 0 or j + x > img_at_scale[n].shape[0] - 1 or k + y < 0 or k + y > img_at_scale[n].shape[1] - 1:#if window out of image
                                    continue
                                weight = mag_at_scale[n][j + x, k + y, i] * window.pdf([j + x, k + y])#probability density function
                                #if center of circle, i.e. j and k are reached, i.e. when x and y are 0, then macumum weight
                                bin_idx = np.clip(np.floor(mag_at_scale[n][j + x, k + y, i]), 0, 35)
                                buckets[int(np.floor(bin_idx))] += weight


                        max = np.amax(buckets)
                        loc_of_max = np.argmax(buckets)
                        #print(maxidx)
                        #print(i)
                        #print(count)
                        if key_count >= keypoints.shape[0]:#TODO add space for hist columns higher than 80%
                            escape = True
                            break
                        keypoints[key_count, :] = np.array([int(j * 0.5), int(k * 0.5), all_scales[i], loc_of_max])
                        key_count += 1
                        buckets[loc_of_max] = 0
                        curr_max = np.amax(buckets)
                        while curr_max > 0.8 * max:
                            loc_of_NEW_max = np.argmax(buckets)
                            np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), all_scales[i], loc_of_NEW_max]]), axis=0)
                            buckets[loc_of_NEW_max] = 0
                            curr_max = np.amax(buckets)

                if escape:
                    break
            if escape:
                break
        if escape:
            break

    '''
    print("Calculating descriptor...")

    magpyr = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], 12))
    oripyr = np.zeros((img_scale_1.shape[0], img_scale_1.shape[1], 12))

    for i in range(0, 3):
        magmax = np.amax(magpyrlvl1[:, :, i])
        magpyr[:, :, i] = misc.imresize(magpyrlvl1[:, :, i], (img_scale_1.shape[0], img_scale_1.shape[1]), "bilinear").astype(
            float)
        magpyr[:, :, i] = (magmax / np.amax(magpyr[:, :, i])) * magpyr[:, :, i]
        oripyr[:, :, i] = misc.imresize(oripyrlvl1[:, :, i], (img_scale_1.shape[0], img_scale_1.shape[1]), "bilinear").astype(int)
        oripyr[:, :, i] = ((36.0 / np.amax(oripyr[:, :, i])) * oripyr[:, :, i]).astype(int)

    for i in range(0, 3):
        magpyr[:, :, i + 3] = (magpyrlvl2[:, :, i]).astype(float)
        oripyr[:, :, i + 3] = (oripyrlvl2[:, :, i]).astype(int)

    for i in range(0, 3):
        magpyr[:, :, i + 6] = misc.imresize(magpyrlvl3[:, :, i], (img_scale_1.shape[0], img_scale_1.shape[1]), "bilinear").astype(
            int)
        oripyr[:, :, i + 6] = misc.imresize(oripyrlvl3[:, :, i], (img_scale_1.shape[0], img_scale_1.shape[1]), "bilinear").astype(
            int)

    for i in range(0, 3):
        magpyr[:, :, i + 9] = misc.imresize(magpyrlvl4[:, :, i], (img_scale_1.shape[0], img_scale_1.shape[1]), "bilinear").astype(
            int)
        oripyr[:, :, i + 9] = misc.imresize(oripyrlvl4[:, :, i], (img_scale_1.shape[0], img_scale_1.shape[1]), "bilinear").astype(
            int)

    descriptors = np.zeros([keypoints.shape[0], 128])

    for i in range(0, keypoints.shape[0]):
        for x in range(-8, 8):
            for y in range(-8, 8):
                theta = 10 * keypoints[i, 3] * np.pi / 180.0
                xrot = np.round((np.cos(theta) * x) - (np.sin(theta) * y))
                yrot = np.round((np.sin(theta) * x) + (np.cos(theta) * y))
                scale_idx = np.argwhere(all_scales == keypoints[i, 2])[0][0]
                x0 = keypoints[i, 0]
                y0 = keypoints[i, 1]
                gaussian_window = multivariate_normal(mean=[x0, y0], cov=8)
                weight = magpyr[int(x0 + xrot), int(y0 + yrot), int(scale_idx)] * gaussian_window.pdf([int(x0 + xrot), int(y0 + yrot)])
                angle = oripyr[int(x0 + xrot), int(y0 + yrot), int(scale_idx)] - keypoints[i, 3]
                if angle < 0:
                    angle = 36 + angle

                bin_idx = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
                descriptors[i, int(32 * int((x + 8) / 4) + 8 * int((y + 8) / 4) + bin_idx)] += weight

        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
        descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)
        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
    '''
    descriptors=0
    return [keypoints, descriptors]

