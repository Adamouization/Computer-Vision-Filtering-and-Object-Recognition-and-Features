# initialisation of gradient magnitude and orientation matrices
mag_at_scale = []
for n in range(0, n_octaves):
    mag_at_scale.append(np.zeros((img_at_scale[n].shape[0],img_at_scale[n].shape[1],3)))
ori_at_scale = []
for n in range(0, n_octaves):
    ori_at_scale.append(np.zeros((img_at_scale[n].shape[0],img_at_scale[n].shape[1],3)))

# calculate all magnitudes and orientations using simple trigonometry 
# (i.e. Pythagoras' theorem and inverse tangent)
for n in range(0, 3):
    for i in range(0, 3):
        for j in range(1, img_at_scale[n].shape[0] - 1):
            for k in range(1, img_at_scale[n].shape[1] - 1):
                mag_at_scale[n][j, k, i] = (((img_at_scale[n][j + 1, k] - img_at_scale[n][j - 1, k]) ** 2) + ((img_at_scale[n][j, k + 1] - img_at_scale[n][j, k - 1]) ** 2)) ** 0.5
                # converted from radians to degrees, 
                # but divided by 10 (will be useful later)
                ori_at_scale[n][j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((img_at_scale[n][j, k + 1] - img_at_scale[n][j, k - 1]), (img_at_scale[n][j + 1, k] - img_at_scale[n][j - 1, k])))

# getting the length of the keypoint matrix
key_sum = int(np.sum(max_min_at_scale[0]) + np.sum(max_min_at_scale[1]) + np.sum(max_min_at_scale[2]))

# initialising keypoint matrix
keypoints = np.zeros((key_sum, 4))

# the below loop assigns orientations and magnitudes to the selected keypoints
key_count = 0
for n in range(0, 3):  # all octaves
    for i in range(0, 3):  # all levels
        
        # loop through every pixel of the image
        for j in range(1, img_at_scale[n].shape[0] - 1):
            for k in range(1, img_at_scale[n].shape[1] - 1):
                
                # pick only pre-selected and filtered maxima and minima
                if max_min_at_scale[n][j, k, i] == 1:

                    # 1.5 times all_scales[i+n*2] as suggested by Lowe (2004)
                    window=multivariate_normal(mean=[j,k],cov=(1.5*all_scales[i+n*2]))
                    win_radius = np.floor(1.5 * all_scales[i + n * 2])

                    # this is the reason earlier we converted radians to this 
                    # different type of degree
                    buckets = np.zeros([36, 1])

                    # for each bin, compute weight inside it,
                    # keeping in mind that the magnitudes close to the keypoint of 
                    # interest are more important
                    for x in range(int(-1 * win_radius * 2), int(win_radius * 2) + 1):
                        # circle equation
                        square_to_circle_shape = int((((win_radius * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)

                        # from 0 when x=-18, r
                        for y in range(-1 * square_to_circle_shape, square_to_circle_shape + 1):
                            # if window out of image
                            if j + x < 0 or \
                                    j + x > img_at_scale[n].shape[0] - 1 or \
                                    k + y < 0 or \
                                    k + y > img_at_scale[n].shape[1] - 1:
                                continue

                            # if center of circle, i.e. j and k are reached, i.e. 
                            # when x and y are 0, then maximum weight is reached, 
                            # for borders, weight is smaller
                            weight = mag_at_scale[n][j + x, k + y, i] * window.pdf([j + x, k + y])  # PDF

                            # refactor float to an int within bounded range
                            bucket_to_choose = int(np.clip(np.floor(ori_at_scale[n][j + x, k + y, i]) - 1, 0, 35))

                            # add the weight to the selected bin
                            buckets[bucket_to_choose] += weight

                    loc_of_max = np.argmax(buckets)
                    # matches scale of image (will be normalised for calculation 
                    # of descriptors)
                    scale_factor = (2 ** (n - 1))
                    # add orientation and scale
                    keypoints[key_count, :] = np.array(
                        [int(j*scale_factor),int(k*scale_factor),all_scales[i+n*2],loc_of_max])
                    key_count += 1