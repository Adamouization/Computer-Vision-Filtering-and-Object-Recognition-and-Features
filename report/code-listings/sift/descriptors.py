descriptors = np.zeros([keypoints.shape[0], 128])

# bin conversion follows proportion -> 8:36=?:ori
for i in range(0, keypoints.shape[0]):
    window = multivariate_normal(mean=[keypoints[i, 0], keypoints[i, 1]], cov=8)

    # blocks hosting the 16 histograms
    for j in range(-2, 2):
        for k in range(-2, 2):
            # if window out of image
            if keypoints[i, 0] + j * 4 < 0 or \
                    keypoints[i, 0] + (j + 1) * 4 > img_at_scale[1].shape[0] - 1 or \
                    keypoints[i, 1] + k * 4 < 0 or \
                    keypoints[i, 1] + (k + 1) * 4 > img_at_scale[1].shape[1] - 1:  
                continue

            # 8 buckets for the 8 directions 
            # (8 was also a value suggested by Lowe (2004))
            buckets = np.zeros([8, 1])

            # go through the 16 cells which make up each block
            # and fit its direction in the the relative bin of the histogram
            for m in range(0, 4):
                for n in range(0, 4):
                    x = j * 4 + m
                    y = k * 4 + n

                    # weight depends on the magnitude of each cell
                    # and on its distance from the centre of the Gaussian
                    weight = mag_at_scale[1][int(keypoints[i, 0] + x), int(keypoints[i, 1] + y), 0] * window.pdf([keypoints[i, 0] + x, keypoints[i, 1] + y])  
                    # pdf = probability density function


                    # if center of circle, i.e. j and k are reached
                    # i.e. when x and y are 0, then maximum weight 
                    # (the same way as before)
                    bucket_to_choose = int(np.clip(  # fit into buckets
                        a=np.floor(
                            ori_at_scale[1][int(keypoints[i, 0] + x), int(keypoints[i, 1] + y), 0] * 8 / 36
                        ) - 1,
                        a_min=0,
                        a_max=7
                    ))
                    buckets[bucket_to_choose] += weight

                    # setting up relative coordinate environment to simplify indices
                    j_helper = j + 2
                    k_helper = k + 2

                    # histogram number
                    current_4by4_chunck = j_helper * 4 + k_helper

                    # range for 1D interpretation of histograms
                    current_location_in_megaarray = current_4by4_chunck * 8

                    # location of exact bin (among the 128 possible bins) 
                    # in the 128-dimensional descriptor array
                    descriptors[i,current_location_in_megaarray+bucket_to_choose]+=weight