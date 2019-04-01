# SIFT Parameters
k = np.sqrt(2)  # base of the exponential which represents the scale coefficient
sig = 1.6       # sigma value suggested by Lowe (2004)
levels = 5      # number of different-size kernels
n_octaves = 3   # number of sampled scales

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
    gauss_at_scale.append(np.zeros((img_at_scale[n].shape[0],img_at_scale[n].shape[1],levels)))

# calculate Gaussian pyramids
for n in range(0, n_octaves):
    for i in range(0, levels):
        gauss_at_scale[n][:, :, i] = misc.imresize(
            ndimage.filters.gaussian_filter(img_at_scale[0],octave_at_scale[1][i]),1/(2**n),'bilinear')

# initialise DoG pyramids
diff_at_scale = []
for n in range(0, n_octaves):
    diff_at_scale.append(np.zeros((img_at_scale[n].shape[0],img_at_scale[n].shape[1],levels)))

# calculate DoG pyramids
for n in range(0, n_octaves):
    for i in range(0, levels-1):
        diff_at_scale[n][:,:,i] = gauss_at_scale[n][:,:,i+1]-gauss_at_scale[n][:,:,i]

# get min and max
max_min_at_scale = []
for n in range(0, n_octaves):
    max_min_at_scale.append(np.zeros((img_at_scale[n].shape[0], img_at_scale[n].shape[1], 3)))

# 3-taks loop:
# Task 1: ignore pixels where the difference of Gaussians is low
# Task 2: compute maxima and minima points in the DoG
# Task 3: reject edge cases and keep corner cases
for n in range(0, 3):  # all octaves
    # skip 0 and 4 because you don't want top and bottom layer because they only have 17 neighbours, not 26
    for i in range(1, 4):
        for j in range(1, img_at_scale[n].shape[0] - 1):
            for k in range(1, img_at_scale[n].shape[1] - 1):
                # first filtering strategy explained in section 4 of Lowe's paper
                if np.absolute(diff_at_scale[n][j, k, i]) < thr:
                    continue

                # elements of each pyramids that are larger or smaller than its 
                # 26 immediate neighbours in space and scale are labelled as extrema
                max = (diff_at_scale[n][j, k, i] > 0)
                min = (diff_at_scale[n][j, k, i] < 0)  
                # one among min and max will be true and one will be false
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

                # at this point we have all minima and maxima,
                # the below code will filter them further by using a threshold based on the eigenvalues of the Hessian matrix
                if max or min:

                    # direct application of second derivatives with help from the 2-tap filter [-1, 1] technique
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

                    # inequality used for separation of interest points referring to 
                    # edges in contrast to the ones referring to corners
                    thr_ratio = (tr_d ** 2) / det_d
                    r_ratio = ((r + 1) ** 2) / r

                    if thr_ratio > r_ratio:
                        # mark the matrix with a value to know which are the 
                        # coordinates of the interest points
                        max_min_at_scale[n][j, k, i - 1] = 1
