def subsample_image(img, pyramid_depth, gaussian_filter):
    downsizing_factor = 2  # scale down by 50% for each depth
    
    # blur the image using the gaussian kernel
    blurred_img = signal.convolve2d(img,gaussian_filter,mode="full",boundary="fill",fillvalue=0)
    blurred_img_reduced = reduce_size(blurred_img, gaussian_filter)

    # recursively downsample image by half
    prev_scaled_img = blurred_img_reduced
    for _ in range(0, pyramid_depth):
        scaled_img = np.zeros((int(img_height/2), int(img_width/2)), dtype=np.uint8)
        i = 0
        for m in range(0, img_height, downsizing_factor):
            j = 0
            for n in range(0, img_width, downsizing_factor):
                scaled_img[i, j] = prev_scaled_img[m, n]
                j += 1
            i += 1
        prev_scaled_img = scaled_img
        prev_scaled_img = signal.convolve2d(prev_scaled_img,gaussian_filter,mode="full",boundary="fill",fillvalue=0)
        prev_scaled_img = reduce_size(prev_scaled_img, gaussian_filter)

    return scaled_img