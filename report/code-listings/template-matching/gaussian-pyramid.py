# Gaussian Pyramid parameters
scaling_pyramid_depths = [1, 2, 3, 4]  # 50% 25% 12.5% 6.25%
rotations = [0.0,30.0,60.0,90.0,120.0,150.0,180.0,210.0,240.0,270.0,300.0,330.0]
gaussian_filter = gaussian_kernel(5, 5, 15)

for image in get_image_filenames("dataset/Training/png/"):
    img = fill_black(img)  # replace white background pixels with black pixels
    img_b, img_g, img_r = cv2.split(img)  # split image into 3 images (R-G-B channels)

    for p in scaling_pyramid_depths:
        # scale the image by subsampling it p times
        # using our custom function "subsample_image" found in Listing B.5
        scaled_img_b = subsample_image(img_b, p, gaussian_filter)
        scaled_img_g = subsample_image(img_g, p, gaussian_filter)
        scaled_img_r = subsample_image(img_r, p, gaussian_filter)

        # merge back the 3 R-G-B channels into one image
        scaled_img = cv2.merge((scaled_img_b, scaled_img_g, scaled_img_r))

        for r in rotations:
            # rotate the scaled image by r degrees and normalize it
            rotated_scaled_img = ndimage.rotate(scaled_img, r)
            rotated_scaled_norm_img = normalize_image(rotated_scaled_img)
            
            # save the rotated & scaled template to a binary .dat
            pickle.dump(rotated_scaled_norm_img, "dataset/Training/templates/{classname}/rot{r}-sca{p}.dat".format(classname, int(r), p))
