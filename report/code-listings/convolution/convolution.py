def convolution(image, kernel):
    # initialise output image
    output_img = np.zeros((img_width, img_height), dtype=np.uint8)
    
    # extend the image with 0s
    extended_img = np.pad(image, [r,c], mode="constant", constant_values=0)  
    
    # flip the kernel horizontally and vertically
    kernel = np.flipud(np.fliplr(kernel)) 
    
    # perform convolution on image
    r = int((kernel_width - 1) / 2)
    c = int((kernel_height - 1) / 2)
    for i in range(r, extended_img_width-r+1):
        for j in range(c, extended_img_height-c+1):
            accumulator = 0
            for k in range(-r-1, r):
                for l in range(-c-1, c):
                    accumulator += kernel[r+k+1, c+l+1] * extended_img[i+k, j+l]
            output_img[i-r-1, j-c-1] = accumulator
    
    return output_img