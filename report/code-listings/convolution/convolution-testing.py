# filters to test convolution with
filters = {
    'gaussian_filter': gaussian_kernel(5, 5, 3),
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'horizontal_edge_detector': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
}

# testing example for Gaussian Filter (identical for other filters)
custom_conv = convolution(img, filters['gaussian_filter'])
library_conv = ndimage.convolve(img,filters['gaussian_filter'],mode='constant',cval=0)
difference = library_conv - custom_conv
is_different = np.all(difference == 0)  # boolean: True if images are identical

# ... code for plotting output seen in Figure 1 ...