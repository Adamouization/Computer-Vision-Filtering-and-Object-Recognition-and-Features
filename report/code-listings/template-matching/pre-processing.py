def fill_black(img):
    """
    Replace white pixels (value 255) with black pixels (value 0) for an RGB image.
    """
    img[img >= [255, 255, 255]] = 0
    return img
    
def normalize(img):
    normalized_img = np.zeros((img.shape[0], img.shape[1]))
    normalized_img = cv2.normalize(img, normalized_img, 0, 255, cv2.NORM_MINMAX)
    return normalized_img