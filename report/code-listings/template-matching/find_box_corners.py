def find_box_corners(angle, img_height):
    angle = int(angle)
    if angle < 90:
        alpha = np.deg2rad(angle)
    elif 90 <= angle < 180:
        alpha = np.deg2rad(angle-90)
    elif 180 <= angle < 270:
        alpha = np.deg2rad(angle-180)
    elif 270 <= angle < 360:
        alpha = np.deg2rad(angle-270)
    else:
        alpha = np.deg2rad(0)

    k = (np.sin(alpha)) * (np.sin(alpha))
    h = img_height
    d = (np.sqrt((h**2 * k**2) - (h**2 * k**4)) + (h * k**2 - h)) / (2 * k**2 - 1)
    c = img_height - d

    p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = (0,) * 8
    if angle < 90:
        p1x = c, p1y = h  # P1
        p2x = h, p2y = h - c  # P2
        p3x = h - c, p3y = 0  # P3
        p4x = 0, p4y = c  # P4
    elif 90 <= angle < 180:
        # ... repeat ...
    elif 180 <= angle < 270:
        # ... repeat ...
    elif 270 <= angle < 360:
        # ... repeat ...

    return {
        "P1": [int(p1x), int(p1y)],
        "P2": [int(p2x), int(p2y)],
        "P3": [int(p3x), int(p3y)],
        "P4": [int(p4x), int(p4y)]
    }