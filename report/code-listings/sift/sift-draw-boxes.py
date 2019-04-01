def draw_sift_boxes_and_directions(img, keypoints):
    
    for i in range(0, keypoints.shape[0]):
        x = int(keypoints[i][1])
        y = int(keypoints[i][0])

        scale = int(keypoints[i][2])
        rotation = keypoints[i][3] * 10
        alpha = np.deg2rad(rotation)

        # draw circle
        cv2.circle(img, (x, y), scale, (0, 255, 0), thickness=6, lineType=8, shift=0)

        # draw line for direction (extend beyond circle perimeter)
        cv2.line(
            img=img,
            pt1=(x, y),
            pt2=(int(x +(np.cos(alpha)*scale*3)), int(y+(np.sin(alpha)*scale*3))),
            color=(0,0,255),
            thickness=4
        )

    return img