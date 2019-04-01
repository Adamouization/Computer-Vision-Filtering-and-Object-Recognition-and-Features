# used to calculate the number of false positives and false negatives
number_of_boxes_drawn = 50  

# read and normalise test image
testing_img = cv2.imread("{}test_3.png".format(directory))
testing_img = normalize_image(testing_img)

# start measuring runtime
start_time = time.time()

for classname in os.listdir(templates_dir):
    template_tracker += 1

    cur_best_template = {'class_name':"", 'template_name':"", 'max_val':0.0, 'max_loc':0.0}

    for t in os.listdir(templates_dir + "/" + classname + "/"):
        template = np.load(templates_dir + "/" + classname + "/" + t)  # load template

        # use OpenCV's matchTemplate to calculate correlation score
        correlation_results = cv2.matchTemplate(testing_img,template,cv2.TM_CCORR_NORMED)
        
        # retrieve maxima value and location
        _, max_val, _, max_loc = cv2.minMaxLoc(correlation_results)

        # apply penalties for smaller images
        scale = get_scaling_from_template(t)
        if scale == '6':
            max_val = max_val * 0.6
        elif scale == '12':
            max_val = max_val * 0.7
        elif scale == '25':
            max_val = max_val * 0.7
        elif scale == '50':
            max_val = max_val * 0.7

        #  keep best template only and data associated to it
        if max_val > cur_best_template['max_val']:
            cur_best_template = {
                'class_name': classname,
                'template_name': t,
                'max_val': max_val,
                'max_loc': max_loc,
                'h': h,
                'rotation': get_rotation_from_template(t)
            }
    
    threshold = 0.5  # threshold to ignore low values
    if cur_best_template['max_val'] > threshold:  
        top_left = cur_best_template['max_loc']
        rotation = cur_best_template['rotation']
        height = cur_best_template['h']
        box_corners = find_box_corners(rotation, height)

        # draw the rectangle by drawing 4 lines between the 4 corners
        cv2.line(
            img=testing_img,
            pt1=(box_corners["P1"][0]+top_left[0],box_corners["P1"][1]+top_left[1]),
            pt2=(box_corners["P2"][0]+top_left[0],box_corners["P2"][1]+top_left[1]),
            color=250,
            thickness=2
        )
        # ... repeat line drawing for the 3 other lines to draw the box...

        # print classname above the box
        cv2.putText(
            img=testing_img,
            text=classname,
            org=top_left
        )
    else:
        number_of_boxes_drawn -= 1

print("\n--- Runtime: {} seconds ---".format(time.time() - start_time))
print("Number of boxes drawn = {}".format(number_of_boxes_drawn))