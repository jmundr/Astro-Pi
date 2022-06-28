def calculate_beauty(file_name):
    # Read the file from the folder
    img = cv2.imread(f'realImages/{file_name}')

    # Mask a circle
    circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (img.shape[1] // 2 + 200, img.shape[0] // 2 + 50), img.shape[0] // 2 - 150, 1, -1)
    img = cv2.bitwise_and(img, img, mask=circle_mask)

    # Convert all colours to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the colour ranges in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_green = np.array([20, 20, 50])
    upper_green = np.array([85, 255, 255])
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 25, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 31])

    # Filter out every colour but the ones in the specified rangee
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Grayscale the masks
    blue_result = cv2.bitwise_and(img, img, mask=mask_blue)
    blue_grayscale = cv2.cvtColor(blue_result, cv2.COLOR_BGR2GRAY)
    white_result = cv2.bitwise_and(img, img, mask=mask_white)
    white_grayscale = cv2.cvtColor(white_result, cv2.COLOR_BGR2GRAY)
    green_result = cv2.bitwise_and(img, img, mask=mask_green)
    green_grayscale = cv2.cvtColor(green_result, cv2.COLOR_BGR2GRAY)
    black_result = cv2.bitwise_and(img, img, mask=mask_black)
    black_grayscale = cv2.cvtColor(black_result, cv2.COLOR_BGR2GRAY)

    rows, columns, _ = img.shape
    total_area = int(rows) * int(columns)

    # Calculate the amount of non-black pixels in the image (These pixels represent the colours specified in the ranges)
    blue_area = cv2.countNonZero(blue_grayscale)
    perc_blue = round(((blue_area / total_area) * 100), 3)

    white_area = cv2.countNonZero(white_grayscale)
    perc_white = round(((white_area / total_area) * 100), 3)

    green_area = cv2.countNonZero(green_grayscale)
    perc_green = round(((green_area / total_area) * 100), 3)

    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    zero_area = total_area - cv2.countNonZero(img_grayscale)
    black_area = cv2.countNonZero(black_grayscale) + zero_area
    perc_black = round(((black_area / total_area) * 100), 3)

    # Calculate the saliency of the image and how far this is away from a third
    saliency_score = calc_thirds(img)

    # Calculate the total beauty of the image
    def total_image_beauty_rating(perc_blue, perc_white, perc_green, perc_black, saliency_score):
        return 100 + 2 * (-abs(perc_blue - perc_green) - 2 * abs(
            saliency_score) + perc_blue + 2 * perc_green - 3 * perc_white - 2 * (perc_black - 57))

    total_score = total_image_beauty_rating(perc_blue, perc_white, perc_green, perc_black, saliency_score)

    # Return the total score
    return total_score
