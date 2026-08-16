import cv2
import numpy as np


startindex = 0
th = 20

def expandSqueeze(orig_image, orientation, roi_x, roi_w, scaleFactor):

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image[image<th] = 0

    sum_col = np.sum(image, axis=0)

    nonzero_indices = np.where(sum_col > th)[0]
    if len(nonzero_indices) > 0:
        if orientation=="right":
            startindex = nonzero_indices[0]
        elif orientation=="left":
            startindex = nonzero_indices[-1]
        else:
            return None  
        expandedImage, new_x, new_w = expandImage(orig_image, startindex, orientation, roi_x, roi_w, scaleFactor)
        return expandedImage, new_x, new_w
    
    return None


def expandImage(image, startindex, orientation, roi_x, roi_w, scaleFactor):
    img_height, img_width, _ = image.shape

    rightPart = image[:, startindex:]
    leftPart = image[:, :startindex]

    newImage = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    if orientation == "right":
        rightPart_scaled_width = int(rightPart.shape[1] * scaleFactor)
        resizedRight = cv2.resize(rightPart, (rightPart_scaled_width, img_height), interpolation=cv2.INTER_AREA)

        new_left_width = img_width - resizedRight.shape[1]        
        resizedLeft = cv2.resize(leftPart, (new_left_width, img_height), interpolation=cv2.INTER_AREA)

        newImage[:, :new_left_width] = resizedLeft
        newImage[:, new_left_width:] = resizedRight

        new_x = int((roi_x - startindex) * scaleFactor + new_left_width)
        new_w = int(roi_w * scaleFactor)

    elif orientation == "left":
        leftPart_scaled_width = int(leftPart.shape[1] * scaleFactor)
        resizedLeft = cv2.resize(leftPart, (leftPart_scaled_width, img_height), interpolation=cv2.INTER_AREA)

        new_right_width = img_width - resizedLeft.shape[1]
        resizedRight = cv2.resize(rightPart, (new_right_width, img_height), interpolation=cv2.INTER_AREA)

        newImage[:, :leftPart_scaled_width] = resizedLeft
        newImage[:, leftPart_scaled_width:] = resizedRight

        new_x = int(roi_x * scaleFactor)
        new_w = int(roi_w * scaleFactor)

    return newImage, new_x, new_w
