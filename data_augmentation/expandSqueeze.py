import cv2
import numpy as np
import matplotlib.pyplot as plt




startindex = 0
th = 20

def expandSqueeze(orig_image, orientation, roi_x, roi_w, scaleFactor):
    print(f"Original image shape: {orig_image.shape}")

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image[image<th] = 0
    # Sumar valores de cada columna
    sum_col = np.sum(image, axis=0)
    # print(f"[DEBUG] Imagen: {image.shape}, orientación: {orientation}")
    # print(f"[DEBUG] ROI_x: {roi_x}, ROI_w: {roi_w}, scaleFactor: {scaleFactor}")

    nonzero_indices = np.where(sum_col > th)[0]
    # if orientation == "right":
    #     # Encontrar la primera columna con un valor diferente de cero
    #     nonzero_indices = np.where(sum_col > th)[0]
    # elif orientation == "left":
    #     # Encontrar la primera columna con valores cero
    #     nonzero_indices = np.where(sum_col <= th)[0]
    # else:
    #     return None  # Manejo de error para orientación incorrecta


    print(len(nonzero_indices), len(sum_col))
    if len(nonzero_indices) > 0:
        if orientation=="right":
            startindex = nonzero_indices[0]
        elif orientation=="left":
            startindex = nonzero_indices[-1]
        else:
            return None  # Manejo de error para orientación incorrecta
        expandedImage, new_x, new_w = expandImage(orig_image, startindex, orientation, roi_x, roi_w, scaleFactor)
        return expandedImage, new_x, new_w
    
    return None


def expandImage(image, startindex, orientation, roi_x, roi_w, scaleFactor):
    img_height, img_width, _ = image.shape
    # print(f"Original Width: {img_width}")  

    rightPart = image[:, startindex:]
    leftPart = image[:, :startindex]



    newImage = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    if orientation == "right":
        print("Expandiendo imagen hacia la derecha")
        print(f"[DEBUG] leftPart shape: {leftPart.shape}, rightPart shape: {rightPart.shape}")
        rightPart_scaled_width = int(rightPart.shape[1] * scaleFactor)
        resizedRight = cv2.resize(rightPart, (rightPart_scaled_width, img_height), interpolation=cv2.INTER_AREA)
        print(f"[DEBUG] leftPart shape: {leftPart.shape}, rightPart shape: {rightPart.shape}")
        new_left_width = img_width - resizedRight.shape[1]        
        resizedLeft = cv2.resize(leftPart, (new_left_width, img_height), interpolation=cv2.INTER_AREA)

        newImage[:, :new_left_width] = resizedLeft
        newImage[:, new_left_width:] = resizedRight

        new_x = int((roi_x - startindex) * scaleFactor + new_left_width)
        new_w = int(roi_w * scaleFactor)

    elif orientation == "left":
        print("Expandiendo imagen hacia la izquierda")
        print(f"[DEBUG] leftPart shape: {leftPart.shape}, rightPart shape: {rightPart.shape}")

        leftPart_scaled_width = int(leftPart.shape[1] * scaleFactor)
        resizedLeft = cv2.resize(leftPart, (leftPart_scaled_width, img_height), interpolation=cv2.INTER_AREA)
        print(f"[DEBUG] leftPart shape: {leftPart.shape}, rightPart shape: {rightPart.shape}")
        new_right_width = img_width - resizedLeft.shape[1]
        resizedRight = cv2.resize(rightPart, (new_right_width, img_height), interpolation=cv2.INTER_AREA)

        newImage[:, :leftPart_scaled_width] = resizedLeft
        newImage[:, leftPart_scaled_width:] = resizedRight

        new_x = int(roi_x * scaleFactor)
        new_w = int(roi_w * scaleFactor)

    print(f"[DEBUG] Resultado imagen: {newImage.shape}, new_x: {new_x}, new_w: {new_w}")

    return newImage, new_x, new_w
