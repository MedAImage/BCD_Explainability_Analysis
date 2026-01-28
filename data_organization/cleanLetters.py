import numpy as np
import cv2
import sys
import os


if(len(sys.argv)!=2):
    print('Please, specify the path to the dataset')
    exit()

sourcePth = sys.argv[1]

def cleanLetters(sourcePath):
    for image in os.listdir(sourcePath):
        imagePath = os.path.join(sourcePath, image)
        imageBasename = os.path.basename(imagePath)
        print(imageBasename)
        
        #READING THE IMAGE AND TURNING GRAY
        imagePNG = cv2.imread(imagePath)
        grayImage = cv2.cvtColor(imagePNG,cv2.COLOR_BGR2GRAY)
        #APPLYING THRESHOLD BINARY FOR THE TEXT
        # _, binaryImage = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
        binaryImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #CREATING A MASK AND CHECKING NAMES TO TRACK THE LETTERS
        mask = np.zeros_like(binaryImage)
        height, width = binaryImage.shape
        
        #CHECKING NAME AND REGION
        if "V2" in imageBasename:
            # FOR FLIPPED IMAGES
            if "L_ML" in imageBasename or "L_CC" in imageBasename:
                #TOP-LEFT
                cv2.rectangle(mask, (0, 0), (width // 2, height // 5), 255, -1) 
            elif "R_ML" in imageBasename or "R_CC" in imageBasename:
                #TOP-RIGHT
                cv2.rectangle(mask, (width // 2, 0), (width, height // 5), 255, -1) 
        else:
            # FOR NORMAL IMAGES
            if "R_ML" in imageBasename or "R_CC" in imageBasename:
                #TOP-LEFT
                cv2.rectangle(mask, (0, 0), (width // 2, height // 5), 255, -1)  
            elif "L_ML" in imageBasename or "L_CC" in imageBasename:
                #TOP-RIGHT
                cv2.rectangle(mask, (width // 2, 0), (width, height // 5), 255, -1) 

        #APPLYING MASK AND MORPHOLOGICAL OPERATIONS TO FIND THE CONTOURNS
        binMaskedImg = cv2.bitwise_and(binaryImage, binaryImage, mask=mask)
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        dilatedImg = cv2.dilate(binMaskedImg, morphKernel, iterations=2)

        #FIND CONTOURS
        contours, _ = cv2.findContours(dilatedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #REMOVE LETTER PROCESS
        detected_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            detected_contours.append((x, y, w, h))
            #THIS ARE THE LIMITS OF THE CONTOURS
            if h > 10 and w > 10:  
                cv2.rectangle(imagePNG, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.imwrite(imagePath, imagePNG)

cleanLetters(sourcePth)