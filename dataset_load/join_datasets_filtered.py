import os
import sys
import copy
import shutil
import json
from dataset import ALL_CLASSES
import pydicom
import numpy as np
import cv2

#ARGS: PATH TO IMAGES PNG || PATH TO THE JSON
if(len(sys.argv)!=2):
    print('Please, specify the path to the dataset and the output path for the joined data')
    exit()
CLASSES = ALL_CLASSES
joined_dataset_v3 = {}
jsonpath = sys.argv[1]

#SobelFilter function vertical and horizontal
def sobelVerticalFilter(image):
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobely 
def sobelHorizontalFilter(image):
    sobely = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return sobely 






if os.path.exists(jsonpath):
    with open(jsonpath, 'r') as jsf:
        try:
            annotations = json.load(jsf)
            print("Archivo JSON cargado correctamente.")
        except json.JSONDecodeError as e:
            print(f"Error en el archivo JSON: {e}")
            exit()
else:
    print(f"Error, el archivo {jsonpath} no se encuentra.")
    exit()
filteredAnnotations=[]
# print(annotations)
for an in annotations:
    # print(an)
    imgpath = annotations[an]['image']
    findings = annotations[an]['label']
    
    new_annotation = {}
    #COPYING FOR ANNOTATION STRUCTURE
    new_find = copy.deepcopy(CLASSES)
    #FLIPPED IMAGE NAME CREATION
    splittedPath = imgpath.split("/")
    filteredPath = splittedPath[0]+"/"+splittedPath[1]
    # print(splittedPath)
    imgBasename = os.path.basename(imgpath)
    splittedImgBasename = imgBasename.split("_")
    if "V2" in imgBasename:
        # continue
        filterName = splittedImgBasename[0] +"_"+"V3F"+"_"+splittedImgBasename[2]+"_"+ splittedImgBasename[3]
        
    else:
        filterName = splittedImgBasename[0] +"_"+"V3"+"_"+splittedImgBasename[1]+"_"+ splittedImgBasename[2]
        # filterName = splittedImgBasename[0] +"_"+"V3"+"_"+"Filtered"+"_"+splittedImgBasename[1]+"_"+ splittedImgBasename[2]
    
    print(imgpath)

    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("No se ha cargado la imagen")
    else:
        image_shape = image.shape
    sobelVertical = sobelVerticalFilter(image)
    sobelHorizontal = sobelHorizontalFilter(image)
    #binary = binary.astype(np.uint8)
    sobelVertical = sobelVertical.astype(np.uint8)
    sobelHorizontal = sobelHorizontal.astype(np.uint8)
    #Showing the shapes of all images
    print(f"Original shape: {image.shape}")
    # print(f"Binary shape: {binary.shape}")
    print(f"Sobel Vertical shape: {sobelVertical.shape}")
    print(f"Sobel Horizontal shape: {sobelHorizontal.shape}")
    preprocImg = cv2.merge((image, sobelVertical, sobelHorizontal))
    print(f"Stacked shape: {preprocImg.shape}")
    newfilteredPath = "/".join(splittedPath[0:-1])
    filteredimageOutPath = os.path.join(newfilteredPath,filterName)
    print(filteredimageOutPath)
    
    print(f"Saving image in:{filteredimageOutPath}")
    #SAVING FLIPPED IMAGE PATH
    new_annotation['image'] = filteredimageOutPath
    cv2.imwrite(filteredimageOutPath, preprocImg)

    #READING CLASSES FINDINGS
    for find, value in findings.items():
        lesion_name = find
        rois = value
        # print(rois)
        #READING THE LIST OF ROIS FROM EACH LESION
        for roi in rois:
            
            newRoi=roi.copy() 
            print(roi)
            print(newRoi)
            
            new_find[lesion_name].append(newRoi)
        print(new_find)
        
    new_annotation['label'] = new_find

    print(new_annotation)
    filteredAnnotations.append(new_annotation)


print("----------------------------NEW ANNOTATIONS ----------------------------")
for a in filteredAnnotations:
    fName = a['image']
    fName = fName.split("/")
    fName = fName[-1]
    annotations[fName] = a
    print("New Annotation:")
    print(annotations[fName])
    


with open(jsonpath, 'w') as jsf:
    json.dump(annotations, jsf, indent=4)


print("Filtered Images Done")
