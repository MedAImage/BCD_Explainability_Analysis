import os
import sys
import copy
import json
# from dataset import ALL_CLASSES
import cv2


#ARGS: PATH TO IMAGES PNG || PATH TO THE JSON
if(len(sys.argv)!=3):
    print('Please, specify the path to the dataset and the output path for the joined data')
    exit()

ALL_CLASSES = {
    "Nodulo": [],
    "Distorsion_arq": [],
    "Densidad_asim_foc": [],
    "Microcalcificaciones": [],
    "Calc_tip_benig": []
}

CLASSES = ALL_CLASSES


joined_dataset_v2 = {}
jsonpath = sys.argv[1]

data_root = sys.argv[2]

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

flippedAnnotations=[]
# print(annotations)
for an in annotations:
    # print(an)
    imgpath = annotations[an]['image']
    if "V2" in imgpath:
        print(f"Saliendo: ya flippeada {imgpath}")
        continue
    
    findings = annotations[an]['label']
    new_annotation = {}

    new_find = copy.deepcopy(CLASSES)
    
    #FLIPPED IMAGE NAME CREATION
    imgBasename = os.path.basename(imgpath)
    name_wo_ext, ext = os.path.splitext(imgBasename)

    splitted = name_wo_ext.split("_")

    flipped_parts = [splitted[0], "V2"] + splitted[1:]
    flippedName = "_".join(flipped_parts) + ext

    splittedPath = imgpath.split("/")
    print(f"Original Path:{splittedPath}")
    flippedPath = os.path.join(splittedPath[0], splittedPath[1])
    print(f"New Path: {flippedPath}")
    
   
    #FLIPPED IMAGE CREATION AND SAVING

    image = cv2.imread(os.path.join(data_root,imgpath))
    if image is None:
        print(f'Image {image} not found')
        continue

    #SAVING FLIPPED IMAGE PATH
    new_path = "/".join(splittedPath[0:-1])
    flippedimage = cv2.flip(image, 1)
    flippedimageOutPath = os.path.join(new_path,flippedName)
    print(f"Path destino: {flippedimageOutPath}")
    cv2.imwrite(os.path.join(data_root, flippedimageOutPath), flippedimage)
    
    image_width = image.shape[1]
    #READING CLASSES FINDINGS
    for find, value in findings.items():
        lesion_name = find
        rois = value
        for roi in rois:
            newRoi=roi.copy() 
            newRoi[0] = image_width - (newRoi[0]+newRoi[2])
            new_find[lesion_name].append(newRoi)
    new_annotation['image'] = flippedimageOutPath
    new_annotation['label'] = new_find
    flippedAnnotations.append(new_annotation)


for a in flippedAnnotations:
    fName = a['image']
    fName = fName.split("/")
    fName = fName[-1]
    annotations[fName] = a

with open(jsonpath, 'w') as jsf:
    json.dump(annotations, jsf, indent=4)


print("Flipped Images Done")