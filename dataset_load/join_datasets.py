import os
import sys
import copy
import shutil
import json
from dataset import ALL_CLASSES
import pydicom
import numpy as np
import cv2

if(len(sys.argv)!=3):
    print('Please, specify the path to the dataset and the output path for the joined data')
    exit()

CLASSES = ALL_CLASSES

input_path = sys.argv[1]
if not os.path.exists(input_path):
    print('The specified path does not exist')
    exit()
print('Processing path', input_path)

output_path = sys.argv[2]
if not os.path.exists(output_path):
    os.mkdir(output_path)


joined_dataset = {}
for root, dirs, files in os.walk(sys.argv[1]):
    for dir in dirs:
        patient_path = os.path.join(root,dir)
        data_files = os.listdir(patient_path)
        img_files = [im for im in data_files if im.endswith(".dcm")]
        ann_file = [an for an in data_files if an.endswith(".json")]
        print('Processing path', patient_path)
        if len(img_files)==4:
            for img in img_files:
                if not img in joined_dataset.keys():
                    orig_img_path = os.path.join(patient_path, img)
                    new_img_path = os.path.join(output_path, img)
                    #Added some security checks
                    try:
                        dicom = pydicom.dcmread(orig_img_path)
                    except Exception as e:
                        print(f"Error al leer {orig_img_path}: {e}")
                        continue

                    array = dicom.pixel_array.astype(np.float32)
                    
                    min_val = np.min(array)
                    max_val = np.max(array)

                    image = (((array - min_val)* 65535) / (max_val - min_val)).astype(np.uint16)

                    try:
                        if(dicom[0x2050,0x0020]).value == 'INVERSE':
                            image = 65535 - image
                    except KeyError:
                        pass
                    
                    outputFile = os.path.join(output_path, os.path.basename(orig_img_path).replace('.dcm', '.png'))
                    cv2.imwrite(outputFile, image)    

                    new_data = {'image': outputFile, 'label':copy.deepcopy(CLASSES)}
                    joined_dataset[img] = new_data

            json_path = os.path.join(patient_path, 'Rois.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as jsf:
                    annotations = json.load(jsf)
                for an in annotations:
                    img = annotations[an]['Nombre']
                    findings = annotations[an]['Anotacion']
                    new_label = copy.deepcopy(CLASSES)
                    for f in findings:
                        lesionType = f['DescripcionLesion']
                        if lesionType in new_label.keys():
                            if 'Rectangulo' in f.keys():
                                roi = f['Rectangulo']
                                region = [roi[0], roi[1], roi[2], roi[3]]
                            elif 'Circulo' in f.keys():
                                roi = f['Circulo']
                                region = [roi[0]-roi[2], roi[1]-roi[3], roi[2]*2, roi[3]*2]
                            else:
                                continue
                            
                            new_label[lesionType].append(region)

                    for l in new_label:
                        joined_dataset[img]['label'][l] += new_label[l]
                        # joined_dataset[img]['label'][l] = joined_dataset[img]['label'][l] or new_label[l]
        else:
            print(f"Directorios ignorados por número de imágenes: {patient_path}")
            
print(f"Total de elementos en el JSON:{len(joined_dataset)}")
with open('joined_dataset.json', 'w') as f:
    f.write(json.dumps(joined_dataset, indent = 4))
    f.close()

