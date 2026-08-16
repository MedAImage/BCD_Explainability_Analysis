import os
import json
import cv2
from tqdm import tqdm


def flip_images(jsonpath, data_root, images_versions):
    if os.path.exists(jsonpath):
        with open(jsonpath, 'r') as jsf:
            try:
                annotations = json.load(jsf)
            except json.JSONDecodeError as e:
                print(f"Error reading the JSON file: {e}")
                exit()
    else:
        print(f"Error, file {jsonpath} not found.")
        exit()

    flippedAnnotations=[]

    for an in tqdm(annotations, desc='Flipping images'):
        imgpath = annotations[an]['image']
        if "V2" in imgpath:
            continue

        name = an.replace('.dcm','')
        name = name.replace('.png','')
        name_parts = name.split('_') 
        orig_name = '_'.join([name_parts[0], name_parts[-2], name_parts[-1]])

        #FLIPPED IMAGE NAME CREATION
        imgBasename = os.path.basename(imgpath)
        name_wo_ext, ext = os.path.splitext(imgBasename)

        splitted = name_wo_ext.split("_")

        flipped_parts = [splitted[0], "V2"] + splitted[1:]
        flippedName = "_".join(flipped_parts)

        if flippedName in images_versions[orig_name]:
            add_annotation = False
        else:
            add_annotation = True

        flippedName = flippedName + ext

        image = cv2.imread(os.path.join(data_root,imgpath))
        if image is None:
            print(f"Error loading image {imgpath}, omitting file...")
            continue

        #SAVING FLIPPED IMAGE PATH
        splittedPath = imgpath.split("/")        
        new_path = "/".join(splittedPath[0:-1])
        flippedimage = cv2.flip(image, 1)
        flippedimageOutPath = os.path.join(new_path,flippedName)
        cv2.imwrite(os.path.join(data_root, flippedimageOutPath), flippedimage)

        if add_annotation:
            image_width = image.shape[1]
            findings = annotations[an]['label']
            new_annotation = {}

            new_find = {key: [] for key in findings.keys()}

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

    print("Flipped Images Done")

    if flippedAnnotations:
        for a in flippedAnnotations:
            fName = a['image']
            fName = fName.split("/")
            fName = fName[-1]
            annotations[fName] = a

        with open(jsonpath, 'w') as jsf:
            json.dump(annotations, jsf, indent=4)
        return True
    
    return False

