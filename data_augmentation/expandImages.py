import cv2
import expandSqueeze as es
import os
import json
from tqdm import tqdm

def expand_images(jsonpath, data_root, images_versions):
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

    new_annotations = []
    for an in tqdm(annotations, desc='Expanding images'):
        image_id = an
        data = annotations[an]

        img_path = data["image"]
        if "_EXP" in image_id:
            continue

        try:
            img = cv2.imread(os.path.join(data_root,img_path))
            if img is None:
                print(f"Error loading image {img_path}, omitting file...")
                continue

            img_basename = os.path.basename(img_path)
            name_without_ext, ext = os.path.splitext(img_basename)
            ext = ".png"

            parts = name_without_ext.split("_")

            if "_V2_" in name_without_ext:
                orientacion = parts[-1]
                lateralidad = parts[-2]
                v2 = parts[-3]
                base_id = "_".join(parts[:-3])
                expanded_name1 = f"{base_id}_{v2}_EXP1_{lateralidad}_{orientacion}"
                expanded_name0 = f"{base_id}_{v2}_EXP0_{lateralidad}_{orientacion}"
            else:
                orientacion = parts[-1]
                lateralidad = parts[-2]
                base_id = "_".join(parts[:-2])
                expanded_name1 = f"{base_id}_EXP1_{lateralidad}_{orientacion}"
                expanded_name0 = f"{base_id}_EXP0_{lateralidad}_{orientacion}"

            orig_name = '_'.join([parts[0], parts[-2], parts[-1]])

            if expanded_name0 in images_versions[orig_name]:
                add_annotation_0 = False
            else:
                add_annotation_0 = True

            if expanded_name1 in images_versions[orig_name]:
                add_annotation_1 = False
            else:
                add_annotation_1 = True

            expanded_name0 = expanded_name0 + ext
            expanded_name1 = expanded_name1 + ext

            original_findings = data.get("label", {})
            new_findings1 = {key: [] for key in original_findings.keys()}
            new_findings0 = {key: [] for key in original_findings.keys()}

            expanded_img1, expanded_img0 = None, None
            expanded_path1 = os.path.join(os.path.dirname(img_path), expanded_name1)
            expanded_path0 = os.path.join(os.path.dirname(img_path), expanded_name0)

            has_rois = any(original_findings.values())
            orientation = "left" if lateralidad == 'L' else "right"
            if "_V2_" in name_without_ext:
                orientation = "right" if orientation == "left" else "left"
            
            if has_rois:
                for find, value in original_findings.items():
                    for roi in value:
                        roi_x, roi_y, roi_w, roi_h = roi
                    
                        exp_img1, new_x1, new_w1 = es.expandSqueeze(
                            img, orientation , roi_x, roi_w, 1.25)
                        
                        exp_img0, new_x0, new_w0 = es.expandSqueeze(
                            img, orientation , roi_x, roi_w, 0.8)
                        
                        if exp_img1 is not None:
                            expanded_img1 = exp_img1
                            new_findings1[find].append([new_x1, roi_y, new_w1, roi_h])
                        
                        if exp_img0 is not None:
                            expanded_img0 = exp_img0
                            new_findings0[find].append([new_x0, roi_y, new_w0, roi_h])
            else:
                expanded_img1, _, _ = es.expandSqueeze(img, orientation, 0, img.shape[1], 1.25)
                expanded_img0, _, _ = es.expandSqueeze(img, orientation, 0, img.shape[1], 0.8)

            if expanded_img1 is not None:
                cv2.imwrite(os.path.join(data_root,expanded_path1), expanded_img1)
                if add_annotation_1:
                    new_annotations.append({
                        "image": expanded_path1,
                        "label": new_findings1})

            if expanded_img0 is not None:
                cv2.imwrite(os.path.join(data_root, expanded_path0), expanded_img0)
                if add_annotation_0:                
                    new_annotations.append({
                        "image": expanded_path0,
                        "label": new_findings0})

        except Exception as e:
            print(f"Error processing {img_path}: {e}, omitting file...")

    print("Expanded Images Done")
    if new_annotations:
        for new_annot in new_annotations:
            annotations[os.path.basename(new_annot["image"])] = new_annot

        with open(jsonpath, 'w') as jsf:
            json.dump(annotations, jsf, indent=4)
        return True
    return False

