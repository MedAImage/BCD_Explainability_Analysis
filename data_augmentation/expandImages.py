import cv2
import numpy as np
import expandSqueeze as es
import sys
import os
import json

if len(sys.argv) != 3:
    print('Usage: python expandImages.py <json_path>')
    exit()

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

new_annotations = []
contadorImagenes = 0
for image_id, data in annotations.items():
    img_path = data["image"]
    if "_EXP" in image_id:
        print(f"Skipping: {img_path}")
        continue

    print(f"Processing: {img_path}")

    try:
        img = cv2.imread(os.path.join(data_root,img_path))
        if img is None:
            print(f"Error: No se pudo cargar la imagen {img_path}, omitiendo...")
            continue
        if True: #"Nodulo" in data["label"] and data["label"]["Nodulo"]:
            contadorImagenes = contadorImagenes + 1
            img_basename = os.path.basename(img_path)
            name_without_ext, ext = os.path.splitext(img_basename)
            ext = ".png"

            # parts = name_without_ext.split("_")

            # if "_V2_" in name_without_ext:
            #     base_id, v2, lateralidad, orientacion = parts[0], parts[1], parts[2], parts[3]
            #     expanded_name1 = f"{base_id}_{v2}_EXP1_{lateralidad}_{orientacion}{ext}"
            #     expanded_name0 = f"{base_id}_{v2}_EXP0_{lateralidad}_{orientacion}{ext}"
            # else:
            #     base_id = "_".join(parts[:-2])
            #     lateralidad = parts[-2]
            #     orientacion = parts[-1]
            #     expanded_name1 = f"{base_id}_EXP1_{lateralidad}_{orientacion}{ext}"
            #     expanded_name0 = f"{base_id}_EXP0_{lateralidad}_{orientacion}{ext}"
            parts = name_without_ext.split("_")

            if "_V2_" in name_without_ext:
                # Obtener los últimos tres campos desde el final
                orientacion = parts[-1]
                lateralidad = parts[-2]
                v2 = parts[-3]
                base_id = "_".join(parts[:-3])
                expanded_name1 = f"{base_id}_{v2}_EXP1_{lateralidad}_{orientacion}{ext}"
                expanded_name0 = f"{base_id}_{v2}_EXP0_{lateralidad}_{orientacion}{ext}"
            else:
                orientacion = parts[-1]
                lateralidad = parts[-2]
                base_id = "_".join(parts[:-2])
                expanded_name1 = f"{base_id}_EXP1_{lateralidad}_{orientacion}{ext}"
                expanded_name0 = f"{base_id}_EXP0_{lateralidad}_{orientacion}{ext}"
            print(f"Generando {expanded_name1} y {expanded_name0}")

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
                    
                        print(f"Intentando expandir {img_basename} con factor 1.25")
                        exp_img1, new_x1, new_w1 = es.expandSqueeze(
                            img, orientation , roi_x, roi_w, 1.25)
                        
                        print(f"Intentando expandir {img_basename} con factor 0.8")
                        exp_img0, new_x0, new_w0 = es.expandSqueeze(
                            img, orientation , roi_x, roi_w, 0.8)
                        
                        if exp_img1 is not None:
                            expanded_img1 = exp_img1
                            new_findings1[find].append([new_x1, roi_y, new_w1, roi_h])
                        
                        if exp_img0 is not None:
                            expanded_img0 = exp_img0
                            new_findings0[find].append([new_x0, roi_y, new_w0, roi_h])
            else:
                # Expandir la imagen si no tiene ROIs
                print(f"Expandiendo imagen completa {img_basename}...")
                expanded_img1, _, _ = es.expandSqueeze(img, orientation, 0, img.shape[1], 1.25)
                expanded_img0, _, _ = es.expandSqueeze(img, orientation, 0, img.shape[1], 0.8)

            if expanded_img1 is not None:
                print(f"Guardando imagen EXP1 en: {expanded_path1}")
                cv2.imwrite(os.path.join(data_root,expanded_path1), expanded_img1)
                new_annotations.append({
                    "image": expanded_path1,
                    "label": new_findings1})

            if expanded_img0 is not None:
                print(f"Guardando imagen EXP0 en: {expanded_path0}")
                cv2.imwrite(os.path.join(data_root, expanded_path0), expanded_img0)
                new_annotations.append({
                    "image": expanded_path0,
                    "label": new_findings0})

    except Exception as e:
        print(f"Error inesperado al procesar {img_path}: {e}, omitiendo...")
print(f"Numero de imágenes procesadas: {contadorImagenes}")
for new_annot in new_annotations:
    annotations[os.path.basename(new_annot["image"])] = new_annot

with open(jsonpath, 'w') as jsf:
    json.dump(annotations, jsf, indent=4)

print("Proceso completado correctamente")
