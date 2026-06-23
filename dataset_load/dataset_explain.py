import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps 
import pydicom
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import random
import math
from utils.transforms import apply_clahe, apply_top_hat, apply_morph_close, apply_entropy, gabor_bank, binarize_and_morph, resize_width_and_rois
from enhance_uniform import enhance_uniform
import yaml
from typing import List, Tuple, Dict
import copy

ALL_CLASSES = {
    "Nodulo": [],
    "Distorsion_arq": [],
    "Densidad_asim_foc": [],
    "Microcalcificaciones": [],
    "Calc_tip_benig": []
}




def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)




################# UTILITY METHODS, FOR SEPARATING ALL INTO ONE PY FILE #################

#METHOD FOR METRICS CALCULATION
def calculate_metrics(l_true, l_predict, threshold = 0.5):
    l_predict_bin = (l_predict>threshold).astype(int)
    precision = precision_score(l_true,l_predict_bin, average="binary", zero_division=0)
    recall = recall_score(l_true,l_predict_bin, average="binary",zero_division=0)
    f1 = f1_score(l_true,l_predict_bin, average="binary",zero_division=0)
    auc_roc = roc_auc_score(l_true,l_predict)
    acc = accuracy_score(l_true,l_predict_bin)

    # print(f"Metrics({'weighted'}): Precision:{precision}||Recall:{recall}|| F1_Score:{f1}|| Accuracy:{acc}")
    return acc, precision, recall, f1, auc_roc


#METHOD FOR CALCULATING CONFUSION MATRIX
def confusion_Matrix(all_labels, all_predicts, classes):
    
    num_classes = len(classes)
    combined_confMatrix = np.zeros((num_classes, 2, 2))  # Cada clase tendrá su propia matriz 2x2
    
    for index,cls in enumerate(classes):
        #LABELS FOR GT AND FOR PREDCLASS
        trueCls = all_labels[:,index]
        predCls = all_predicts[:,index]

        #CREATING CONFUSION MATRIX FOR EACH CLASS
        cm = confusion_matrix(trueCls, predCls)
        combined_confMatrix[index, :, :]=cm

    return combined_confMatrix
#METHOD FOR PLOTTING THE CONFUSION MATRIX
def plottingConfMatrx(cm, class_names):
    num_classes = len(class_names)

    # Imprime el número de clases para verificar
    # print(f"Número de clases: {num_classes}")

    # Crea una sola figura con subplots para cada matriz de confusión
    fig, axes = plt.subplots(1, num_classes, figsize=(10 * num_classes, 10)) # Ajusta el tamaño de la figura

    # Si solo hay una clase, axes no será una lista, así que lo convertimos en una lista
    if num_classes == 1:
        axes = [axes]

    for index, className in enumerate(class_names):
        # Imprime el índice y el nombre de la clase
        # print(f"Ploteando matriz para clase: {className} (índice: {index})")

        cm_int = cm[index].astype(int)
        #UNCOMMONT FOR PLOTTING THE CM

        try:
            sns.heatmap(cm_int, annot=True, fmt="d", cmap="plasma", cbar=False, square=True, ax=axes[index])
            axes[index].set_title(f'Confusion Matrix for {className}')
            axes[index].set_xlabel('Prediction')
            axes[index].set_ylabel('True Label')
        except Exception as e:
            print(f"Error al plotear heatmap para {className}: {e}")
    plt.tight_layout()  # Ajusta el espaciado entre subplots
    # plt.show()
    plt.close(fig)  
    return fig
    

#SCRIPT TO PRINT THE DATASET
def print_dataset(dataset):
    for index in range(len(dataset)):
        image, labels = dataset[index]  
        print(f"Index {index}:")
        print(f" - Image Path: {dataset.data[index][0]}") 
        print(f" - Labels: {labels}") 
        print("-" * 50)  


################# TRANSFORM METHODS #################

### THIS IS DONE LIKE THIS TO ENSURE THE REPRODUCIBILITY OF THE DATA AUGMENTATION ###

#METHOD FOR RETURNING THE TRANSFORM DATA AGUMENTATION
def data_augmentation_transform():
    return transforms.Compose([
    # transforms.Resize((512,512)),
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast = (0.8, 1.5)),
    transforms.RandomRotation(degrees = 10),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.078], std=[0.19]),  # Normalización
])



#METHOD FOR RETURNING THE TRANSFORM NORMAL DATA
def normal_transform():
    return transforms.Compose([
    transforms.RandomRotation(degrees = 0),        
    # transforms.Resize((512,512)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.078], std=[0.19]),  # Normalización
])


def rescale_image_and_rois(image, scaled_rois):
    scale_factor = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    scaled_rois = {
        roi_type: [[box[0] * scale_factor, box[1] * scale_factor, box[2] * scale_factor, box[3] * scale_factor]for box in boxes]
    for roi_type, boxes in scaled_rois.items()
    }

    return image, scaled_rois


################## BASIC IMPLEMENTATION FOR CUSTOM DATASET FOR DICOMS ################## 
class dicomDataset(Dataset):
    def __init__(self, dataPath, positive_classes, transform_with_class = None, limit = 10000, shuffleTrain = False, testDebug = False, seed = None,transforms_config = None, dataroot='.', withLTimeAugmentation=False):
        
        # fix_seed(seed)
        self.dataPath = dataPath
        self.dataroot = dataroot
        self.testDebug = testDebug
        # self.transforms_config = transforms_config
        # print(f"Transforms config in dataset: {self.transforms_config}")
        self.withLTimeAugmentation = withLTimeAugmentation

        cfg = transforms_config or {}

        # Flags y canales desde YAML
        self.load_flipped   = bool(cfg.get("flipped",  False))
        self.load_expanded  = bool(cfg.get("expanded", False))
        self.channels_spec  = cfg.get("channels", [])

        self.load_expanded_cropped = bool(cfg.get("expanded_cropped", False))
        self.expand_factors_cropped = cfg.get("expand_factors_cropped", [1.25, 0.8])  

        # legacy (no usado si hay channels)
        self.transforms_config = cfg.get("transformations", None)

        print(f"[dataset] flipped(V2): {self.load_flipped} | expanded(EXP): {self.load_expanded}")
        print(f"[dataset] channels: {self.channels_spec if self.channels_spec else 'None'}")

        if self.load_expanded_cropped:
            print(f"[dataset] expanded_cropped factors: {self.expand_factors_cropped}")

        self.transform = {
            "Clahe": apply_clahe,
            "Gabor": gabor_bank,
            "TopHat": apply_top_hat,
            "BinaryMask": binarize_and_morph,
        }
        # self.transform = transform
        self.data = []
        self.labels = []
        self.image_sizes = []
        self.rois = []
        self.positiveData =[]
        self.negativeData =[]
        self.orderedData =[]
        self.shuffleData = []
        self.CLASSES = dict()

        for id_class, class_name in enumerate(positive_classes):
            self.CLASSES[class_name] = id_class
        self.NUM_CLASSES = len(positive_classes)
        
        self.transform_with_class = transform_with_class
        self.shuffleTrain = shuffleTrain

        # ## Leer de augment_transform.yaml si flipped y/o expanded
        # self.load_flipped  = False
        # self.load_expanded = False
        # try:
        #     yaml_path = os.path.join(self.dataroot, "augment_transform.yaml")
        #     with open(yaml_path, "r") as yf:
        #         _cfg = yaml.safe_load(yf) or {}
        #         self.load_flipped  = bool(_cfg.get("flipped", False))
        #         self.load_expanded = bool(_cfg.get("expanded", False))
        # except FileNotFoundError:
        #     pass
        # print(f"[dataset] flipped(V2): {self.load_flipped} | expanded(EXP): {self.load_expanded}")


        if os.path.isdir(dataPath):
            self.load_dataset_from_dir(dataPath, limit)
        elif os.path.isfile(dataPath) and dataPath.endswith(".json"):
            self.load_dataset_from_json(dataPath,limit)
        else:
            print('No valid format for loading the dataset')
            exit()   
        self.datasetSize = len(self.data)
        if self.shuffleTrain: 
            self.orderDataset()
            self.shuffleDataset()

    def load_dataset_from_dir(self, dataPath, limit):
        patientElements = 0
        imageElements = 0
        roisElements = 0
        patientNoJson = 0
        for folder in os.listdir(dataPath):
            if len(self.data) > limit:
                break
            folderpath = os.path.join(dataPath,folder)    
            #DELVE INTO THE PATIENT FOLDERS
            for patient in os.listdir(folderpath):
                patient_pth = os.path.join(folderpath, patient)            
                print(patient_pth)
                patientElements+=1
                if os.path.isdir(patient_pth):
                    json_pth = os.path.join(patient_pth, "Rois.json")
                    #CHECK IF THE PATIENT FOLDER CONTAINS JSON ANNOTATIONS FILE
                    if not os.path.exists(json_pth):
                        patientNoJson+=1
                        #AQUI HAGO EL PROCESAMIENTO DE SI NO TIENE UN FICHERO JSON
                        for img in os.listdir(patient_pth):
                            imageElements+=1
                            imgpath = os.path.join(patient_pth, img)
                            # imgName = os.path.basename(imgpath)
                            labels = [0] * self.NUM_CLASSES
                            self.data.append((imgpath, labels))
                            self.labels.append(labels)
                        # print(f"Folder {patient} doesnt contain annotation 'Rois.json' file.")
                    else:
                        #JSON FILE LOAD WITH CHECKS
                        with open(json_pth, 'r') as jsf:
                            try:
                                annotations = json.load(jsf)
                            except json.JSONDecodeError:
                                # print(f"Json file {json_pth} is empty or corrupted.")
                                continue
                        roisElements+=1
                        #READING FOR THE JSON ANNOTATIONS
                        for key, value in annotations.items():
                            imageName = value["Nombre"]
                            image_pth = os.path.join(patient_pth, imageName)
                            if not os.path.isfile(image_pth):
                                # print(f"Image {image_pth} is not a file:")
                                continue    
                            imageElements+=1
                            #VECTOR FOR LABELING MULTICLASS
                            labels = [0] * self.NUM_CLASSES
                            if not value["Anotacion"]:
                                # print(f"Annotation dont exist")
                                # self.AnnotationNotcount +=1
                                self.data.append((image_pth, labels))
                                self.labels.append(labels)
                            else:
                                for lesion in value["Anotacion"]:
                                    lesionType = lesion["DescripcionLesion"]
                                    if lesionType in self.CLASSES:
                                        labels[self.CLASSES[lesionType]] = 1
                                self.data.append((image_pth, labels))
                                self.labels.append(labels)


        print(f"Número de pacientes:{patientElements}")
        print(f"Número de imágenes:{imageElements}")
        print(f"Número de jsons:{roisElements}")
        print(f"Número de pacientes sin json: {patientNoJson}")

    def load_dataset_from_json(self, dataPath, limit):
        count = 0
        discarded=0 
        imageDontreaded = 0
        with open(dataPath, 'r') as f:
            dataset = json.load(f)
        print(f"Total elementos en el json:{len(dataset)}")
        for _, d in dataset.items():
            if len(self.data) > limit:
                break

            #Flipped y expanded
            basename = os.path.basename(d["image"])
            is_flipped  = "_V2_" in basename          
            is_expanded = "_EXP"  in basename       

            if not self.withLTimeAugmentation and (is_flipped or is_expanded):
                continue

            is_original = (not is_flipped) and (not is_expanded)
            is_pure_v2  = is_flipped and (not is_expanded)
            is_pure_exp = is_expanded and (not is_flipped)
            is_combined = is_flipped and is_expanded  

            include = False
            if is_original:
                include = True  
            else:
                if self.load_expanded and self.load_flipped:
                    include = is_pure_v2 or is_pure_exp or is_combined                      
                elif self.load_expanded and not self.load_flipped:
                    include = is_pure_exp           
                elif self.load_flipped and not self.load_expanded:
                    include = is_pure_v2                
                else:
                    include = False                    

            if not include:
                continue

            if not os.path.exists(os.path.join(self.dataroot,d["image"])):
                discarded += 1
                print(f"Descartado por ruta inválida: {d['image']}")
                continue
            label = [0] * self.NUM_CLASSES
            for l, v in d["label"].items():
                if l in self.CLASSES:
                    if len(v) > 0:
                        label[self.CLASSES[l]] = 1
                    else:
                        label[self.CLASSES[l]] = 0

            file_name = d['image'].split('.')[-2]
            left_or_right = file_name.split('_')[-2]
            ml_or_cc = file_name.split('_')[-1]
            # print(f"File name: {file_name}, Left or Right: {left_or_right}, ML or CC: {ml_or_cc}")

            if is_flipped:
                if left_or_right == 'L':
                    left_or_right = 'R'
                elif left_or_right == 'R':
                    left_or_right = 'L'

            type_dict = {
                ('L', 'ML'): [1, 0, 0, 0],
                ('L', 'CC'): [0, 1, 0, 0],
                ('R', 'ML'): [0, 0, 1, 0],
                ('R', 'CC'): [0, 0, 0, 1]
            }
            image_type = np.array(type_dict.get((left_or_right, ml_or_cc), [0, 0, 0, 0]))
            # print(f"Image type: {image_type}")

            image_type = torch.tensor(image_type, dtype=torch.float32)            

            image = cv2.imread(os.path.join(self.dataroot,d["image"]))
            if image is None:
                print(f"Imagen no encontrada: {d['image']}")
                imageDontreaded += 1
                continue
            rois = d["label"]
            self.data.append((os.path.join(self.dataroot,d["image"]), rois, image_type, label))

            
            self.labels.append(label)
            shape = image.shape
            
            self.rois.append(rois)            
            self.image_sizes.append(shape)
        print(f"Total de imágenes no leidas por error{imageDontreaded}")
        print(f"Total de elementos descartados: {discarded}")
        print(f"Total de samples: {len(self.data)}")
        print(f"Total de labels: {len(self.data)}")


    def orderDataset(self):
        print(f"Longitud del dataset original: {len(self.data)}")
        for sample in self.data:
            data = ["", "", "",]
            #AQUI NECESITA ACCEDER CORRÉCTAMENTE A LOS DATOS.
            data = sample
            # print("Estos son los datos:")
            # print(data)
            if data[2][0]==1 :
                # print("Etiqueta positiva")
                self.positiveData.append(data)
            elif data[2][0]==0:
                # print("Etiqueta negativa")
                self.negativeData.append(data)
        #WARNING: CHECK IF ARRAY TYPE IS NUMPY OR PYTHON, IT CHANGES THE LOAD
        self.data = self.positiveData + self.negativeData
        self.datasetSize = len(self.positiveData)*2
        print(f"Longitud del dataset de muestras positivas:{len(self.positiveData)}")
        print(f"Longitud del dataset de muestras negativas{len(self.negativeData)}")
        print(f"Longitud del dataset ordenado:{len(self.data)}")
        
    def shuffleDataset(self):
        shuffledData = self.negativeData.copy()
        random.shuffle(shuffledData)
        self.data = self.positiveData + shuffledData[:len(self.positiveData)]   
        print(f"Longitud del dataset shuffled:{len(self.data)}")

    def generate_mask_image(self, rois, shape):
        mask_image = np.zeros(shape, dtype=np.float32)
        for r in rois:
            x, y, w, h = [int(v) for v in r]
            mask_image[y:y+h, x:x+w] = 1.0
        return mask_image


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_pth, image_rois, image_type, labels = self.data[index]
        scale_factor = 0.2

        image = cv2.imread(image_pth, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        scaled_rois = copy.deepcopy(image_rois)

        scaled_rois = {
            roi_type: [[box[0] * scale_factor, box[1] * scale_factor, box[2] * scale_factor, box[3] * scale_factor]for box in boxes]
        for roi_type, boxes in scaled_rois.items()
        }

        mask_img = self.generate_mask_image(scaled_rois[list(self.CLASSES.keys())[0]], image.shape)


        # Expandir/contraer las imagenes recortadas
        if self.load_expanded_cropped and self.expand_factors_cropped:
            factor = random.choice(self.expand_factors_cropped)  
            image, scaled_rois = resize_width_and_rois(image, scaled_rois, factor)
            if getattr(self, "testDebug", False):
                print(f"[expanded_cropped] factor={factor:.3f} -> nueva shape: {image.shape}")

        img_name = os.path.basename(image_pth)
        
        image_array_pil = Image.fromarray(image)
        
        if self.transform_with_class:
            image_array = self.transform_with_class(image_array_pil)
        else:
            image_array = image_array_pil

        base = np.array(image_array)

        # Usar "channels" del yaml (hasta 3 canales, mezclas ponderadas)
        if getattr(self, "channels_spec", None):
            channels_out = []

            # detectar si necesitamos Gabor normalizado o bruto
            needs_gabor_norm = False
            needs_gabor_raw  = False
            for spec in self.channels_spec:
                if spec in (0, "0", None) or (isinstance(spec, dict) and len(spec) == 0):
                    continue
                if not isinstance(spec, dict):
                    continue
                for k in spec.keys():
                    name = str(k).strip()
                    if name == "Gabor":
                        needs_gabor_norm = True
                    elif name == "Binary":
                        needs_gabor_raw = True

            gabor_norm_cached = None
            gabor_raw_cached  = None
            if needs_gabor_norm or needs_gabor_raw:
                gabor_raw_cached = gabor_bank(base)
            if needs_gabor_norm:
                g = gabor_raw_cached.astype(np.float32)
                mn, mx = float(g.min()), float(g.max())
                gabor_norm_cached = (
                    np.zeros_like(base, dtype=np.uint8)
                    if mx == mn else ((g - mn) / (mx - mn) * 255).astype(np.uint8)
                )


            def get_transform_image(name):
                if name == "Copy":
                    return base
                elif name == "Clahe":
                    return apply_clahe(base)
                elif name == "BigClahe":
                    # base2 = apply_morph_close(base, (5,5))
                    # return apply_clahe(base2, 2, (8,8))
                    base2 = apply_clahe(base)
                    base2 = cv2.medianBlur(base2, 9)

                    return base2
                elif name == "TopHat":
                    return apply_top_hat(base)
                elif name == "MorphClose":
                    return apply_morph_close(base)
                elif name == "Entropy":
                    return apply_entropy(base)
                elif name == 'EnhanceUniform':
                    return enhance_uniform(base)
                elif name == "Gabor":
                    if gabor_norm_cached is None:
                        g = gabor_bank(base).astype(np.float32)
                        mn, mx = float(g.min()), float(g.max())
                        return np.zeros_like(base, dtype=np.uint8) if mx == mn else ((g - mn) / (mx - mn) * 255).astype(np.uint8)
                    return gabor_norm_cached
                elif name == "Binary":
                    gr = gabor_raw_cached if gabor_raw_cached is not None else gabor_bank(base)
                    return binarize_and_morph(gr).astype(np.uint8)
                elif name.startswith("Binarize_"):
                    parts = name.split("_")
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        lo, hi = int(parts[1]), int(parts[2])
                        base_clahe = apply_clahe(base)
                        return np.where((base_clahe >= lo) & (base_clahe < hi), 255, 0).astype(np.uint8)
                    raise ValueError(f"Canal '{name}' mal definido; usa Binarize_min_max (p.ej. Binarize_125_175)")
                else:
                    if name in self.transform:
                        out = self.transform[name](base)
                        return out if out.dtype == np.uint8 else np.clip(out, 0, 255).astype(np.uint8)
                    raise ValueError(f"Transform '{name}' no soportada")
                
            
            for spec in self.channels_spec:
                if len(channels_out) >= 3:
                    break
                # canal vacío
                if spec in (0, "0", None) or (isinstance(spec, dict) and len(spec) == 0):
                    channels_out.append(np.zeros_like(base, dtype=np.uint8))
                    continue
                if not isinstance(spec, dict):
                    raise ValueError(f"Canal mal definido en YAML: {spec} (usa dict, 0 o {{}})")

                names, weights = [], []
                for k, v in spec.items():
                    k_key = str(k).strip()
                    if k_key in ("Zero", "None"):
                        names, weights = ["Zero"], [1.0]
                    else:
                        names.append(k_key)
                        weights.append(float(v))

                w = np.array(weights, dtype=np.float32)
                # w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / max(1, len(w)))
                w = w/100

                acc = np.zeros_like(base, dtype=np.float32)
                for k_key, alpha in zip(names, w):
                    if k_key in ("Zero",):
                        arr = np.zeros_like(base, dtype=np.uint8)
                    else:
                        arr = get_transform_image(k_key)
                    acc += alpha * arr.astype(np.float32)

                channels_out.append(np.clip(acc, 0, 255).astype(np.uint8))

            finalImage = np.stack(channels_out, axis=0) if channels_out else base[None, ...]
            image_array = (torch.from_numpy(finalImage).float()) / 255.0
            mask_array = torch.from_numpy(mask_img).float()

        else:
            image_array = transforms.ToTensor()(image_array)

        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return image_array, scaled_rois, image_type, label_tensor, img_name, mask_array
    

# Collate and help functions

def round_up(x: int, base: int) -> int:
    return int(math.ceil(x / base)) * base

def make_pad_to_multiple(sizes: List[Tuple[int,int]], multiple: int = 32) -> Tuple[int,int]:
    """
    Given a list of (H, W) for the batch, return padded (H_pad, W_pad),
    each rounded up to the given multiple (32 for ResNet50).
    """
    H_max = max(h for h, w in sizes)
    W_max = max(w for h, w in sizes)
    return round_up(H_max, multiple), round_up(W_max, multiple)


# --------- Collate: pad to (H_pad, W_pad) and build masks ---------

def collate_pad_to32(batch):
    """
    batch: list of (img, label)
      - img should be [3,H,W], variable H,W
    """
    imgs, image_rois, img_types, labels, img_names, mask_imgs = zip(*batch)
    B = len(imgs)

    proc_imgs = []
    sizes = []
    for x, m in zip(imgs, mask_imgs):
        assert x.dim() == 3, "img must be [C,H,W]"
        C, H, W = x.shape
        sizes.append((H, W))
        proc_imgs.append((x, m))



    H_pad, W_pad = make_pad_to_multiple(sizes, multiple=32)

    batch_imgs = torch.zeros(B, 3, H_pad, W_pad, dtype=proc_imgs[0][0].dtype)
    batch_masks = torch.zeros(B, 1, H_pad, W_pad, dtype=proc_imgs[0][1].dtype)
    # orig_sizes = torch.zeros(B, 2, dtype=torch.long)

    for i, x in enumerate(proc_imgs):
        _, H, W = x[0].shape
        batch_imgs[i, :, :H, :W] = x[0]
        batch_masks[i, 0, :H, :W] = x[1]
        # orig_sizes[i] = torch.tensor([H, W])

    labels = torch.stack(labels).view(B, 1)
    img_types = torch.stack(img_types).view(B, 4)

    return batch_imgs, list(image_rois) ,img_types, labels, list(img_names), batch_masks

    # return {
    #     "imgs": batch_imgs,
    #     "labels": labels,
    #     "masks": batch_masks,
    #     "orig_sizes": orig_sizes,
    #     "meta": list(metas),
    # }


