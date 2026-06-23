
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import models
import cv2
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_load.dataset import dicomDataset, normal_transform, calculate_metrics, confusion_Matrix, print_dataset, collate_pad_to32
import random
from models.base_models_final import  EfficientNetB0, CustomResNetBinary,  CustomResNetBinary34, CustomResNetBinary50,  VGG16, CustomDenseNet, CustomMobileNetV3
import argparse
import os
import json
import yaml
import copy
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, kendalltau, weightedtau
from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.ops import box_iou
from utils.utils import pos_weight_samples
from collections import Counter
from tqdm import tqdm
WIMG = 709
HIMG = 800
SHOW_IMAGES = True
PR_TH = 0.5
PEARSON_THRESHOLD = 0.5

General_Exp_metrics = { "Total_positives": 0, "Total_negatives": 0, "True_positives": 0,
                        "True_negatives": 0, "False_positives": 0, "False_negatives": 0,
                        "TP_Perfect": 0, "TP_Good": 0, "TP_Weak": 0, "TP_Failed": 0,
                        "FN_Perfect": 0, "FN_Good": 0, "FN_Weak": 0, "FN_Failed": 0}

###ARGUMENTS PARSER METHOD FOR LOAD A CONFIG FILE IF EXIST###
def parse_args(parser):
    '''     
    Standard argument parser
    '''
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args



def find_last_spatial_layer(model, input_size=(1, 3, 224, 224), device="cpu"):
    model = model.to(device).eval()
    x = torch.zeros(input_size).to(device)

    last_spatial_module = None

    def hook(module, inp, out):
        nonlocal last_spatial_module
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            last_spatial_module = module

    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return last_spatial_module

#WRAPPER ADAPTER FOR THE FORWARD, PASSING XTYPE AND OUT_ADAPTER NEEDS TO BE ADDRESSED
class CamWrapperAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._types = None

    def set_types(self, t: torch.Tensor):
        self._types = t


    #MAKES DISTINCTION BETWEEN FORMS OF RETURN, LIKE TYPES AND ADAPTER OUTPUT
    def forward(self, x: torch.Tensor):
        out = self.model(x, self._types)
        if isinstance(out,(tuple, list)):
            out = out[0]
        return out

def draw_rois(image, rois, size, classes, color = (0,255,0)):
    h_img, w_img = image.shape[0], image.shape[1]
    for k, r_list in rois.items():
        if k in classes:
            for r in r_list:
                p1 = (int(r[0]*w_img/size[1]), int(r[1]*h_img/size[0]))
                p2 = (int((r[0]+r[2])*w_img/size[1]), int((r[1]+r[3])*h_img/size[0]))
                image = cv2.rectangle(image, p1, p2, color, 2)

    return image



def get_gradCam_map(cam_model, types, target_layers, inputs):
    with torch.enable_grad():
        cam_model.set_types(types)
        gradcam_maps = []
        for i, layer in enumerate(target_layers):
            cam_ctx = GradCAMPlusPlus(model=cam_model, target_layers=[layer])
            with cam_ctx as cam:
                grayscale_cam = cam(input_tensor=inputs, targets=[BinaryClassifierOutputTarget(1)])[0]
                gradcam_maps.append(grayscale_cam)

    return gradcam_maps

def get_eigenCam_map(cam_model, types, target_layers, inputs):
    with torch.enable_grad():
        cam_model.set_types(types)
        eigencam_maps = []
        for i, layer in enumerate(target_layers):
            cam_ctx = EigenCAM(model=cam_model, target_layers=[layer])
            with cam_ctx as cam:
                grayscale_cam = cam(input_tensor=inputs, targets=[BinaryClassifierOutputTarget(1)])[0]
                eigencam_maps.append(grayscale_cam)

    return eigencam_maps


def show_combined_images(img1, img2, title):
    concat_img = np.concatenate((img1, img2), axis = 1)
    concat_img = cv2.resize(concat_img, (800, 400))
    cv2.imshow(title, concat_img)
    k = cv2.waitKey(0)
    return k
                

def cuantitative_metrics_report(all_labels, all_predictions, bestLoss, loadedseed, bestmodel, json_suffix, save_completeMetrics_path, Bias):
    print("\n" + "="*25)
    print("CUANTITATIVE PERFORMANCE METRICS REPORT")
    print("\n" + "="*25)
    if True:
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        # print(f"All labels shape: {all_labels.shape}")
        # print(f"All predictions shape: {all_predictions.shape}")


        acc, precision, recall, f1, auc_roc = calculate_metrics(all_labels, all_predictions, threshold=0.5)
        precision_auprc, recall_auprc, _ = precision_recall_curve(all_labels, all_predictions)
        auprc = auc(recall_auprc, precision_auprc)
        print(f"TestLoss:{bestLoss:.4f} || Precision:{precision:.4f} || Recall:{recall:.4f} || F1 Score:{f1:.4f} || AUC-ROC:{auc_roc:.4f} || Accuracy:{acc:.4f}")
        #OPENING A JSON TO SAVE THE METRICS
        metrics = {
            "Seed": loadedseed,
            "Model-Run": bestmodel,
            "Test Loss": bestLoss,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC-ROC": auc_roc,
            "AUPRC": auprc,
            "Accuracy": acc
        }
        if Bias is not None:
            metrics["Bias"] = Bias

    # except ValueError as ve:
    #     print(f"Error to concatenate: {ve}")
    return metrics

def explainable_acc_report(all_labels, all_predictions_explained, topK):
    try:
        all_labels = np.array(all_labels)
        k = topK[0]
        all_predictions_explained[k] = np.array(all_predictions_explained[k])

        acc, precision, recall, f1, auc_roc = calculate_metrics(all_labels, all_predictions_explained[k], threshold=0.5)
        precision_auprc, recall_auprc, _ = precision_recall_curve(all_labels, all_predictions_explained[k])
        auprc = auc(recall_auprc, precision_auprc)

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC-ROC": auc_roc,
            "AUPRC": auprc,
            "Accuracy": acc
        }

    except ValueError as ve:
        print(f"Error to concatenate: {ve}")
    return metrics

def explainable_weighted_report(all_labels, all_predictions_explained):
    metrics = {}
    try:
        all_labels = np.array(all_labels)
        for th in all_predictions_explained:
            all_predictions = np.array(all_predictions_explained[th])

            acc, precision, recall, f1, auc_roc = calculate_metrics(all_labels, all_predictions, threshold=0.5)
            precision_auprc, recall_auprc, _ = precision_recall_curve(all_labels, all_predictions)
            auprc = auc(recall_auprc, precision_auprc)

            metrics[th] = {
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUC-ROC": auc_roc,
                "AUPRC": auprc,
                "Accuracy": acc
            }

    except ValueError as ve:
        print(f"Error to concatenate: {ve}")
    return metrics


def explainable_metrics_report(map_results, pos_classes_predictions, logits):
    metrics = {}
    for mtype in map_results:
        for th in map_results[mtype]['energy']:
            mean_energy = float(np.mean(np.array(map_results[mtype]['energy'][th])))
            if 'energy_'+str(th) not in metrics:
                metrics['energy_'+str(th)] = {}
            metrics['energy_'+str(th)][mtype] = mean_energy
            
        if 'PG_1-top' not in metrics.keys():
            metrics['PG_1-top'] = {}
        metrics['PG_1-top'][mtype] = float(np.mean(np.array(map_results[mtype]['PG_1-top'], dtype=np.float32)))*100.
            
        if 'PG_5-top' not in metrics.keys():
            metrics['PG_5-top'] = {}
        metrics['PG_5-top'][mtype] = float(np.mean(np.array(map_results[mtype]['PG_5-top'], dtype=np.float32)))*100.

        if 'pearson' not in metrics.keys():
            metrics['pearson'] = {}
        if 'spearman' not in metrics.keys():
            metrics['spearman'] = {}
        if 'ccc' not in metrics.keys():
            metrics['ccc'] = {}

        np_logits = np.array(logits)
        np_mass = np.array(map_results[mtype]['map_mass'])
        if (np_logits>0).sum()>=2:
            np_logits_pos = np_logits[np_logits>0]
            np_mass_pos = np_mass[np_logits>0]

            pearson_result = pearsonr(np_logits_pos, np_mass_pos)
            spearman_result = spearmanr(np_logits_pos, np_mass_pos)
            ccc_result = concordance_corrcoef(np_logits_pos, np_mass_pos)
            metrics['pearson'][mtype] = [pearson_result.statistic, pearson_result.pvalue]
            metrics['spearman'][mtype] = [spearman_result.statistic, spearman_result.pvalue]
            metrics['ccc'][mtype] = [ccc_result, 0]
        else:
            metrics['pearson'][mtype] = [0, 1]
            metrics['spearman'][mtype] = [0, 1]
            metrics['ccc'][mtype] = [0,1]


    print(metrics)
    return metrics



def get_roi_attention_level(att_map, roi, topK):
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    top_K_att = dict()
    for k in topK:
        n_pixels = int(w*h*k/100)
        top_att = np.mean(np.sort(att_map[y:y+h, x:x+h].flatten())[-n_pixels:])
        # top_K_att[k] = top_att
        pct = 10
        y_min = max(0, y-int(pct*h/100))
        y_max = min(y+h+int(pct*h/100), att_map.shape[0])
        x_min = max(0, x-int(pct*w/100))
        x_max = min(x+w+int(pct*w/100), att_map.shape[1])

        top_K_att[k] = np.sum(att_map[y_min:y_max, x_min:x_max])    
    sum_att = np.sum(att_map[y:y+h, x:x+h])

    return top_K_att, sum_att
  
def get_attention_weight_2(att_map, rois, img_size, topK = [20, 40, 60, 80]):
    H_orig, W_orig = img_size[0], img_size[1]
    half = (np.min(att_map)+np.max(att_map))/2
    norm_att_map = np.copy(att_map) #cv2.resize(att_map, (W_orig, H_orig))
    norm_att_map[norm_att_map<half] = 0
    ground_truth_mask = np.zeros_like(norm_att_map).astype(np.uint8)

    H_scale, W_scale = norm_att_map.shape[0]/H_orig, norm_att_map.shape[1]/W_orig

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    rois_att_list = []
    area_roi = []
    total_area_roi = 0
    for r in rois:
        roi_mask = np.zeros_like(norm_att_map).astype(np.uint8)
        cx, cy = r[0]+r[2]/2, r[1]+r[3]/2
        w, h = max(int(r[2]*W_scale), 1), max(int(r[3]*H_scale), 1)
        x, y = int(cx*W_scale-w/2), int(cy*H_scale-h/2)
        # x, y = int((r[0])*W_scale), int(r[1]*H_scale)
        # w, h = max(int(r[2]*W_scale), 1), max(int(r[3]*H_scale), 1)
        ground_truth_mask[y:y+h, x:x+w] = 1
        roi_mask[y:y+h, x:x+w] = 1
        roi_mask = cv2.dilate(roi_mask, kernel)
        rois_att_list.append(np.sum(norm_att_map[roi_mask==1]))
        area_roi.append(w*h)
        total_area_roi += w*h

    ground_truth_mask = cv2.dilate(ground_truth_mask, kernel)
    rois_att = np.sum(norm_att_map[ground_truth_mask==1])
    bg_att = np.sum(norm_att_map[ground_truth_mask==0])

    img_to_show = cv2.resize(ground_truth_mask*255, (400, 400))
    # cv2.imshow("mask", img_to_show)

    rois_att_np = np.array(rois_att_list).astype(np.float32)

    w_roi = np.array(area_roi, dtype=np.float32)/total_area_roi
    rois_att_np = rois_att_np/(rois_att_np+bg_att*w_roi+0.000000001)
    
    print(rois_att_np)
    mean_att = np.average(rois_att_np, weights=w_roi)


    attention_level = dict()
    all_att = np.sum(norm_att_map)
    for k in topK:
        attention_level[k] = mean_att #rois_att / all_att


    return attention_level

def get_ground_truth_mask(orig_size, map_size, rois):
    H_orig, W_orig = orig_size[0], orig_size[1]
    H_map, W_map = map_size[0], map_size[1]
    ground_truth_mask = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)

    H_scale, W_scale = H_map/H_orig, W_map/W_orig
    for r in rois:
        w, h = max(int(r[2]*W_scale+1), 1), max(int(r[3]*H_scale+1), 1)
        x, y = max(0, int(r[0]*W_scale)), max(0, int(r[1]*H_scale))
        x2, y2 = min(x+w, map_size[1]), min(y+h, map_size[0]) 
        ground_truth_mask[y:y2, x:x2] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    ground_truth_mask = cv2.dilate(ground_truth_mask, kernel)

    return ground_truth_mask

def get_attention_weight(map, ground_truth_mask, th_zero = 0.5, normalize = True):
    norm_map = np.copy(map)
    norm_map[norm_map<th_zero] = 0

    rois_att = np.sum(norm_map[ground_truth_mask==1])

    total_attention = np.sum(norm_map)
    attention_level = rois_att
    if normalize and total_attention>0:
        attention_level = attention_level/np.sum(norm_map)
    return attention_level

def pointing_game_topK(map, rois, img_size, K=1):
    H_orig, W_orig = img_size[0], img_size[1]
    H_scale, W_scale = map.shape[0]/H_orig, map.shape[1]/W_orig

    hit = 0
    max_indices = np.argsort(map, axis=None)[-K:]
    for r in rois:
        cx, cy = r[0]+r[2]/2, r[1]+r[3]/2
        w, h = max(int(r[2]*W_scale), 1), max(int(r[3]*H_scale), 1)
        for idx in max_indices:
            max_y = idx // map.shape[1]
            max_x = idx % map.shape[1]
            if abs(max_x-cx)<w/2 and abs(max_y-cy)<h/2:
                hit = 1
                break
    return hit

def get_map_mass(map, K):
    # max_indices = np.argsort(map, axis=None)
    # nIndices = max(int(len(max_indices)*K/100), 1)
    # topKIndices = max_indices[-nIndices:]
    map_mass = map.sum()
    # mass = map.flatten()[topKIndices].sum()
    # mass = mass/(map_mass+np.finfo(np.float32).eps)
    # mass = 0
    # # positions = []
    # for idx in topKIndices:s
    #     y = idx // map.shape[1]
    #     x = idx % map.shape[1]
    #     mass += map[y,x]
    #     # positions.append((x,y))
    return map_mass


def normalize_map(map):
    map_norm = (map-np.min(map))
    diff_max_min = (np.max(map)-np.min(map))
    if diff_max_min>0:
        map_norm = map_norm/diff_max_min
    grayscale_map = (map_norm*255).astype(np.uint8)

    return map_norm, grayscale_map

def concordance_corrcoef(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)


def my_metric(logits, mass):
    dist = 1.-np.abs(logits-mass)/(np.abs(logits)+np.abs(mass)+np.finfo(np.float32).eps)
    # print("logits",logits)
    # print("mass",mass)
    # print("dist",dist)
    dist = np.mean(dist)
    # errors = []
    # for l, m in zip(logits, mass):
    #     e = abs(l-m) / (abs(l)+abs(m)+np.finfo(np.float32).eps)
    #     errors.append(e)
    # dist = np.mean(np.array(errors))
    return dist

def get_model_metrics(testDataset, positive_classes, loadedseed, modelName, bestModelPth, dataroot='.',   transformsConfig=None, inChannels=None, save_completeMetrics_path=None, json_suffix=None, show_image = False, testDebug = False, limit = 10000, Bias=None):

    seed = torch.manual_seed(loadedseed)
    print(f"Best model path: {bestModelPth}")
    bestmodel = bestModelPth.split("/")[-1]
    print(f"Best model: {bestmodel}")
    testDebugging = testDebug

    NUM_CLASSES = len(positive_classes)
    #SETTING THE ARCHITECTURE OF THE MODEL
    if modelName == "CustomResNetBinary":
        model = CustomResNetBinary(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [model.base_model[-1][-1]]
    elif modelName == "CustomResNetBinary34":
        model = CustomResNetBinary34(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [model.base_model.layer4[-1]]
    elif modelName == "CustomResNetBinary50":
        model = CustomResNetBinary50(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [model.base_model[-1][-1], model.head.attn[-1]]
    elif modelName == "EfficientNetB0":
        model = EfficientNetB0(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [find_last_conv(model.base_model)]
    elif modelName == "CustomDenseNet":
        model = CustomDenseNet(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [find_last_conv(model.base_model)]
    elif modelName == "VGG16":
        model = VGG16(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [find_last_conv(model.features)]
    elif modelName == "CustomMobileNetV3":
        model = CustomMobileNetV3(num_classes=NUM_CLASSES, in_channels=inChannels)
        # target_layer = [find_last_conv(model.base_model)]
    else:
        raise ValueError(f"Model {modelName} not recognized. Please choose a valid model.")

    # target_layer = [find_last_spatial_layer(model.base_model), model.head.attn[-1]]
    target_layer = [find_last_spatial_layer(model.base_model), model.head.proj, model.head.attn[-1]]

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    
    state_dict = torch.load(bestModelPth, map_location = device)
    model.load_state_dict(state_dict)
    model = model.to(device) 
    # print(model)

    cam_model = CamWrapperAdapter(model).to(device)

    normal_data = normal_transform()
    #GETTING WHOLE DATASET
    print(f"[DEBUG] test_model.py | dataroot: {dataroot}")
    print(f"[DEBUG] test_model.py | testDataset: {testDataset}")
    DatasetDicom = dicomDataset(dataPath = testDataset ,positive_classes = positive_classes, transform_with_class = normal_data, transforms_config=transformsConfig, testDebug=testDebugging ,dataroot=dataroot, limit=limit)

    #MAKING THE STRATIFICATION
    test_dataset = DatasetDicom
    test_labels = np.array(test_dataset.labels)
    print(f"Test Dataset Size: {len(test_dataset)}")
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_test)

    #  ##CONFIGURATION MODE FOR THE MODEL
    pos_weights = pos_weight_samples(test_labels, device)
    criterion_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight = pos_weights)
    #INFERENCE  
    model.eval()
    all_labels = []
    all_logits = []
    all_predictions = []
    all_predictions_explained_W = {}
    pos_classes_predictions = []
    
    LContrib_Th = [0.25, 0.5, 0.75]
    for th in LContrib_Th:
        all_predictions_explained_W[th] = []
    
    
    map_results = {}
    map_types = ['contribution', 'attention', "contrib_no_bias",
                 'grad_cam_cnn', 'grad_cam_proj', 'grad_cam_att',
                 'eigen_cam_cnn', 'eigen_cam_proj', 'eigen_cam_att']

    # map_types = ['contribution', 'attention', "contrib_no_bias"]

    energy = {}
    for th in LContrib_Th:
        energy[th] = []
    for t in map_types:
        map_results[t] = {'map': None, 'norm_map': None, 'gray_map': None, # temporal. Used for the current sample
                          'raw_energy': [], 'energy': copy.deepcopy(energy),
                          'PG_1-top': [], 'PG_5-top': [], 'map_mass': []}

    if len(positive_classes) == 1:
        title = positive_classes[0]
    else:
        title = 'image'
    font = cv2.FONT_HERSHEY_SIMPLEX

    stop_showing = False
    fp_names = []
    test_loss = 0.0
    

    images_to_save = ["0a3018e7ad1d1d7d2e142c2ca7c518fa_L_CC.png"]
    with torch.no_grad():
        for inputs, rois, types, labels, img_name in test_data_loader:
            if len(rois[0][positive_classes[0]])<1:
                continue
            img_name = str(img_name[0])
            # if img_name not in images_to_save:
            #     continue
            # print(f"\n----------Metrics score for image : {img_name}----------", flush=True)
            inputs, labels = inputs.to(device), labels.to(device)
            types = types.to(device)
            # outputs, att_map, imp_map, feat_map, max_w = model(inputs, types)
            outputs, att_map, imp_map, no_bias_map = model(inputs, types)

            loss = criterion_loss(outputs, labels)

            test_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted_class = (probabilities >= PR_TH).float()

            for img, batch_rois ,label, pred_class, prob, logits in zip(inputs.tolist(), rois ,labels.tolist(), predicted_class.tolist(), probabilities.tolist(), outputs.tolist()):
                # print(f"-True image label: {l} || Label image prediction: {p} || Correct prediction probability: {pr}")
                pr_text = str(int(prob[0]*100)) + '%'

                ####IMAGE CONSTRUCTION####

                cvimg = torch.permute(torch.tensor(img), (1, 2, 0))
                cvimg = np.array((cvimg)*255).astype(np.uint8)
                img_size = cvimg.shape
                cvimg_orig = cv2.resize(cvimg[:,:,0], (WIMG, HIMG))
                cvimg = cv2.cvtColor(cvimg_orig, cv2.COLOR_GRAY2BGR)
                cvimg = draw_rois(cvimg, batch_rois, img_size, positive_classes)
                cvimg = cv2.putText(cvimg, pr_text, (50,50), font, 1, (0,255,0), 2)
                base = cvimg_orig.astype(np.float32) / 255.0
                base = np.stack([base, base, base], axis=-1)


                cv_imp_map = np.array(torch.permute(imp_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach())
                # print("Total contribution with bias", cv_imp_map.sum())
                # cv_imp_map_norm, grayscale_imp_map = normalize_map(np.maximum(0, cv_imp_map))
                cv_imp_map_norm, grayscale_imp_map = normalize_map(cv_imp_map)
                map_results['contribution']['map'] = cv_imp_map
                map_results['contribution']['norm_map'] = cv_imp_map_norm
                map_results['contribution']['gray_map'] = grayscale_imp_map
                
                cv_att_map = np.array(torch.permute(att_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach())
                cv_att_map_norm, grayscale_att_map = normalize_map(cv_att_map)
                map_results['attention']['map'] = cv_att_map
                map_results['attention']['norm_map'] = cv_att_map_norm
                map_results['attention']['gray_map'] = grayscale_att_map
                
                cv_contrib_no_bias_map = np.array(torch.permute(no_bias_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach())
                # print("Total contribution without bias", cv_contrib_no_bias_map.sum())
                # cv_contrib_no_bias_map_norm, grayscale_contrib_no_bias_map = normalize_map(np.maximum(0, cv_contrib_no_bias_map))
                cv_contrib_no_bias_map_norm, grayscale_contrib_no_bias_map = normalize_map(cv_contrib_no_bias_map)                
                map_results['contrib_no_bias']['map'] = cv_contrib_no_bias_map                
                map_results['contrib_no_bias']['norm_map'] = cv_contrib_no_bias_map_norm
                map_results['contrib_no_bias']['gray_map'] = grayscale_contrib_no_bias_map



                # results from EIGEN-CAM
                eigen_cam_types = ['eigen_cam_cnn', 'eigen_cam_proj', 'eigen_cam_att']
                att_maps_eigencam = get_eigenCam_map(cam_model, types, target_layer, inputs)
                for imap, tmap in enumerate(eigen_cam_types):
                    norm_map, gray_map = normalize_map(att_maps_eigencam[imap])
                    map_results[tmap]['map'] = att_maps_eigencam[imap]
                    map_results[tmap]['norm_map'] = norm_map
                    map_results[tmap]['gray_map'] = gray_map


                # results from GRAD-CAM
                grad_cam_types = ['grad_cam_cnn', 'grad_cam_proj', 'grad_cam_att']
                att_maps_gradcam = get_gradCam_map(cam_model, types, target_layer, inputs)
                for imap, tmap in enumerate(grad_cam_types):
                    norm_map, gray_map = normalize_map(att_maps_gradcam[imap])
                    map_results[tmap]['map'] = att_maps_gradcam[imap]
                    map_results[tmap]['norm_map'] = norm_map
                    map_results[tmap]['gray_map'] = gray_map
                    
                # topK_points = dict()
                for mtype in map_types:
                    # print(mtype)
                    map_mass = get_map_mass(map_results[mtype]['map'], K = 5)
                    map_results[mtype]['map_mass'].append(map_mass)                            

                ground_truth_mask = get_ground_truth_mask(img_size, img_size,batch_rois[positive_classes[0]])
                contrib_weights = {th: 1 for th in LContrib_Th}                    
                if len(batch_rois[positive_classes[0]])>0:
                    pos_classes_predictions.append(prob[0])
                    for mtype in map_types:
                        resized_map = cv2.resize(map_results[mtype]['norm_map'], (img_size[1], img_size[0]))                    
                        for th in LContrib_Th:
                            map_results[mtype]['energy'][th].append(get_attention_weight(resized_map, ground_truth_mask, th_zero=th))
                        map_results[mtype]['PG_1-top'].append(pointing_game_topK(resized_map, batch_rois[positive_classes[0]], img_size, K = 1))                            
                        map_results[mtype]['PG_5-top'].append(pointing_game_topK(resized_map, batch_rois[positive_classes[0]], img_size, K = 5))
                        # resized_orig_map = cv2.resize(map_results[mtype]['map'], (img_size[1], img_size[0]))                    
                        # map_results[mtype]['raw_energy'].append(get_attention_weight(resized_orig_map, ground_truth_mask, th_zero=0, normalize=False))
                    for th in LContrib_Th:
                        contrib_weights[th] = map_results['contribution']['energy'][th][-1]
            
                # print("logits - bias - no bias", logits[0], map_results["contribution"]['topk_mass'][-1], map_results["contrib_no_bias"]["topk_mass"][-1],
                #       map_results["contribution"]['topk_mass'][-1]-map_results["contrib_no_bias"]["topk_mass"][-1])                    
                if show_image:
                    wpr_text = str(int(prob[0]*contrib_weights[0.5]*100)) + '%'
                    # print(contrib_weights)
                    # print(prob[0], contrib_weights[0.5], wpr_text)
                    # gt_map = ground_truth_mask*255
                    # gt_map = cv2.cvtColor(gt_map, cv2.COLOR_GRAY2BGR)
                    # gt_map = draw_rois(gt_map, batch_rois, img_size, positive_classes)    
                    # gt_map = cv2.resize(gt_map, (400, 400))
                    # cv2.imshow('GT', gt_map.astype(np.uint8))
                    maps_to_show = ['contribution', 'eigen_cam_cnn']
                    maps_to_save = ['contribution', 'attention', 'grad_cam_cnn', 'eigen_cam_cnn']
                    vis_maps = []
                    clean_maps = []
                    for im, mtype in enumerate(maps_to_save):
                        gray_map = map_results[mtype]['gray_map']
                        if gray_map.shape[:2] != base.shape[:2]:
                            gray_map = cv2.resize(gray_map, (base.shape[1], base.shape[0]))

                        text = pr_text
                        if im>0:
                            text = wpr_text
                        heatmap = show_cam_on_image(base, gray_map/255, use_rgb=False, image_weight=0.7)      
                        clean_maps.append(heatmap.copy())
                        if mtype in maps_to_show:
                            heatmap = draw_rois(heatmap, batch_rois, img_size, positive_classes)    
                            heatmap = cv2.putText(heatmap, text, (50,50), font, 1, (0,255,0), 2)                        
                            vis_maps.append(heatmap)
                       
                    
                    prefix = ""
                    k = show_combined_images(vis_maps[0], vis_maps[1], title)
                    if k==115: # 's'
                        cvimg_orig = draw_rois(cv2.cvtColor(cvimg_orig, cv2.COLOR_GRAY2BGR), batch_rois, img_size, positive_classes)                            
                        cv2.imwrite('./images_against_posthoc/'+'input_'+prefix+"_"+img_name+'.png', cvimg_orig)
                        names = ['./images_against_posthoc/'+mname+"_"+prefix+"_"+img_name+".png"for mname in maps_to_save]
                        for map_to_save, f_path in zip(clean_maps, names):
                            cv2.imwrite(f_path, map_to_save)
                    elif k == 27:  # Esc
                        cv2.destroyAllWindows()
                        exit()


            all_labels.append(label[0])
            all_logits.append(logits[0])
            all_predictions.append(prob[0])
            predict_explained = prob[0]
            for th in LContrib_Th:
                all_predictions_explained_W[th].append(predict_explained*contrib_weights[th])
            if stop_showing:
                break
        
    
    print("\n" + "="*50)
    print("="*12 + "FINAL TEST MODEL REPORT" + "="*14)
    print("\n" + "="*50)
    
    average_test_loss = test_loss / len(test_data_loader)
    cuantitative_metrics = cuantitative_metrics_report(all_labels, all_predictions, average_test_loss, loadedseed, bestmodel, json_suffix, save_completeMetrics_path, Bias)
    explainable_weighted_metrics = explainable_weighted_report(all_labels, all_predictions_explained_W)
    explainable_metrics = explainable_metrics_report(map_results, pos_classes_predictions, all_logits)
    
    outputFile = "false_positives.txt"
    with open(outputFile, 'w')as f:
        for filename in fp_names:
            f.write(f"{filename}\n")
    print(f"False positive images list saved in: {outputFile}")

    
    
    #### CUANTITATIVE METRICS REPORT ####
    
    final_metrics_report={
        "cuantitative_metrics":cuantitative_metrics,
        "explainable_weighted_metrics":explainable_weighted_metrics,
        "explainable_metrics":explainable_metrics
    }

    json_name = "Final_metrics_runs.jsonl" if not json_suffix else f"Final_metrics_runs_{json_suffix}.jsonl"
    with open(os.path.join(save_completeMetrics_path, json_name), "a") as f:
        json.dump(final_metrics_report, f, indent=4)

def collate_test(batch):
    """
    batch: list of (img, label)
      - img should be [3,H,W], variable H,W
    """
    imgs, image_rois, img_types, labels, img_names = zip(*batch)
    B = len(imgs)

    proc_imgs = []
    sizes = []
    for x in imgs:
        assert x.dim() == 3, "img must be [C,H,W]"
        C, H, W = x.shape
        sizes.append((H, W))
        proc_imgs.append(x)
        H_pad, W_pad = H, W



    batch_imgs = torch.zeros(B, 3, H_pad, W_pad, dtype=proc_imgs[0].dtype)
    # batch_masks = torch.zeros(B, 1, H_pad, W_pad, dtype=torch.bool)
    # orig_sizes = torch.zeros(B, 2, dtype=torch.long)

    for i, x in enumerate(proc_imgs):
        _, H, W = x.shape
        batch_imgs[i, :, :H, :W] = x
        # batch_masks[i, 0, :H, :W] = True
        # orig_sizes[i] = torch.tensor([H, W])

    labels = torch.stack(labels).view(B, 1)
    img_types = torch.stack(img_types).view(B, 4)

    return batch_imgs, list(image_rois) ,img_types, labels, list(img_names)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MedImage binary classification test")
    parser.add_argument('--testset', type=str, required=True, help='Test set')
    parser.add_argument("--seed", type=int, default=51, help="Torch seed")
    parser.add_argument("--model", type=str, default="CustomResNetBinary34", help="Selection of the model to train")
    parser.add_argument("--model_weights_path", type=str, default="/bestModels", help="Path to the model weights")
    parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--metrics_run_path", type=str, help="Path to the metrics run folder")
    parser.add_argument("--json_suffix", type=str, default=None, help="Suffix to append to metrics jsonl filename")
    parser.add_argument("--augmentation_config_path", type=str, default="augment_transform.yaml", help="Path to the augmentation config file")
    
    args = parser.parse_args()

    with open(args.augmentation_config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    transformsConfig = config


    channels_list = config.get("channels", [])
    print(f"[test] YAML cargado de: {args.augmentation_config_path}")
    print(f"[test] channels: {channels_list}")

    inchannels = 3 if (isinstance(channels_list, list) and len(channels_list) == 3) else 1
    print(f"[test] in_channels = {inchannels}")

    get_model_metrics(args.testset, args.positive_classes, args.seed, args.model , args.model_weights_path, dataroot=args.dataroot, transformsConfig=transformsConfig,inChannels =inchannels, save_completeMetrics_path=args.metrics_run_path, json_suffix=args.json_suffix, show_image = True, limit = 10000)

