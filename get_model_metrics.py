
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from dataset_load.dataset import lesionDataset, normal_transform
from models.models import  EfficientNetB0, CustomResNetBinary, CustomResNetBinary50, CustomDenseNet, CustomMobileNetV3
import argparse
import os
import json
import yaml
import copy
from scipy.stats import pearsonr, spearmanr
import sys
sys.path.append('./pytorch-grad-cam')
from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.ops import box_iou
from utils.pos_weight_samples import pos_weight_samples
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

def get_shapleyCam_map(cam_model, types, target_layers, inputs):
    with torch.enable_grad():
        cam_model.set_types(types)
        shapleycam_maps = []
        for i, layer in enumerate(target_layers):
            cam_ctx = ShapleyCAM(model=cam_model, target_layers=[layer])
            with cam_ctx as cam:
                grayscale_cam = cam(input_tensor=inputs, targets=[BinaryClassifierOutputTarget(1)])[0]
                shapleycam_maps.append(grayscale_cam)

    return shapleycam_maps

def show_combined_images(images, title):
    concat_img = np.zeros((800,800,3), dtype=np.uint8)
    for i, img in enumerate(images):
        res_img = cv2.resize(img, (400, 400))
        r,c = i//2, i%2
        concat_img[r*400: (r+1)*400, c*400: (c+1)*400] = res_img
    cv2.imshow(title, concat_img)
    k = cv2.waitKey(0)
    return k

def calculate_metrics(l_true, l_predict, threshold = 0.5):
    l_predict_bin = (l_predict>threshold).astype(int)
    precision = precision_score(l_true,l_predict_bin, average="binary", zero_division=0)
    recall = recall_score(l_true,l_predict_bin, average="binary",zero_division=0)
    f1 = f1_score(l_true,l_predict_bin, average="binary",zero_division=0)
    auc_roc = roc_auc_score(l_true,l_predict)
    acc = accuracy_score(l_true,l_predict_bin)

    # print(f"Metrics({'weighted'}): Precision:{precision}||Recall:{recall}|| F1_Score:{f1}|| Accuracy:{acc}")
    return acc, precision, recall, f1, auc_roc


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

def get_roi_energy_fraction(map, ground_truth_mask, th_zero = 0.5):
    norm_map = np.copy(map)
    norm_map[norm_map<th_zero] = 0

    rois_energy = np.sum(norm_map[ground_truth_mask==1])

    total_energy = np.sum(norm_map)
    fR = rois_energy/(total_energy+np.finfo(np.float32).eps)

    return fR

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

def get_map_mass(map):
    map_mass = map.sum()
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

def get_model_metrics(testDataset, positive_classes, loadedseed, modelName, bestModelPth, dataroot='.',   transformsConfig=None, inChannels=None, save_completeMetrics_path=None, json_suffix=None, show_image = False, testDebug = False, limit = 10000, Bias=None):

    seed = torch.manual_seed(loadedseed)
    print(f"Best model path: {bestModelPth}")
    bestmodel = bestModelPth.split("/")[-1]
    print(f"Best model: {bestmodel}")
    testDebugging = testDebug

    NUM_CLASSES = len(positive_classes)
    #SETTING THE ARCHITECTURE OF THE MODEL
    if modelName == "CustomResNetBinary":
        model = CustomResNetBinary()
    elif modelName == "CustomResNetBinary50":
        model = CustomResNetBinary50()
    elif modelName == "EfficientNetB0":
        model = EfficientNetB0()
    elif modelName == "CustomDenseNet":
        model = CustomDenseNet()
    elif modelName == "CustomMobileNetV3":
        model = CustomMobileNetV3()
    else:
        raise ValueError(f"Model {modelName} not recognized. Please choose a valid model.")

    target_layer = [find_last_spatial_layer(model.base_model), model.head.proj, model.head.attn[-1]]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    state_dict = torch.load(bestModelPth, map_location = device)
    model.load_state_dict(state_dict)
    model = model.to(device) 

    cam_model = CamWrapperAdapter(model).to(device)

    normal_data = normal_transform()

    DatasetLesion = lesionDataset(dataPath = testDataset ,positive_classes = positive_classes, transform_with_class = normal_data, transforms_config=transformsConfig, testDebug=testDebugging ,dataroot=dataroot, limit=limit)

    test_dataset = DatasetLesion
    test_labels = np.array(test_dataset.labels)
    print(f"Test Dataset Size: {len(test_dataset)}")
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_test)

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
    map_types = ['contribution', 'attention', 
                 'grad_cam_cnn', 'grad_cam_proj', 'grad_cam_att',
                 'eigen_cam_cnn', 'eigen_cam_proj', 'eigen_cam_att',
                 'shapley_cam_cnn', 'shapley_cam_proj', 'shapley_cam_att']



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
    

    # images_to_save = ["0a3018e7ad1d1d7d2e142c2ca7c518fa_L_CC.png"]
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

            outputs, att_map, contrib_map = model(inputs, types)

            loss = criterion_loss(outputs, labels)

            test_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted_class = (probabilities >= PR_TH).float()

            for img, batch_rois ,label, pred_class, prob, logits in zip(inputs.tolist(), rois ,labels.tolist(), predicted_class.tolist(), probabilities.tolist(), outputs.tolist()):
                # print(f"-True image label: {l} || Label image prediction: {p} || Correct prediction probability: {pr}")
                pr_text = str(int(prob[0]*100)) + '%'

                ####IMAGE CONSTRUCTION####

                cvimg = torch.permute(torch.tensor(img), (1, 2, 0)).cpu().numpy()
                cvimg = (cvimg*255).astype(np.uint8)

                img_size = cvimg.shape
                cvimg_orig = cv2.resize(cvimg[:,:,0], (WIMG, HIMG))
                cvimg = cv2.cvtColor(cvimg_orig, cv2.COLOR_GRAY2BGR)
                cvimg = draw_rois(cvimg, batch_rois, img_size, positive_classes)
                cvimg = cv2.putText(cvimg, pr_text, (50,50), font, 1, (0,255,0), 2)
                base = cvimg_orig.astype(np.float32) / 255.0
                base = np.stack([base, base, base], axis=-1)


                cv_contrib_map = torch.permute(contrib_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach().numpy()
                cv_contrib_map_norm, grayscale_contrib_map = normalize_map(cv_contrib_map)
                map_results['contribution']['map'] = cv_contrib_map
                map_results['contribution']['norm_map'] = cv_contrib_map_norm
                map_results['contribution']['gray_map'] = grayscale_contrib_map
                
                cv_att_map = torch.permute(att_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach().numpy()
                cv_att_map_norm, grayscale_att_map = normalize_map(cv_att_map)
                map_results['attention']['map'] = cv_att_map
                map_results['attention']['norm_map'] = cv_att_map_norm
                map_results['attention']['gray_map'] = grayscale_att_map
                

                # results from EIGEN-CAM
                eigen_cam_types = ['eigen_cam_cnn', 'eigen_cam_proj', 'eigen_cam_att']
                sal_maps_eigencam = get_eigenCam_map(cam_model, types, target_layer, inputs)
                for imap, tmap in enumerate(eigen_cam_types):
                    norm_map, gray_map = normalize_map(sal_maps_eigencam[imap])
                    map_results[tmap]['map'] = sal_maps_eigencam[imap]
                    map_results[tmap]['norm_map'] = norm_map
                    map_results[tmap]['gray_map'] = gray_map


                # results from GRAD-CAM++
                grad_cam_types = ['grad_cam_cnn', 'grad_cam_proj', 'grad_cam_att']
                sal_maps_gradcam = get_gradCam_map(cam_model, types, target_layer, inputs)
                for imap, tmap in enumerate(grad_cam_types):
                    norm_map, gray_map = normalize_map(sal_maps_gradcam[imap])
                    map_results[tmap]['map'] = sal_maps_gradcam[imap]
                    map_results[tmap]['norm_map'] = norm_map
                    map_results[tmap]['gray_map'] = gray_map

                # results from SHAPLEY-CAM
                shapley_cam_types = ['shapley_cam_cnn', 'shapley_cam_proj', 'shapley_cam_att']
                sal_maps_shapleycam = get_shapleyCam_map(cam_model, types, target_layer, inputs)
                for imap, tmap in enumerate(shapley_cam_types):
                    norm_map, gray_map = normalize_map(sal_maps_shapleycam[imap])
                    map_results[tmap]['map'] = sal_maps_shapleycam[imap]
                    map_results[tmap]['norm_map'] = norm_map
                    map_results[tmap]['gray_map'] = gray_map


                for mtype in map_types:
                    map_mass = get_map_mass(map_results[mtype]['map'])
                    map_results[mtype]['map_mass'].append(map_mass)                            

                ground_truth_mask = get_ground_truth_mask(img_size, img_size,batch_rois[positive_classes[0]])
                contrib_weights = {th: 1 for th in LContrib_Th}                    
                if len(batch_rois[positive_classes[0]])>0:
                    pos_classes_predictions.append(prob[0])
                    for mtype in map_types:
                        resized_map = cv2.resize(map_results[mtype]['norm_map'], (img_size[1], img_size[0]))                    
                        for th in LContrib_Th:
                            map_results[mtype]['energy'][th].append(get_roi_energy_fraction(resized_map, ground_truth_mask, th_zero=th))
                        map_results[mtype]['PG_1-top'].append(pointing_game_topK(resized_map, batch_rois[positive_classes[0]], img_size, K = 1))                            
                        map_results[mtype]['PG_5-top'].append(pointing_game_topK(resized_map, batch_rois[positive_classes[0]], img_size, K = 5))
                    for th in LContrib_Th:
                        contrib_weights[th] = map_results['contribution']['energy'][th][-1]
                if show_image:
                    wpr_text = str(int(prob[0]*contrib_weights[0.5]*100)) + '%'
                    maps_to_show = ['contribution', 'eigen_cam_cnn', 'grad_cam_cnn', 'shapley_cam_cnn']
                    maps_to_save = ['contribution', 'attention', 'grad_cam_cnn', 'eigen_cam_cnn', 'shapley_cam_cnn']
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
                    k = show_combined_images(vis_maps, title)
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
    parser.add_argument("--model", type=str, default="CustomResNetBinary50", help="Selection of the model to test")
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

