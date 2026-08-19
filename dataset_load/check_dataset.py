import dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import yaml
import argparse



def draw_rois(image, rois, size, classes, color = (0,255,0)):
    HIMG = image.shape[0]    
    WIMG = image.shape[1]
    for k, r_list in rois.items():
        if k in classes:
            for r in r_list:
                p1 = (int(r[0]*WIMG/size[1]), int(r[1]*HIMG/size[0]))
                p2 = (int((r[0]+r[2])*WIMG/size[1]), int((r[1]+r[3])*HIMG/size[0]))
                image = cv2.rectangle(image, p1, p2, color, 2)

    return image

parser = argparse.ArgumentParser(description="Check dataset")
parser.add_argument('--dataset', type=str, required=True, help='Datasetset')
parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
parser.add_argument("--augmentation_config_path", type=str, default="augment_transform.yaml", help="Path to the augmentation config file")

args = parser.parse_args()

data_root = args.dataroot

TestDatasetFile =args.dataset


classes = args.positive_classes

with open(args.augmentation_config_path, 'r') as file:
    config = yaml.safe_load(file)

transformsConfig = config#["transformations"]

DatasetDicom = dataset.lesionDataset(dataPath = str(TestDatasetFile), positive_classes = classes, transform_with_class = dataset.normal_transform(), 
                                    seed = 666, transforms_config=transformsConfig ,dataroot=data_root, limit = 100)

data_loader = DataLoader(DatasetDicom, batch_size=1, shuffle=False, num_workers=1, collate_fn = dataset.collate_pad_to32)

for img, rois, imgtype, label, img_name in data_loader:  
    print(img_name)
    cvimg = np.array(torch.permute(img.squeeze(dim=0), (1, 2, 0)))
    cvimg = (cvimg*255).astype(np.uint8)
    for channel in range(cvimg.shape[2]):
        channel_img = cvimg[:,:,channel]
        channel_img = cv2.cvtColor(channel_img, cv2.COLOR_GRAY2BGR)
        draw_rois(channel_img, rois[0], channel_img.shape, classes=classes[0])
        cv2.imshow('image channel '+str(channel), channel_img)

    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break


