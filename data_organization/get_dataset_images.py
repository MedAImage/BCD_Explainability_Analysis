import pandas as pd
# import pydicom
# import numpy as np
import os
import json
import shutil
import argparse
from dicom2png import dicom_to_png


def get_vindr_files(vindr_findings, vindr_dir):
    vindr_image_files = {}
    findings_df = pd.read_csv(vindr_findings)
    for _, sample in findings_df.iterrows():
        patient_id = sample['study_id']
        laterality = sample['laterality']
        view_vindr = sample['view_position']
        view = view_vindr if view_vindr=='CC' else 'ML'
        vindr_patient_dir = os.path.join(vindr_dir, patient_id)
        joined_dataset_id = patient_id+'_'+laterality+'_'+view+'.dcm'
        vindr_image_files[joined_dataset_id] = os.path.join(vindr_patient_dir, sample['image_id']+'.dicom')
    return vindr_image_files

def get_inbreast_files(inbreast_dir):
    inbreast_image_files = {}
    image_files = [f for f in os.listdir(inbreast_dir) if os.path.isfile(os.path.join(inbreast_dir, f)) and f.endswith('.dcm')]
    for f in image_files:
        fields = f.split('_')
        patient_id = fields[1]
        laterality = fields[3]
        view = fields[4]
        joined_dataset_id = patient_id+'_'+laterality+'_'+view+'.dcm'
        inbreast_image_files[joined_dataset_id] = os.path.join(inbreast_dir,f)

    return inbreast_image_files

def get_all_image_files(dataset_json, vindr_findings, vindr_dir, inbreast_dir, output_dir):
    vindr_image_files = get_vindr_files(vindr_findings, vindr_dir)        
    inbreast_image_files = get_inbreast_files(inbreast_dir)

    with open(dataset_json, 'r') as f:
       dataset = json.load(f)

    dataset_images_dir = os.path.join(output_dir, 'outDat')

    os.makedirs(dataset_images_dir, exist_ok=True)

    for d in dataset:
        if 'V2' in d or 'EXP0' in d or 'EXP1' in d:
            continue
        output_file = os.path.join(dataset_images_dir, d.replace('.dcm', '.png'))        
        if d in vindr_image_files.keys():
            dicom_img = vindr_image_files[d]
        elif d in inbreast_image_files.keys():
            dicom_img = inbreast_image_files[d]
        else:
            print(f'sample {d} not found')
            continue
        dicom_to_png(dicom_img, output_file)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get dataset images from Vindr and Inbreast")
    parser.add_argument('--dataset', type=str, required=True, help='Json file containing the dataset')
    parser.add_argument("--outputdir", type=str, required=True, help="Output directory")
    parser.add_argument("--vindrcsv", type=str, required=True, help="CSV file with the Vindr findings")
    parser.add_argument("--vindrdir", type=str, required=True, help="Vindr directory (DICOMs)")
    parser.add_argument("--inbreastdir", type=str, required=True, help="Inbreast directory (DICOMs)")


    args = parser.parse_args()
    get_all_image_files(args.dataset, args.vindrcsv, args.vindrdir, args.inbreastdir, args.outputdir)
