import pandas as pd
import pydicom
import numpy as np
import os
import json
import shutil
import argparse
import shutil


def export_dicoms(dataset_json, vindr_findings, dataset_dir, vindr_dir):
    with open(dataset_json, 'r') as f:
       dataset = json.load(f)

    findings_df = pd.read_csv(vindr_findings)

    for d in dataset:
        data_fields = d.split('.dcm')[0].split('_')
        patient_id = data_fields[0]
        laterality = data_fields[1]
        view_pos = data_fields[2]        
        vindr_view = view_pos if view_pos=='CC' else view_pos+"O"

        patient_dir = os.path.join(dataset_dir, patient_id)

        if not os.path.exists(patient_dir):
            os.mkdir(patient_dir)

        vindr_patient = findings_df[(findings_df['study_id']==patient_id)]
        vindr_patient_dir = os.path.join(vindr_dir, patient_id)
        if not vindr_patient.empty:
            vindr_img_id = vindr_patient[(vindr_patient['laterality']==laterality)
                                        & (vindr_patient['view_position']==vindr_view)]
            
            vindr_img = os.path.join(vindr_patient_dir, vindr_img_id.iloc[0]['image_id']+'.dicom')
            our_img = os.path.join(patient_dir, d)

            vindr_img_exists = os.path.exists(vindr_img)
            our_img_exists = os.path.exists(our_img)

            print('our img', our_img, our_img_exists)
            print('vindr img', vindr_img, vindr_img_exists)
            if not vindr_img_exists:
                print("ERROR: vindr_img not found")
                exit()

            if not our_img_exists:
                shutil.copyfile(vindr_img, our_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Vindr DICOMs")
    parser.add_argument('--dataset', type=str, required=True, help='Json file containing the dataset')
    parser.add_argument("--dicomsdir", type=str, required=True, help="Dataset directory (DICOMs)")
    parser.add_argument("--vindrcsv", type=str, required=True, help="CSV file with the Vindr findings")
    parser.add_argument("--vindrdir", type=str, required=True, help="Vindr directory (DICOMs)")

    args = parser.parse_args()
    export_dicoms(args.dataset, args.vindrcsv, args.dicomsdir, args.vindrdir)
