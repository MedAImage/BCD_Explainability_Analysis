import subprocess
import re
import argparse
import os

PROJECT_DIRPATH = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument("--positive_class", type=str, required=True,
                    help="Name of the positive_class")
parser.add_argument("--device", type=str, required=True,
                    help="Device (cuda:0, cuda:1)")
args = parser.parse_args()

models = [
    # "CustomDenseNet",
    # "CustomResNetBinary",
    # "CustomResNetBinary50",
    # "EfficientNetB0",
    "CustomMobileNetV3"
]


seeds = [
    17143,
    # 67291,    
    #88078,
    #51,
    #666
]


positive_class = args.positive_class 

# if positive_class=="Nodulo":
#     augment_files = ["augment_transform_copy_copy_copy.yaml",
#                      "augment_transform_copy_copy_clahe.yaml",
#                      "augment_transform_copy_clahe_enhance.yaml",
#                      "augment_transform_expand_flip.yaml",
#                      "augment_transform_copy_clahe_enhance_AUGM.yaml"]
#     json_suffix = ["copy_copy_copy",
#                     "copy_copy_clahe",
#                     "copy_clahe_enhance",
#                     "expand_flip",
#                     "copy_clahe_enhance_AUGM"]
# elif positive_class=="Microcalcificaciones":
#     augment_files = ["augment_transform_copy_copy_copy.yaml",
#                      "augment_transform_copy_copy_clahe.yaml",
#                      "augment_transform_copy_clahe_tophat.yaml",
#                      "augment_transform_expand_flip.yaml",
#                      "augment_transform_copy_clahe_tophat_AUGM.yaml"]
#     json_suffix = ["copy_copy_copy",
#                     "copy_copy_clahe",
#                     "copy_clahe_tophat",
#                     "expand_flip",
#                     "copy_clahe_tophat_AUGM"]
if positive_class=="Microcalcificaciones":
    augment_files = ["augment_transform_copy_clahe_topHat.yaml",
                    "augment_transform_copy_clahe_topHat5x5.yaml"]
    json_suffix = [ "copy_clahe_topHat",
                    "copy_clahe_topHat5x5"]
    
    
else:
    print("Wrong positive class", positive_class)
    exit()

device = args.device
if device not in ["cuda:0", "cuda:1"]:
    print("Wrong device", device)

for seed in seeds:
    for i in range(5):
        k = 'K'+str(i+1)

        trainset = f'data_organization/json_splits/{positive_class}/76014/{k}/joined_dataset_training_{positive_class}_76014.json'
        valset = f'data_organization/json_splits/{positive_class}/76014/{k}/joined_dataset_validation_{positive_class}_76014.json'
        testset = f'data_organization/json_splits/{positive_class}/76014/{k}/joined_dataset_test_{positive_class}_76014.json'


        match = re.search(r'_(\d+)\.json$', trainset)
        seed_split = match.group(1) if match else None



        for model in models:
            for suffix, augm_file in zip(json_suffix, augment_files):
                cmd = [
                    "python3", "train.py",
                    "--number_of_epochs", "10",
                    "--patience", "1",
                    "--model", model,
                    "--batch_size", "8",
                    "--seed", str(seed),
                    "--seed_split", str(seed_split),
                    "--trainset", trainset,
                    "--valset", valset,
                    "--testset", testset,
                    "--positive_classes", positive_class,
                    "--dataroot", PROJECT_DIRPATH,
                    "--augmentation_config_path", augm_file,
                    "--device", device
                ]

                cmd += ["--json_suffix", suffix +'_'+k]

                print(f"Running: {' '.join(cmd)}\n")
                    
                process = subprocess.run(
                    cmd, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    check=True
                )
