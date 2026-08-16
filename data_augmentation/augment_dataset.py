import os
import json
import argparse
from flipImages import flip_images
from expandImages import expand_images

def get_images_versions(jsonpath):
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

    images_versions = {}
    for an in annotations:
        name = an.replace('.dcm','')
        name = name.replace('.png','')
        name_parts = name.split('_') 
        orig_name = '_'.join([name_parts[0], name_parts[-2], name_parts[-1]])

        if orig_name not in images_versions.keys():
            images_versions[orig_name] = []
        images_versions[orig_name].append(name)  

    return images_versions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset with flipped and expanded versions of the images")
    parser.add_argument('--dataset', type=str, required=True, help='JSON file containing the dataset')
    parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")

    args = parser.parse_args()

    images_versions = get_images_versions(args.dataset)

    json_file_changed = flip_images(args.dataset, args.dataroot, images_versions)

    if json_file_changed:
        images_versions = get_images_versions(args.dataset)
        print('json file changed')
    else:
        print('json file did not change')        

    json_file_changed = expand_images(args.dataset, args.dataroot, images_versions)
    if json_file_changed:
        print('json file changed')
    else:
        print('json file did not change')        
    