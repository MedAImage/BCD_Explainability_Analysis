import os
import sys
import json
import random
import argparse
from collections import defaultdict


def stratified_split(pacientes_dict):
    ids = list(pacientes_dict.keys())
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(SPLITS["train"] * n_total)
    n_val = int(SPLITS["val"] * n_total)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids

def dataset_from_patients(pacientes_dict, ids):
    d = {}
    for pid in ids:
        d.update(pacientes_dict[pid])
    return d


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset split into training, validation and test.")
    parser.add_argument('--dataset', type=str, required=True, help='Json file with the whole dataset')
    parser.add_argument("--seed", type=int, default=76014, help="Random seed")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--split_root", type=str, default=".", help="Root path to the json checkpoints splits")

    args = parser.parse_args()

    random.seed(args.seed)
    SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
    POSITIVE_CLASSES = args.positive_classes

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    pacientes = defaultdict(dict)
    for k, d in dataset.items():
        paciente_id = k.split("_")[0]
        pacientes[paciente_id][k] = d

    pacientes_positivos = {}
    pacientes_negativos = {}

    for paciente_id, imagenes in pacientes.items():
        es_positivo = False
        for _, datos_img in imagenes.items():
            for clase in POSITIVE_CLASSES:
                if len(datos_img["label"].get(clase, [])) > 0:
                    es_positivo = True
                    break
            if es_positivo:
                break
        if es_positivo:
            pacientes_positivos[paciente_id] = imagenes
        else:
            pacientes_negativos[paciente_id] = imagenes

    train_pos, val_pos, test_pos = stratified_split(pacientes_positivos)
    train_neg, val_neg, test_neg = stratified_split(pacientes_negativos)

    d_training = dataset_from_patients(pacientes_positivos, train_pos)
    d_training.update(dataset_from_patients(pacientes_negativos, train_neg))

    d_validation = dataset_from_patients(pacientes_positivos, val_pos)
    d_validation.update(dataset_from_patients(pacientes_negativos, val_neg))

    d_test = dataset_from_patients(pacientes_positivos, test_pos)
    d_test.update(dataset_from_patients(pacientes_negativos, test_neg))

    suffix = '_'.join(args.positive_classes) + '_' + str(args.seed)

    json_checkpoint_path = os.path.join(args.split_root, 'json_splits')
    if os.path.exists(json_checkpoint_path) is False:
        os.makedirs(json_checkpoint_path)

    lesion_folder_name = '__'.join(args.positive_classes)
    lesion_folder = os.path.join(json_checkpoint_path, lesion_folder_name)
    if not os.path.exists(lesion_folder):
        os.makedirs(lesion_folder)

    seed_folder = os.path.join(lesion_folder, str(args.seed))
    if not os.path.exists(seed_folder):
        os.makedirs(seed_folder)


    with open(os.path.join(seed_folder, f'joined_dataset_training_{suffix}.json'), 'w') as f:
        json.dump(d_training, f, indent=4)

    with open(os.path.join(seed_folder, f'joined_dataset_validation_{suffix}.json'), 'w') as f:
        json.dump(d_validation, f, indent=4)

    with open(os.path.join(seed_folder, f'joined_dataset_test_{suffix}.json'), 'w') as f:
        json.dump(d_test, f, indent=4)

    print("---")
    print(f"Train: {len(d_training)} images")
    print(f"Val:   {len(d_validation)} images")
    print(f"Test:  {len(d_test)} images")