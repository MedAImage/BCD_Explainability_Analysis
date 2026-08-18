import os
import json
import random
import argparse
from collections import defaultdict
from train_val_test_split import dataset_from_patients



def kfold_split(pacientes_dict, N = 5):
    kfold = []
    ids = list(pacientes_dict.keys())
    random.shuffle(ids)
    ksize = len(ids)//N
    for i in range(N-1):
        kfold.append(ids[i*ksize:(i+1)*ksize])
    kfold.append(ids[(N-1)*ksize:])

    return kfold
    
def create_kfold_train_val_test(kfold_pos, kfold_neg, pacientes_positivos, pacientes_negativos):
    train_sets = []
    val_sets = []
    test_sets = []

    N = len(kfold_pos)
    s_list = range(N)

    for k in range(N):

        train_pos = [kfold_pos[s][i] for s in s_list if s!=k for i in range(len(kfold_pos[s]))]
        train_neg = [kfold_neg[s][i] for s in s_list if s!=k for i in range(len(kfold_neg[s]))]
        sizetrain_pos = len(train_pos)
        sizetrain_neg = len(train_neg)
        ini_val_pos = int(sizetrain_pos*0.9)
        ini_val_neg = int(sizetrain_neg*0.9)
        val_pos = train_pos[ini_val_pos:]
        val_neg = train_neg[ini_val_neg:]
        train_pos = train_pos[:ini_val_pos]
        train_neg = train_neg[:ini_val_neg]
        test_pos = kfold_pos[k]
        test_neg = kfold_neg[k]

        d_training = dataset_from_patients(pacientes_positivos, train_pos)
        d_training.update(dataset_from_patients(pacientes_negativos, train_neg))
        d_validation = dataset_from_patients(pacientes_positivos, val_pos)
        d_validation.update(dataset_from_patients(pacientes_negativos, val_neg))
        d_test = dataset_from_patients(pacientes_positivos, test_pos)
        d_test.update(dataset_from_patients(pacientes_negativos, test_neg))
        train_sets.append(d_training)
        val_sets.append(d_validation)
        test_sets.append(d_test)

    return train_sets, val_sets, test_sets



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset split into training, validation and test.")
    parser.add_argument('--dataset', type=str, required=True, help='Json file with the whole dataset')
    parser.add_argument("--seed", type=int, default=76014, help="Random seed")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--split_root", type=str, default=".", help="Root path to the json checkpoints splits")

    args = parser.parse_args()

    random.seed(args.seed)
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


    kfold_pos = kfold_split(pacientes_positivos)
    kfold_neg = kfold_split(pacientes_negativos)

    train_sets, val_sets, test_sets = create_kfold_train_val_test(kfold_pos, kfold_neg, pacientes_positivos, pacientes_negativos)

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

    N_fold = len(train_sets)

    for k in range(N_fold):
        k_folder = os.path.join(seed_folder, 'K'+str(k+1))
        if not os.path.exists(k_folder):
            os.makedirs(k_folder)

        with open(os.path.join(k_folder, f'joined_dataset_training_{suffix}.json'), 'w') as f:
            json.dump(train_sets[k], f, indent=4)

        with open(os.path.join(k_folder, f'joined_dataset_validation_{suffix}.json'), 'w') as f:
            json.dump(val_sets[k], f, indent=4)

        with open(os.path.join(k_folder, f'joined_dataset_test_{suffix}.json'), 'w') as f:
            json.dump(test_sets[k], f, indent=4)

        print("--- ", k+1, " ---")
        print(f"Train: {len(train_sets[k])} images")
        print(f"Val:   {len(val_sets[k])} images")
        print(f"Test:  {len(test_sets[k])} images")