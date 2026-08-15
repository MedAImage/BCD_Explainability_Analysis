import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from models.base_models_final import  EfficientNetB0, CustomResNetBinary,  CustomResNetBinary50,  CustomDenseNet, CustomMobileNetV3
import numpy as np
import matplotlib.pyplot as plt
from dataset_load.dataset import lesionDataset, data_augmentation_transform, normal_transform, collate_pad_to32
from get_model_metrics import calculate_metrics
import os
import argparse
import yaml
from tqdm import tqdm
import random
from utils.pos_weight_samples import pos_weight_samples

probability_threshold = 0.5
def parse_args(parser):
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if arg_dict.get(key) in [None, parser.get_default(key)]:
                arg_dict[key] = value
    return args

def load_yaml_config(file_path):
    if not file_path or file_path is None:
        return None
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Convert logits to probabilities
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)

        # Apply focal loss formula
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


##WEIGHT HE INITIALIZATION
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# METHO TO FIX THE SEEDS
def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###METHOD FOR TRAINING###
def train(epoch = 0):
    model.train()
    running_loss = 0.0

    #AVOIDING POSSIBLE COMBINATIONS IN FIRST EPOCH
    if epoch !=0 and datasetshuffled == True:
        print(f"Suffling dataset...")
        train_data_loader.dataset.shuffleDataset()
    for inputs, _,types, labels, _ in train_data_loader:
        if epoch == 0 and running_loss == 0.0:
            input_hash = batch_hash(inputs)
            label_hash = batch_hash(labels)
            print(f"Hash del primer batch de inputs: {input_hash}")
            print(f"Hash del primer batch de labels: {label_hash}")
        #SENDING INPUT AND LABELS TO GPU
        inputs, labels = inputs.to(device), labels.to(device)
        types = types.to(device)
        #RESETTING GRADIANTS
        optimizer.zero_grad()
        outputs, _, _ = model(inputs, types) #ignore maps
        # outpushash = batch_hash(outputs[0])
        # print(f"Hash del primer batch de outputs: {outpushash}")
        loss = criterion_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_data_loader)
    return avg_train_loss


###METHOD FOR VALIDATION###
def validation (epoch = 0):
    model.eval()
    val_labels=[]
    val_predictions=[]
    val_loss = 0.0

    with torch.no_grad():
        for inputs, _,types, labels, _ in val_data_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            types = types.to(device)
            outputs, _, _= model(inputs, types) #ignore maps
            probabilities = torch.sigmoid(outputs)
            loss = criterion_loss(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            predicted_class = (probabilities >= probability_threshold ).float()
            val_labels.append(labels.cpu().numpy())
            val_predictions.append(predicted_class.cpu().numpy())
        avgValLoss = val_loss / len(val_data_loader.dataset)
        return avgValLoss, val_labels, val_predictions

###METHOD FOR METRICS###
def metrics(labels, predictions):
    acc, precision, recall, f1, auc_roc = calculate_metrics(labels, predictions, threshold=probability_threshold)
    print(f"Precision:{precision:.4f} || Recall:{recall:.4f} || F1 Score:{f1:.4f} || AUC-ROC:{auc_roc:.4f} || Accuracy:{acc:.4f}")

    return acc, precision, recall, f1, auc_roc 



if __name__=="__main__":
    #ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="MedImage binarty classification training")
    parser.add_argument("--config_file", type=str, default=None, help="Path to a config file")
    parser.add_argument('--trainset', type=str, required=True, help='Training set')
    parser.add_argument('--valset', type=str, required=True, help='Validation set')
    parser.add_argument("--seed", type=int, default=51, help="Seed torch for training")
    parser.add_argument("--seed_split", type=str, default=None, help="Seed split for saving the model")
    parser.add_argument("--number_of_epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for improvement")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--shuffle_dataset", type=bool, default=False, help="Shuffle the dataset on each epoch")
    parser.add_argument("--model", type=str, default="CustomResNetBinary", help="Selection of the model to train")
    parser.add_argument("--model_weights_path", type=str, default="/bestModels", help="Path to the model weights")
    parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., cuda:1, cpu)")
    parser.add_argument("--model_save_path", type=str, default="bestModels/", help="Path to save the best model")
    parser.add_argument("--metrics_run_path", type=str, default="Metrics_runs/", help="Path to save metrics run for test model")
    parser.add_argument("--augmentation_config_path", type=str, default="augment_transform.yaml", help="Path to the augmentation config file")
    parser.add_argument("--json_suffix", type=str, default=None, help="Suffix to append to metrics jsonl filename")
    parser.add_argument("--bias", type=float, default=None, help="MLP bias")    
    # args = parse_args(parser)
    args = parser.parse_args()
    print(args)

    #CONFIGURATION WITH ARGUMENTS
    EPOCHS = args.number_of_epochs
    PATIENCE = args.patience
    batchSamples = args.batch_size
    numChannels = 3
    numtypes = 4
    learning_rate = args.learning_rate
    shuffleDataset = args.shuffle_dataset
    model_weights_pth = args.model_weights_path
    positive_classes = args.positive_classes
    bias = args.bias
    print(f"Probability threshold for classification: {probability_threshold}")
    fix_seed(args.seed)

    # LOAD YAML CONFIG (usar solo channels)
    config = load_yaml_config(args.augmentation_config_path) or {}
    transformsConfig = config  # pasamos el dict al dataset
    testDebug = bool(config.get("testDebug", False))

    channels_list = config.get("channels", [])
    print(f"[train] YAML cargado de: {args.augmentation_config_path}")
    print(f"[train] channels: {channels_list}")


    assert isinstance(channels_list, list) and len(channels_list) == 3, 'The number of channels should be 3. Review your configuration file'

    inchannels = 3 
    
    NUM_CLASSES = len(positive_classes)
    #SETTING THE ARCHITECTURE OF THE MODEL
    if args.model == "CustomResNetBinary":
        model = CustomResNetBinary(num_classes=NUM_CLASSES, in_channels=inchannels)
    elif args.model == "CustomResNetBinary50":
        model = CustomResNetBinary50(num_classes=NUM_CLASSES, in_channels=inchannels, Bias=bias)
    elif args.model == "EfficientNetB0":
        model = EfficientNetB0(num_classes=NUM_CLASSES, in_channels=inchannels)
    elif args.model == "CustomDenseNet":
        model = CustomDenseNet(num_classes=NUM_CLASSES, in_channels=inchannels)
    elif args.model == "CustomMobileNetV3":
        model = CustomMobileNetV3(num_classes=NUM_CLASSES, in_channels=inchannels)

    else:
        raise ValueError(f"Model {args.model} not recognized. Please choose a valid model.")

    fix_seed(args.seed)
    
    #SAVING PATHS CONSTRUCTION
    save_root_path = args.model_save_path
    save_seed_path = os.path.join(save_root_path, args.seed_split)
    if not os.path.exists(save_seed_path):
        os.makedirs(save_seed_path)
    positive_classes_str = '__'.join(args.positive_classes)
    save_completeBestModel_path = os.path.join(save_seed_path, positive_classes_str)
    if not os.path.exists(save_completeBestModel_path):
        os.makedirs(save_completeBestModel_path)

    save_rootMetrics_path = args.metrics_run_path
    if not os.path.exists(save_rootMetrics_path):
        os.makedirs(save_rootMetrics_path)
    save_metricsSeed_path = os.path.join(save_rootMetrics_path, args.seed_split)
    if not os.path.exists(save_metricsSeed_path):
        os.makedirs(save_metricsSeed_path)
    save_completeMetrics_path = os.path.join(save_metricsSeed_path, positive_classes_str)
    if not os.path.exists(save_completeMetrics_path):
        os.makedirs(save_completeMetrics_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device) 
    
    #LOADING THE DATASET:
    TrainDatasetJson = args.trainset
    ValDatasetJson = args.valset
    print(f"[DEBUG] train.py | dataroot: {args.dataroot}")
    print(f"[DEBUG] train.py | TrainDatasetJson: {TrainDatasetJson}")
    print(f"[DEBUG] train.py | ValDatasetJson: {ValDatasetJson}")

    data_augmentation = data_augmentation_transform()
    normal_data = normal_transform()

    train_dataset = lesionDataset(dataPath = TrainDatasetJson, positive_classes = positive_classes, transform_with_class=data_augmentation, shuffleTrain=shuffleDataset, seed = args.seed,transforms_config=transformsConfig ,dataroot=args.dataroot, withLTimeAugmentation=True)
    if train_dataset.shuffleTrain == True:
        print("Dataset is shuffled")
        datasetshuffled = True
    else:
        print("Dataset is not shuffled")
        datasetshuffled = False

    val_dataset = lesionDataset(dataPath = ValDatasetJson, positive_classes = positive_classes, transform_with_class = normal_data, seed = args.seed, transforms_config=transformsConfig ,dataroot=args.dataroot)
    #LOADING THE DATA LOADERS AND LABELS
    train_data_loader = DataLoader(train_dataset, batch_size=batchSamples, shuffle=True, num_workers=4, collate_fn = collate_pad_to32)
    val_data_loader = DataLoader(val_dataset, batch_size=batchSamples, shuffle=False, num_workers=4, collate_fn = collate_pad_to32)

    train_labels = np.array(train_dataset.labels)
    val_labels = np.array(val_dataset.labels)

    if(args.shuffle_dataset == False):
        print("Dataset is not shuffled")    
        pos_weight = pos_weight_samples(train_labels, device)
        criterion_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight = pos_weight)
        
    #CHOOSING CRITERION AND OPTIMIZER
    else:
        print("Dataset is shuffled")
        criterion_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        
    train_loss = []
    validation_loss =[]
    loss_increase_steps = 0
    best_loss = np.inf

    #TRAINING/VALIDATION LOOP
    for epoch in tqdm(range(EPOCHS)):
        trainloss = train(epoch)
        avgValLoss, val_labels, val_predictions = validation(epoch)
        train_loss.append(trainloss)
        validation_loss.append(avgValLoss)
        val_labels = np.concatenate(val_labels, axis = 0)
        val_predictions = np.concatenate(val_predictions, axis = 0)
        print(f"Epoch [{epoch+1}/{EPOCHS}]||Train Loss: {trainloss:.4f}||Validation Loss: {avgValLoss:.4f}\n")
        #CALCULATING  VALIDATION METRICS|| UNCOMMENT FOR TRACKING METRICS DONT ERASE
        print("VAL-METRICS:")
        # Revisa las etiquetas verdaderas
        print("Classes distribution in val_labels:", np.unique(val_labels, return_counts=True))
        # Revisa las predicciones realizadas por el modelo
        print("Classes distribution in val_predictions:", np.unique(val_predictions >= 0.5, return_counts=True))
        acc, precision, recall, f1, auc_roc  = metrics(val_labels, val_predictions)
        #UPDATING THE BEST LOSS
        if avgValLoss < best_loss:
            best_loss = avgValLoss
            save_name = os.path.join(save_completeBestModel_path, args.model)
            if args.json_suffix:
                save_route = f'{save_name}_seed_{args.seed}_{args.json_suffix}.pth'
            else:
                save_route = f'{save_name}_seed_{args.seed}.pth'
            print(save_name)
            print("Saving the model...")
            torch.save(model.state_dict(), f'{save_route}')
            loss_increase_steps = 0
        else:
            loss_increase_steps += 1
            print("Not saving. Waiting for the model to improve with patience", loss_increase_steps)
            if loss_increase_steps>PATIENCE:
                break

