import torch
import numpy as np



def pos_weight_samples(train_labels, device):
    
    class_dist = np.sum(train_labels, axis=0)
    if class_dist[0] > 0:
        num_negative_samples = np.sum(np.all(train_labels == 0, axis=1))  
        num_positive_samples = len(train_labels) - num_negative_samples  
        pos_weight = torch.tensor([num_negative_samples / num_positive_samples], dtype=torch.float32).to(device)
    else:
        print("Warning: No positive samples in the training set. Setting pos_weight to 1.0.")
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    return pos_weight