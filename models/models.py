import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, in_ch: int, proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_dim, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(proj_dim, attn_hidden, 1),
            nn.Tanh(),
            nn.Conv2d(attn_hidden, 1, 1),
        )
        self.cls = nn.Linear(proj_dim, 1)

    def forward(self, feat: torch.Tensor):
        A = self.proj(feat)                    # [B, D, H', W']

        w = self.attn(A)                       # [B, 1, H', W']
        alpha = F.softmax(w.flatten(2), dim=2) # [B, 1, N]
        attn_map = alpha.view_as(w)            # [B, 1, H', W']

        Aflat = A.flatten(2).transpose(1, 2)   # [B, N, D]
        z = torch.bmm(alpha, Aflat).squeeze(1) # [B, D]
        logit = self.cls(z).unsqueeze(1)       # [B, 1, 1]
        logit = logit.squeeze(dim=2)           # [B, 1]

        w_cls = self.cls.weight
        w_reshape = w_cls.view(1, -1, 1, 1)          # [1, D, 1, 1]
        b_cls = self.cls.bias

        score_map = (A * w_reshape).sum(dim=1, keepdim=True) + b_cls  # [B, 1, H', W']

        # Contribution map: alpha(i) * score(i)
        contribution_map = attn_map * score_map              # [B, 1, H', W']
        contribution_map = F.relu(contribution_map)


        return logit, attn_map, contribution_map



class CustomResNetBinary(nn.Module):
    def __init__(self,  proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model = nn.Sequential(*list(net.children())[:-2])

        self.head = AttentionHead(512, proj_dim=proj_dim, attn_hidden=attn_hidden)
    def forward(self, x):
        """
        x: [B, 3, H, W] 
        """
        feat = self.base_model(x)
        logit, map_att, map_cont = self.head(feat)
        return logit, map_att, map_cont


class CustomResNetBinary50(nn.Module):
    def __init__(self, proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.base_model = nn.Sequential(*list(net.children())[:-2])  

        self.head = AttentionHead(2048, proj_dim=proj_dim, attn_hidden=attn_hidden)

    def forward(self, x):
        """
        x: [B, 3, H, W] 
        """
        feat = self.base_model(x)                
        logit, map_att, map_cont = self.head(feat)          
        return logit, map_att, map_cont



class CustomDenseNet(nn.Module):
    def __init__(self, proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        self.base_model = net.features
        feature_dim = net.classifier.in_features

        self.head = AttentionHead(feature_dim, proj_dim=proj_dim, attn_hidden=attn_hidden)

    def forward(self, x):
        features = self.base_model(x)
        feat_map = F.relu(features)
        logit, map_att, map_cont = self.head(feat_map)
        return logit, map_att, map_cont


class CustomMobileNetV3(nn.Module):
    def __init__(self, proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.base_model = net.features
        feature_dim = net.classifier[0].in_features
        
        self.head = AttentionHead(feature_dim, proj_dim=proj_dim, attn_hidden=attn_hidden)

    def forward(self, x):
        feat_map = self.base_model(x)

        logit, map_att, map_cont = self.head(feat_map)
        return logit, map_att, map_cont


class EfficientNetB0(nn.Module):
    def __init__(self, proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model = net.features

        features_dim = net.classifier[1].in_features

        self.head = AttentionHead(features_dim, proj_dim=proj_dim, attn_hidden=attn_hidden)

    def forward(self, x):
        feat_map = self.base_model(x)

        logit, map_att, map_cont = self.head(feat_map)
        return logit, map_att, map_cont
