import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Literal

################### UTILITY FUNCTIONS FOR MODEL MODIFICATION ###################
# ALLOWS TO GET THE MODULES OF A MODEL BY NAME  
def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """
    Replaces a submodule within a PyTorch model using its dot-separated name.

    """
    
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def adapt_first_conv_to_n_channels(model: nn.Module, in_channels: int, copy_weights = True) -> nn.Module:
    """
    Modifies the first Conv2d layer of a model to accept a new number of input channels.

    Args:
        model (nn.Module): The model to modify.
        in_channels (int): The new number of input channels.
        copy_weights (bool): Whether to copy and adapt existing weights. Defaults to True.

    Returns:
        nn.Module: The modified model.

    Raises:
        RuntimeError: If no Conv2d layer is found in the model.
    """
    
    first_name, first_conv = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            first_name, first_conv = name, m
            break
    if first_conv is None:
        raise RuntimeError("Conv2d layer not found in the model")
    bias = (first_conv.bias is not None)
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=1, 
        bias=bias,
        padding_mode=first_conv.padding_mode,
    )

    if copy_weights:
        with torch.no_grad():
            w_old = first_conv.weight  
            in_c_old = w_old.shape[1]
            
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

            
            n_copy = min(in_c_old, in_channels)
            new_conv.weight[:, :n_copy] = w_old[:, :n_copy]

            
            if in_channels > in_c_old:
                mean_w = w_old.mean(dim=1, keepdim=True) 
                for c in range(in_c_old, in_channels):
                    new_conv.weight[:, c:c+1] = mean_w

            if in_channels < in_c_old:
                mean_w = w_old.mean(dim=1, keepdim=True)  
                for c in range(in_channels):
                    new_conv.weight[:, c:c+1] = mean_w

    set_module_by_name(model, first_name, new_conv)
    return model




class AttentionMILHead(nn.Module):
    """
    1x1 projection -> attention weights over HxW -> weighted sum -> linear logit.

    Returns:
      logit: [B, 1]
      importance_map: [B, 1, H', W']  (class-specific, no gradients needed)
    """
    def __init__(self, in_ch: int, proj_dim: int = 512, attn_hidden: int = 256, Bias = None):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_dim, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(proj_dim, attn_hidden, 1),
            nn.Tanh(),
            nn.Conv2d(attn_hidden, 1, 1),
        )
        self.cls = nn.Linear(proj_dim, 1)
        if Bias is not None:
            self.cls.bias.requires_grad = False
            nn.init.constant_(self.cls.bias, Bias)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial features for pooling
        A = self.proj(feat)                    # [B, D, H', W']

        # Attention logits -> spatial softmax
        w = self.attn(A)                       # [B, 1, H', W']
        alpha = F.softmax(w.flatten(2), dim=2) # [B, 1, N]
        attn_map = alpha.view_as(w)            # [B, 1, H', W']

        # Pool
        Aflat = A.flatten(2).transpose(1, 2)   # [B, N, D]
        z = torch.bmm(alpha, Aflat).squeeze(1) # [B, D]
        logit = self.cls(z).unsqueeze(1)       # [B, 1, 1]
        logit = logit.squeeze(dim=2)           # [B, 1]

        # ----- Option 2: class-specific importance map (no grads) -----
        # cls.weight: [1, D]  -> make it [1, D, 1, 1] for broadcasting
        w_cls = self.cls.weight
        w_reshape = w_cls.view(1, -1, 1, 1)          # [1, D, 1, 1]
        b_cls = self.cls.bias
        # norm2_w = torch.sum(w_cls*w_cls)

        # tr_weights = b_cls*w_cls/norm2_w
        # tr_weights = tr_weights.view(1, -1, 1, 1)


        # Per-location linear score: s(i) = w_cls^T A(:,i)
        # score_map = ((A+tr_weights) * w_reshape).sum(dim=1, keepdim=True)   # [B, 1, H', W']
        score_map = (A * w_reshape).sum(dim=1, keepdim=True) + b_cls  # [B, 1, H', W']

        # Contribution map: alpha(i) * score(i)
        importance_map = attn_map * score_map              # [B, 1, H', W']
        importance_map = F.relu(importance_map)
        # importance_map = importance_map/torch.abs(importance_map).sum()


        alt_map = attn_map*((A * w_reshape).sum(dim=1, keepdim=True))
        # alt_map = alt_map/torch.abs(alt_map).sum()
        alt_map = F.relu(alt_map)

        return logit, attn_map, importance_map, alt_map



##### ADAPTED MODELS ######
# class CustomResNetBinary(nn.Module):
#     def __init__(self, num_classes, in_channels=3 ,head: Literal['attn']='attn', k: int = 5,
#                 proj_dim: int = 512, attn_hidden: int = 256):
#         super().__init__()
#         net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         self.base_model = nn.Sequential(*list(net.children())[:-2])
#         # self.base_model = adapt_first_conv_to_n_channels(self.base_model, in_channels)
#         if head == 'attn':
#             self.head = AttentionMILHead(512, proj_dim=proj_dim, attn_hidden=attn_hidden)
#         else:
#             raise ValueError("head must be 'lse' or 'attn'")
#         self.head_name = head
#     def forward(self, x, xtype):
#         """
#         x: [B, 3, H, W] 
#         """
#         feat = self.base_model(x)
#         logit, map_att, map_imp, map_no_bias = self.head(feat)
#         return logit, map_att, map_imp, map_no_bias
    


class CustomResNetBinary(nn.Module):
    def __init__(self, num_classes, in_channels=3 ,head: Literal['attn']='attn',
                proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model = nn.Sequential(*list(net.children())[:-2])
        # self.base_model = adapt_first_conv_to_n_channels(self.base_model, in_channels)
        if head == 'attn':
            self.head = AttentionMILHead(512, proj_dim=proj_dim, attn_hidden=attn_hidden)
        else:
            raise ValueError("head must be 'lse' or 'attn'")
        self.head_name = head
    def forward(self, x, xtype):
        """
        x: [B, 3, H, W] 
        """
        feat = self.base_model(x)
        logit, map_att, map_contrib, _ = self.head(feat)
        return logit, map_att, map_contrib


class CustomResNetBinary50(nn.Module):
    def __init__(self, num_classes, in_channels=3, head: Literal['lse','attn']='attn', k: int = 5,
                proj_dim: int = 512, attn_hidden: int = 256, Bias=None):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.base_model = nn.Sequential(*list(net.children())[:-2])  # drop avgpool & fc

        if head == 'attn':
            self.head = AttentionMILHead(2048, proj_dim=proj_dim, attn_hidden=attn_hidden, Bias=Bias)
        else:
            raise ValueError("head must be 'attn'")
        self.head_name = head

    def forward(self, x, xtype):
        """
        x: [B, 3, H, W] 
        """
        feat = self.base_model(x)                # [B,2048,H/32,W/32]
        logit, map_att, map_imp, map_no_bias = self.head(feat)          # [B,1], [B,1,H',W']
        # prob = torch.sigmoid(logit)
        # key = 'logit_map' if self.head_name=='lse' else 'attn_map'
        return logit, map_att, map_imp, map_no_bias #Cambiar map_imp por map_contrib y quitar el map_no_bias



class CustomDenseNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, head: Literal['lse','attn']='attn', k: int = 5,
                proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        self.base_model = net.features
        feature_dim = net.classifier.in_features

        if head == 'attn':
            self.head = AttentionMILHead(feature_dim, proj_dim=proj_dim, attn_hidden=attn_hidden)
        else:
            raise ValueError("head must be 'attn'")
        self.head_name = head

    def forward(self, x, xtype):
        features = self.base_model(x)

        feat_map = F.relu(features)
        logit, map_att, map_imp, map_no_bias = self.head(feat_map)
        return logit, map_att, map_imp, map_no_bias


class CustomMobileNetV3(nn.Module):
    def __init__(self, num_classes, in_channels=3, head: Literal['lse','attn']='attn', k: int = 5,
                proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.base_model = net.features
        feature_dim = net.classifier[0].in_features
        
        if head == 'attn':
            self.head = AttentionMILHead(feature_dim, proj_dim=proj_dim, attn_hidden=attn_hidden)
        else:
            raise ValueError("head must be 'attn'")
        self.head_name = head


    def forward(self, x, xtype):
        feat_map = self.base_model(x)

        logit, map_att, map_imp, map_no_bias = self.head(feat_map)
        return logit, map_att, map_imp, map_no_bias


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes, in_channels=3, head: Literal['lse','attn']='attn', k: int = 5,
                proj_dim: int = 512, attn_hidden: int = 256):
        super().__init__()
        
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model = net.features

        features_dim = net.classifier[1].in_features

        if head == 'attn':
            self.head = AttentionMILHead(features_dim, proj_dim=proj_dim, attn_hidden=attn_hidden)
        else:
            raise ValueError("head must be 'attn'")
        self.head_name = head

    def forward(self, x, xtype):
        feat_map = self.base_model(x)

        # logit, map_ = self.head(feat_map, xtype)
        logit, map_att, map_imp, map_no_bias = self.head(feat_map)
        return logit, map_att, map_imp, map_no_bias
