import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) model for spectrogram classification or fine-tuning.
    - Automatically resizes spectrograms to expected ViT input size.
    - Supports 1-channel input.
    - Can freeze or unfreeze the transformer backbone.
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 num_classes: int = 10,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 in_chans: int = 1):
        super().__init__()

        # Create base timm ViT model
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans
        )

        # Extract input size from config
        default_cfg = getattr(self.model, "default_cfg", {})
        input_size = default_cfg.get("input_size", (3, 224, 224))
        self.expected_h, self.expected_w = input_size[1], input_size[2]

        # Optionally freeze backbone (except classifier head)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "head" not in name and "fc" not in name:
                    param.requires_grad = False

    def maybe_resize(self, x):
        """Resize input spectrograms to ViT expected input size."""
        if x.shape[2] != self.expected_h or x.shape[3] != self.expected_w:
            x = F.interpolate(x, size=(self.expected_h, self.expected_w), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.maybe_resize(x)
        return self.model(x)
    
class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone that outputs features before the classifier head.
    Supports 1-channel input and automatic resizing.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, in_chans=1):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
        default_cfg = getattr(self.model, "default_cfg", {})
        input_size = default_cfg.get("input_size", (3, 224, 224))
        self.expected_h, self.expected_w = input_size[1], input_size[2]

    def maybe_resize(self, x):
        if x.shape[2] != self.expected_h or x.shape[3] != self.expected_w:
            x = F.interpolate(x, size=(self.expected_h, self.expected_w), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.maybe_resize(x)
        features = self.model.forward_features(x)
        return features



class ContrastiveViT(nn.Module):
    """
    ViT model for contrastive learning with projection head.
    """
    def __init__(self, latent_dim, model_name='vit_base_patch16_224', pretrained=True, in_chans=1):
        super().__init__()
        self.backbone = ViTBackbone(model_name=model_name, pretrained=pretrained, in_chans=in_chans)
        feature_dim = self.backbone.model.num_features  # usually 768 for base
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, latent_dim)
        )

    def forward(self, x):
        features = self.backbone(x)           # [B, 197, 768]
        cls_token = features[:, 0]            # [B, 768] — extract CLS token
        latent = self.projection_head(cls_token)  # [B, latent_dim]
        return latent

    def get_features(self, x):
        features = self.backbone(x)
        cls_token = features[:, 0]            # same extraction here if needed
        return cls_token


# class ContrastiveViT(nn.Module):
#     """
#     ViT model for contrastive learning with projection head.
#     """
#     def __init__(self, latent_dim, model_name='vit_base_patch16_224', pretrained=True, in_chans=1):
#         super().__init__()
#         self.backbone = ViTBackbone(model_name=model_name, pretrained=pretrained, in_chans=in_chans)
#         feature_dim = self.backbone.model.num_features  # usually 768 for base
#         self.projection_head = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim),
#             nn.ReLU(),
#             nn.Linear(feature_dim, latent_dim)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         latent = self.projection_head(features)
#         return latent

#     def get_features(self, x):
#         return self.backbone(x)



class ViTFinetuningClassifier(nn.Module):
    """
    Fine-tuning classifier using a pre-trained ViT backbone or ContrastiveViT model.
    """
    def __init__(self, backbone_model, num_classes, requires_grad=False):
        super().__init__()
        self.feature_extractor = backbone_model
        self.requires_grad = requires_grad
        self.num_classes = num_classes
        # Freeze backbone parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = self.requires_grad

        # Determine input feature size
        if hasattr(backbone_model, "projection_head"):
            # ContrastiveViT: projection_head outputs latent_dim
            in_features = backbone_model.projection_head[-1].out_features
        else:
            # Raw ViT backbone: use its feature dimension
            in_features = backbone_model.backbone.model.num_features

        # Simple linear classifier
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if self.requires_grad:
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)

        # Handle both cases
        if features.ndim == 3:  # [B, 197, D] → take CLS token
            features = features[:, 0]

        logits = self.classifier(features)
        return logits

    def unfreeze_last_n_layers(self, n):
        """
        Optionally unfreeze the last n layers of the backbone.
        """
        trainable_layers = list(self.feature_extractor.modules())[-n:]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

# class ViTFinetuningClassifier(nn.Module):
#     """
#     Fine-tuning classifier using a pre-trained ViT backbone or ContrastiveViT model.
#     """
#     def __init__(self, backbone_model, num_classes, requires_grad=False):
#         super().__init__()
#         self.feature_extractor = backbone_model
#         self.requires_grad = requires_grad

#         # Freeze backbone parameters
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = self.requires_grad

#         # Classifier on top of latent_dim or feature_dim
#         if hasattr(backbone_model, "projection_head"):
#             in_features = backbone_model.projection_head[-1].out_features
#         else:
#             in_features = backbone_model.backbone.model.num_features

#         self.classifier = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         with torch.no_grad():
#             features = self.feature_extractor(x)
#         return self.classifier(features)

#     def unfreeze_last_n_layers(self, n):
#         # Optionally unfreeze last n layers of backbone
#         trainable_layers = list(self.feature_extractor.modules())[-n:]
#         for layer in trainable_layers:
#             for param in layer.parameters():
#                 param.requires_grad = True
