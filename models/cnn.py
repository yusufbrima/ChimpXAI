import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchaudio
from config import LATENT_DIM



class ResidualBlock(nn.Module):
    """Lightweight residual block with optional dropout."""
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Shortcut if dimensions mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out



class ResidualBlockSE(nn.Module):
    """Residual block with optional SE and dropout."""
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.3, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.use_se = use_se

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # SE module
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 8, out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = out * self.se(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class SmallResCNNv5(nn.Module):
    def __init__(self, num_classes, input_channels=1, base_channels=96, dropout_p=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = ResidualBlockSE(input_channels, base_channels, stride=2, dropout_p=dropout_p)  # 128x173 -> 64x87
        self.layer2 = ResidualBlockSE(base_channels, base_channels, stride=2, dropout_p=dropout_p)   # 64x87 -> 32x44
        self.layer3 = ResidualBlockSE(base_channels, base_channels*2, stride=2, dropout_p=dropout_p) # 32x44 -> 16x22
        self.layer4 = ResidualBlockSE(base_channels*2, base_channels*2, stride=2, dropout_p=dropout_p) # 16x22 -> 8x11
        self.layer5 = ResidualBlockSE(base_channels*2, base_channels*4, stride=2, dropout_p=dropout_p) # 8x11 -> 4x6

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*4, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

class SmallCNNModel(nn.Module):
    def __init__(self, num_classes, modelstr='smallcnn', input_height=257, input_width=345):
        """
        A lightweight CNN designed for small datasets
        
        Parameters:
            num_classes (int): Number of output classes
            modelsrt (str): The type of model to use ('smallcnn')
        """
        super(SmallCNNModel, self).__init__()
        
        self.num_classes = num_classes
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        # This will need to be adjusted based on your input spectrogram size
        self.fc1 = nn.Linear(128 * (input_height//8) * (input_width//8), 256)

        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass of the network
        
        Parameters:
            x (torch.Tensor): Input tensor (batch_size, 1, height, width)
        
        Returns:
            torch.Tensor: Output logits
        """
        # print("Shape of input tensor:", x.shape)
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class CustomCNNModel(nn.Module):
    """
    A customized CNN model that accepts a single input channel and outputs a specified number of classes.

    Attributes:
        base_model (nn.Module): The base CNN model with modifications.
    """
    
    def __init__(self, num_classes, weights=None, modelstr='resnet18'):
        """
        Initializes the custom model with a single input channel and a custom number of output classes.

        Parameters:
            num_classes (int): The number of output classes for the final classification layer.
            weights (str, optional): The type of pre-trained weights to use (e.g., 'IMAGENET1K_V1').
        """
        super(CustomCNNModel, self).__init__()

        self.num_classes = num_classes
        
        # Load the ResNet-18 model, optionally with pre-trained weights
        if modelstr == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                          kernel_size=self.base_model.conv1.kernel_size,
                                          stride=self.base_model.conv1.stride,
                                          padding=self.base_model.conv1.padding,
                                          bias=False)
        
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif modelstr == 'resnet34':
            self.base_model = models.resnet34(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                          kernel_size=self.base_model.conv1.kernel_size,
                                          stride=self.base_model.conv1.stride,
                                          padding=self.base_model.conv1.padding,
                                          bias=False)
        
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif modelstr == 'resnet50':
            self.base_model = models.resnet50(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                          kernel_size=self.base_model.conv1.kernel_size,
                                          stride=self.base_model.conv1.stride,
                                          padding=self.base_model.conv1.padding,
                                          bias=False)
        
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif modelstr == 'dense121':
            self.base_model = models.densenet121(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.features.conv0 = nn.Conv2d(1, self.base_model.features.conv0.out_channels,
                                          kernel_size=self.base_model.features.conv0.kernel_size,
                                          stride=self.base_model.features.conv0.stride,
                                          padding=self.base_model.features.conv0.padding,
                                          bias=False)
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)
        else:
            self.base_model = models.efficientnet_b0(weights=weights)
            self.base_model.features[0][0]  = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
            self.base_model.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)    

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        return self.base_model(x)

class ContrastiveCNN(nn.Module):
    """
    A customized CNN model for contrastive learning that accepts a single input channel
    and outputs latent representations of a specified dimension.
    """
    def __init__(self, latent_dim, weights=None, modelstr='resnet18'):
        """
        Initializes the contrastive learning model with a single input channel and a custom latent dimension.
        
        Parameters:
        latent_dim (int): The dimension of the latent representation.
        weights (str, optional): The type of pre-trained weights to use (e.g., 'IMAGENET1K_V1').
        modelstr (str): The type of model to use ('resnet18', 'dense121', or 'efficientnet_b0').
        """
        super(ContrastiveCNN, self).__init__()
        
        if modelstr == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                              kernel_size=self.base_model.conv1.kernel_size,
                                              stride=self.base_model.conv1.stride,
                                              padding=self.base_model.conv1.padding,
                                              bias=False)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
        elif modelstr == 'dense121':
            self.base_model = models.densenet121(weights=weights)
            self.base_model.features.conv0 = nn.Conv2d(1, self.base_model.features.conv0.out_channels,
                                                       kernel_size=self.base_model.features.conv0.kernel_size,
                                                       stride=self.base_model.features.conv0.stride,
                                                       padding=self.base_model.features.conv0.padding,
                                                       bias=False)
            self.feature_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
            
        else:  # efficientnet_b0
            self.base_model = models.efficientnet_b0(weights=weights)
            self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.feature_dim = 1280
            self.base_model.classifier = nn.Identity()
        
        # Add a new projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, latent_dim)
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        x (torch.Tensor): The input tensor.
        
        Returns:
        torch.Tensor: The latent representation.
        """
        features = self.base_model(x)
        latent = self.projection_head(features)
        return latent

    def get_features(self, x):
        """
        A method to get the features before the projection head.
        
        Parameters:
        x (torch.Tensor): The input tensor.
        
        Returns:
        torch.Tensor: The features before the projection head.
        """
        return self.base_model(x)


class FinetuningClassifier(nn.Module):
    """
    A classifier built on top of a pre-trained contrastive model.
    
    This module wraps a frozen (or partially frozen) contrastive model 
    as a feature extractor and adds a linear classification head for 
    downstream fine-tuning.
    """
    def __init__(self, contrastive_model, num_classes, requires_grad=False):
        """
        Initializes the fine-tuning classifier.

        Parameters
        ----------
        contrastive_model : ContrastiveCNN or ContrastiveViT
            The pre-trained contrastive model used as a feature extractor.
        num_classes : int
            Number of output classes for the classification task.
        requires_grad : bool, optional
            Whether to allow gradient updates on the feature extractor parameters.
            Default is False (frozen backbone).
        """
        super(FinetuningClassifier, self).__init__()
        
        # Store the pre-trained contrastive feature extractor
        self.feature_extractor = contrastive_model
        self.requires_grad = requires_grad
        self.num_classes = num_classes
        
        # Optionally freeze or unfreeze the feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = self.requires_grad

        # Dynamically infer the latent dimension from the contrastive model's projection head
        # This ensures compatibility regardless of what latent_dim was used in pretraining
        latent_dim = contrastive_model.projection_head[-1].out_features

        # Define the classification layer mapping from latent space to class logits
        self.classifier = nn.Linear(latent_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the feature extractor and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output logits for each class of shape (batch_size, num_classes).
        """
        # Extract features without modifying the contrastive model’s weights
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        # Optional sanity check for dimensional consistency
        if features.shape[1] != self.classifier.in_features:
            raise ValueError(
                f"Feature dimension mismatch: got {features.shape[1]}, "
                f"expected {self.classifier.in_features}"
            )
        
        # Compute class logits
        return self.classifier(features)

    def unfreeze_last_n_layers(self, n):
        """
        Unfreezes the last n layers of the feature extractor for fine-tuning.
        
        This enables partial fine-tuning, allowing the model to adapt deeper
        layers while keeping early feature extraction stable.
        
        Parameters
        ----------
        n : int
            Number of layers (from the end) to unfreeze.
        """
        # Retrieve all modules and select the last n for fine-tuning
        trainable_layers = list(self.feature_extractor.modules())[-n:]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True



# class FinetuningClassifier(nn.Module):
#     def __init__(self, contrastive_model, num_classes, requires_grad=False):
#         super(FinetuningClassifier, self).__init__()
        
#         self.feature_extractor = contrastive_model
#         self.requires_grad = requires_grad
#         self.num_classes = num_classes
        
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = self.requires_grad

#         # Use the actual latent dimension of the contrastive model
#         latent_dim = contrastive_model.projection_head[-1].out_features
#         self.classifier = nn.Linear(latent_dim, num_classes)
    
#     def forward(self, x):
#         with torch.no_grad():
#             features = self.feature_extractor(x)
#         return self.classifier(features)


# class FinetuningClassifier(nn.Module):
#     """
#     A classifier that uses a pre-trained ContrastiveCNN model as a feature extractor
#     and adds a single trainable linear layer for classification.
#     """
#     def __init__(self, contrastive_model, num_classes,requires_grad=False):
#         """
#         Initializes the finetuning classifier.
        
#         Parameters:
#         contrastive_model (ContrastiveCNN): A pre-trained ContrastiveCNN model.
#         num_classes (int): The number of classes for the classification task.
#         """
#         super(FinetuningClassifier, self).__init__()
        
#         self.feature_extractor = contrastive_model
#         self.requires_grad =  requires_grad
#         self.num_classes =  num_classes
        
#         # Freeze all parameters of the feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = self.requires_grad
        
#         # Add a single trainable linear layer for classification
#         self.classifier = nn.Linear(LATENT_DIM, num_classes)
    
#     def forward(self, x):
#         """
#         Defines the forward pass of the model.
        
#         Parameters:
#         x (torch.Tensor): The input tensor.
        
#         Returns:
#         torch.Tensor: The classification logits.
#         """
#         with torch.no_grad():
#             features = self.feature_extractor(x)
#         return self.classifier(features)

#     def unfreeze_last_n_layers(self, n):
#         """
#         Unfreezes the last n layers of the feature extractor for fine-tuning.
        
#         Parameters:
#         n (int): The number of layers to unfreeze, counting from the end.
#         """
#         trainable_layers = list(self.feature_extractor.modules())[-n:]
#         for layer in trainable_layers:
#             for param in layer.parameters():
#                 param.requires_grad = True

# Example usage
if __name__ == "__main__":
    pass
