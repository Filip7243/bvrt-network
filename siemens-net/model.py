import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SiameseEfficientNet(nn.Module):
    """
    Siamese Network architecture using EfficientNet-B0 as a backbone.
    This model processes two images (child drawing and pattern) through a shared backbone,
    fuses their features, and performs multi-label classification.
    """

    def __init__(self, num_classes=6, dropout_rate=0.5, spatial_dropout_rate=0.0):
        """
        Initializes the Siamese network.

        @param num_classes The number of output classes for multi-label classification.
        @param dropout_rate Dropout probability for the fully connected layer.
        @param spatial_dropout_rate Dropout probability for the spatial dropout (Dropout2d) 
                                   applied to the feature maps.
        """
        super(SiameseEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Extract features (remove the classifier head)
        # EfficientNet-B0 output channels before global average pooling is 1280
        self.feature_extractor = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout_rate) if spatial_dropout_rate > 0 else nn.Identity()
        
        feature_dim = 1280
        
        # Fusion layer results in 2 * feature_dim because of concat(|f1-f2|, f1*f2)
        fusion_dim = 2 * feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        """
        Passes one image through the shared backbone.

        @param x Input tensor representing one image.
        @return Feature vector after global average pooling.
        """
        x = self.feature_extractor(x)
        x = self.spatial_dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, img_child, img_pattern):
        """
        Forward pass for the Siamese network.

        @param img_child Input tensor for the child's drawing.
        @param img_pattern Input tensor for the pattern image.
        @return Output logits for multi-label classification.
        """
        # Get features from both arms (shared weights)
        f_child = self.forward_one(img_child)
        f_pattern = self.forward_one(img_pattern)
        
        # Feature Fusion: concat(|f1 - f2|, f1 * f2)
        f_diff = torch.abs(f_child - f_pattern)
        f_mul = f_child * f_pattern
        
        combined = torch.cat([f_diff, f_mul], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits

    def freeze_backbone(self):
        """
        Freezes all parameters in the EfficientNet backbone.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_blocks(self, blocks_to_unfreeze=None):
        """
        Unfreezes specific blocks of the EfficientNet backbone.
        By default, unfreezes blocks 6 and 7 as recommended.

        @param blocks_to_unfreeze List of block indices to unfreeze.
        """
        if blocks_to_unfreeze is None:
            # EfficientNet-B0 has blocks in self.feature_extractor (0 to 8)
            # Blocks 6 and 7 are usually the last ones before the final 1x1 conv (block 8)
            blocks_to_unfreeze = [6, 7]
            
        for i in blocks_to_unfreeze:
            for param in self.feature_extractor[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze the final 1x1 conv (block 8)
        for param in self.feature_extractor[8].parameters():
            param.requires_grad = True
