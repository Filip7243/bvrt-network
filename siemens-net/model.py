import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18


def build_small_classifier(input_dim, num_classes, dropout_rate):
    """
    Small classification head for very small BVRT folds.

    The previous BatchNorm-based head was statistically fragile because every
    LOSO fold contains only about 120 training drawings and mini-batches of 8.
    LayerNorm does not estimate running batch statistics, so it is more stable
    for this setting.
    """
    hidden_dim = 128
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_classes),
    )


class SiameseEfficientNet(nn.Module):
    """
    Siamese Network architecture using EfficientNet-B0 as a backbone.
    This model processes two images (child drawing and pattern) through a shared backbone,
    fuses their features, and performs multi-label classification.
    """

    def __init__(self, num_classes=6, dropout_rate=0.5, spatial_dropout_rate=0.0,
                 include_raw_features=True, pretrained=True):
        """
        Initializes the Siamese network.

        @param num_classes The number of output classes for multi-label classification.
        @param dropout_rate Dropout probability for the fully connected layer.
        @param spatial_dropout_rate Dropout probability for the spatial dropout (Dropout2d) 
                                   applied to the feature maps.
        """
        super(SiameseEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.include_raw_features = include_raw_features
        
        # Extract features (remove the classifier head)
        # EfficientNet-B0 output channels before global average pooling is 1280
        self.feature_extractor = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout_rate) if spatial_dropout_rate > 0 else nn.Identity()
        
        feature_dim = 1280
        
        # Directional features help distinguish omissions from additions/perseverations.
        fusion_dim = 4 * feature_dim if include_raw_features else 2 * feature_dim
        
        self.classifier = build_small_classifier(fusion_dim, num_classes, dropout_rate)

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
        
        if self.include_raw_features:
            combined = torch.cat([f_child, f_pattern, f_diff, f_mul], dim=1)
        else:
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

    def freeze_backbone_batchnorm(self):
        """
        Keeps frozen EfficientNet normalization/dropout behavior deterministic while
        training only the fusion head on very small folds.
        """
        self.feature_extractor.eval()

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


class SiameseEfficientNetGeometryFusion(nn.Module):
    """
    Compact vector-fusion siamese model assisted by explicit geometry features.

    This model is designed for the current small BVRT dataset. It avoids the
    large late-fusion convolutional block and uses only relational CNN features
    by default: |child - pattern| and child * pattern. Interpretable geometric
    descriptors are processed by a tiny MLP and fused with the CNN embedding at
    the classifier level.
    """

    def __init__(
        self,
        num_classes=6,
        geometry_feature_dim=19,
        dropout_rate=0.35,
        spatial_dropout_rate=0.0,
        include_raw_features=False,
        pretrained=True,
    ):
        super(SiameseEfficientNetGeometryFusion, self).__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.feature_extractor = self.backbone.features
        self.avgpool = self.backbone.avgpool
        self.include_raw_features = include_raw_features
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout_rate) if spatial_dropout_rate > 0 else nn.Identity()

        feature_dim = 1280
        fusion_dim = 4 * feature_dim if include_raw_features else 2 * feature_dim

        # Compress the frozen CNN relation embedding before adding the geometry
        # branch. This keeps the trainable part small relative to n=14 patients.
        self.image_projection = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Geometry features are already meaningful numeric descriptors. The MLP
        # is intentionally tiny: it only rescales and lightly mixes them before
        # final fusion with the CNN representation.
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_feature_dim),
            nn.Linear(geometry_feature_dim, 16),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward_one(self, x):
        """Extracts one global EfficientNet embedding."""
        x = self.feature_extractor(x)
        x = self.spatial_dropout(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, img_child, img_pattern, geometry_features):
        f_child = self.forward_one(img_child)
        f_pattern = self.forward_one(img_pattern)
        f_diff = torch.abs(f_child - f_pattern)
        f_mul = f_child * f_pattern

        if self.include_raw_features:
            image_features = torch.cat([f_child, f_pattern, f_diff, f_mul], dim=1)
        else:
            image_features = torch.cat([f_diff, f_mul], dim=1)

        image_embedding = self.image_projection(image_features)
        geometry_embedding = self.geometry_projection(geometry_features)
        combined = torch.cat([image_embedding, geometry_embedding], dim=1)
        return self.classifier(combined)

    def freeze_backbone(self):
        """Freezes the shared EfficientNet feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_backbone_batchnorm(self):
        """
        Keeps the frozen EfficientNet normalization/dropout behavior stable
        while training only projection/classifier layers.
        """
        self.feature_extractor.eval()

    def unfreeze_blocks(self, blocks_to_unfreeze=None):
        """Unfreezes selected late EfficientNet blocks for cautious fine-tuning."""
        if blocks_to_unfreeze is None:
            blocks_to_unfreeze = [6, 7]

        for i in blocks_to_unfreeze:
            for param in self.feature_extractor[i].parameters():
                param.requires_grad = True

        for param in self.feature_extractor[8].parameters():
            param.requires_grad = True


class SiameseResNet18GeometryFusion(nn.Module):
    """
    Compact Siamese ResNet18 assisted by explicit geometry features.

    This mirrors the best direction from the 3D-input experiments: use a stable
    pretrained ResNet18 representation, compare child/pattern embeddings, and
    keep the trainable fusion head small. ResNet18 is intentionally smaller
    than EfficientNet-B0, which is useful when every LOSO fold has only about a
    dozen patients.
    """

    def __init__(
        self,
        num_classes=6,
        geometry_feature_dim=19,
        dropout_rate=0.35,
        spatial_dropout_rate=0.0,
        include_raw_features=False,
        pretrained=True,
    ):
        super(SiameseResNet18GeometryFusion, self).__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.include_raw_features = include_raw_features
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout_rate) if spatial_dropout_rate > 0 else nn.Identity()

        feature_dim = 512
        fusion_dim = 4 * feature_dim if include_raw_features else 2 * feature_dim

        self.image_projection = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_feature_dim),
            nn.Linear(geometry_feature_dim, 16),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = self.spatial_dropout(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, img_child, img_pattern, geometry_features):
        f_child = self.forward_one(img_child)
        f_pattern = self.forward_one(img_pattern)
        f_diff = torch.abs(f_child - f_pattern)
        f_mul = f_child * f_pattern

        if self.include_raw_features:
            image_features = torch.cat([f_child, f_pattern, f_diff, f_mul], dim=1)
        else:
            image_features = torch.cat([f_diff, f_mul], dim=1)

        image_embedding = self.image_projection(image_features)
        geometry_embedding = self.geometry_projection(geometry_features)
        combined = torch.cat([image_embedding, geometry_embedding], dim=1)
        return self.classifier(combined)

    def freeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_backbone_batchnorm(self):
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def unfreeze_blocks(self, blocks_to_unfreeze=None):
        # ResNet18 does not have EfficientNet-style numbered blocks. For cautious
        # fine-tuning, expose only the last residual stage.
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True


class SiameseEfficientNetLateFusion(nn.Module):
    """
    Siamese EfficientNet with feature-map late fusion.

    The vector-fusion model pools each branch to a single 1280-dimensional
    vector before comparing child and pattern images. That is convenient, but
    it discards spatial layout early. BVRT errors are strongly geometric
    (omissions, rotations, displacements, relative size), so this variant
    compares the two branches while the backbone still holds 2-D feature maps.
    """

    def __init__(
        self,
        num_classes=6,
        dropout_rate=0.35,
        spatial_dropout_rate=0.1,
        include_raw_features=True,
        pretrained=True,
    ):
        super(SiameseEfficientNetLateFusion, self).__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.feature_extractor = self.backbone.features
        self.include_raw_features = include_raw_features

        feature_channels = 1280
        fusion_channels = 4 * feature_channels if include_raw_features else 2 * feature_channels

        # A compact convolutional fusion block learns local relations between
        # the child's drawing and the reference pattern before global pooling.
        # GroupNorm is used instead of BatchNorm for stability with tiny folds.
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 512, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=spatial_dropout_rate),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.SiLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = build_small_classifier(256, num_classes, dropout_rate)

    def forward_one_map(self, x):
        """Returns the last EfficientNet feature map without global pooling."""
        return self.feature_extractor(x)

    def forward(self, img_child, img_pattern):
        f_child = self.forward_one_map(img_child)
        f_pattern = self.forward_one_map(img_pattern)

        f_diff = torch.abs(f_child - f_pattern)
        f_mul = f_child * f_pattern
        if self.include_raw_features:
            combined = torch.cat([f_child, f_pattern, f_diff, f_mul], dim=1)
        else:
            combined = torch.cat([f_diff, f_mul], dim=1)

        fused = self.fusion(combined)
        pooled = self.avgpool(fused)
        pooled = torch.flatten(pooled, 1)
        return self.classifier(pooled)

    def freeze_backbone(self):
        """Freezes the shared EfficientNet feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_backbone_batchnorm(self):
        """
        Keeps the frozen EfficientNet branch deterministic during head training.
        The fusion block remains in training mode, so dropout still regularizes
        the newly added BVRT-specific layers.
        """
        self.feature_extractor.eval()

    def unfreeze_blocks(self, blocks_to_unfreeze=None):
        """Unfreezes selected late EfficientNet blocks for cautious fine-tuning."""
        if blocks_to_unfreeze is None:
            blocks_to_unfreeze = [6, 7]

        for i in blocks_to_unfreeze:
            for param in self.feature_extractor[i].parameters():
                param.requires_grad = True

        for param in self.feature_extractor[8].parameters():
            param.requires_grad = True
