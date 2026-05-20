import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights
from typing import Optional, Tuple

class HybridBVRTModel(nn.Module):
    """
    Hybrydowa architektura CNN-ViT dla diagnozy błędów w teście BVRT.
    Łączy cechy lokalne z CNN (ResNet/EfficientNet) oraz globalne relacje przestrzenne z ViT.
    """

    def __init__(
        self, 
        cnn_type: str = "efficientnet_b0", 
        vit_type: str = "vit_b_16", 
        num_classes: int = 6,
        pretrained: bool = True
    ):
        super(HybridBVRTModel, self).__init__()
        
        # 1. Ekstraktor Lokalny (CNN)
        if cnn_type == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.cnn = models.resnet18(weights=weights)
            self.cnn_features_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity() # Usuwamy warstwę klasyfikacyjną
        elif cnn_type == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.cnn = models.efficientnet_b0(weights=weights)
            self.cnn_features_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
        else:
            raise ValueError(f"Nieobsługiwany typ CNN: {cnn_type}")

        # 2. Ekstraktor Globalny (ViT)
        if vit_type == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            self.vit_features_dim = self.vit.heads[0].in_features
            self.vit.heads = nn.Identity() # Usuwamy głowę
        else:
            # Można tu dodać mniejsze wersje ViT (np. z biblioteki timm), 
            # ale trzymamy się standardowego torchvision dla czystości.
            raise ValueError(f"Nieobsługiwany typ ViT: {vit_type}")

        # 3. Fusion Layer (Łączenie wyników)
        combined_dim = self.cnn_features_dim + self.vit_features_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass modelu.
        Args:
            x: Tensor wejściowy (B, 3, 224, 224)
        Returns:
            Logity dla 6 kategorii błędów.
        """
        # Ekstrakcja cech z obu strumieni
        cnn_feats = self.cnn(x) # (B, cnn_features_dim)
        vit_feats = self.vit(x) # (B, vit_features_dim)
        
        # Konkatenacja wektorów cech
        combined = torch.cat((cnn_feats, vit_feats), dim=1)
        
        # Klasyfikacja finalna
        logits = self.fusion_head(combined)
        return logits

    def set_train_phase(self, phase: int):
        """
        Ustawia fazę trenowania zgodnie ze strategią Two-Phase Training.
        Phase 1: Frozen backbones, train only fusion head.
        Phase 2: Unfreeze last blocks of CNN and ViT for fine-tuning.
        """
        if phase == 1:
            print("Ustawianie Fazy 1: Zamrażanie backbone'ów, trenowanie tylko głowy fusion.")
            # Zamrażamy wszystko
            for param in self.parameters():
                param.requires_grad = False
            # Odmrażamy tylko głowę
            for param in self.fusion_head.parameters():
                param.requires_grad = True
                
        elif phase == 2:
            print("Ustawianie Fazy 2: Odmrażanie wybranych warstw do Fine-tuningu.")
            # 1. Odmrażamy głowę (zawsze trenowalna)
            for param in self.fusion_head.parameters():
                param.requires_grad = True
            
            # 2. Odmrażamy końcówkę CNN
            if hasattr(self.cnn, 'layer4'): # ResNet
                for param in self.cnn.layer4.parameters():
                    param.requires_grad = True
            elif hasattr(self.cnn, 'features'): # EfficientNet
                # Odmrażamy ostatnie 2 bloki
                for param in self.cnn.features[7:].parameters():
                    param.requires_grad = True
                    
            # 3. Odmrażamy końcówkę ViT (ostatni blok EncoderBlock)
            # W torchvision.models.vit_b_16: vit.encoder.layers to Sequential
            # Odmrażamy ostatnią warstwę (indeks -1)
            for param in self.vit.encoder.layers[-1].parameters():
                param.requires_grad = True
        else:
            raise ValueError("Faza musi być 1 lub 2.")

    def get_trainable_parameters(self):
        """Zwraca parametry, które mają requires_grad=True."""
        return filter(lambda p: p.requires_grad, self.parameters())
