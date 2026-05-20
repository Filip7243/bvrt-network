import torch
import sys
from pathlib import Path

# Dodanie ścieżki src do sys.path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parents[0]))

from src.models.hybrid_model import HybridBVRTModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_phases():
    print("Testowanie modelu HybridBVRTModel (ResNet18 + ViT-B-16)...")
    model = HybridBVRTModel(cnn_type="resnet18", vit_type="vit_b_16", pretrained=False)
    
    # Przykładowy input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape} (Oczekiwano: [2, 6])")
    
    print("\n--- FAZA 1 ---")
    model.set_train_phase(1)
    trainable_p1 = count_parameters(model)
    print(f"Liczba trenowalnych parametrów w Fazie 1: {trainable_p1}")
    
    print("\n--- FAZA 2 ---")
    model.set_train_phase(2)
    trainable_p2 = count_parameters(model)
    print(f"Liczba trenowalnych parametrów w Fazie 2: {trainable_p2}")
    
    if trainable_p2 > trainable_p1:
        print("\nSukces: Liczba parametrów w Fazie 2 jest większa niż w Fazie 1.")
    else:
        print("\nBłąd: Liczba parametrów w Fazie 2 nie wzrosła!")

    print("\nTestowanie wersji z EfficientNet-B0...")
    model_eff = HybridBVRTModel(cnn_type="efficientnet_b0", vit_type="vit_b_16", pretrained=False)
    model_eff.set_train_phase(2)
    print(f"Liczba trenowalnych parametrów (EffNet B0 Phase 2): {count_parameters(model_eff)}")

if __name__ == "__main__":
    test_model_phases()
