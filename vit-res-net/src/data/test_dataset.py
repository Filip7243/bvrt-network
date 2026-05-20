import torch
from torchvision import transforms
from dataset import HybridBVRTDataset
from pathlib import Path

def test_dataset():
    # Ścieżka do danych (relatywna do głównego folderu projektu, ale jesteśmy w src/data)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    data_dir = project_root / "data/processed/vit-resnet-data"
    
    print(f"Sprawdzanie danych w: {data_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    try:
        ds = HybridBVRTDataset(root_dir=str(data_dir), transform=transform)
        print(f"Załadowano dataset: {len(ds)} próbek.")
        
        if len(ds) > 0:
            img, target = ds[0]
            print(f"Kształt obrazu: {img.shape}")
            print(f"Etykiety (target): {target}")
            print(f"Typ etykiet: {target.dtype}")
            
            pos_weights = ds.get_pos_weights()
            print(f"Wagi pos_weight: {pos_weights}")
        else:
            print("Błąd: Dataset jest pusty!")
            
    except Exception as e:
        print(f"Wystąpił błąd podczas testowania datasetu: {e}")

if __name__ == "__main__":
    test_dataset()
