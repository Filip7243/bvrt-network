import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from pathlib import Path

class BVRTDataset(Dataset):
    def __init__(self, processed_dir, raw_dir, transform=None):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.transform = transform
        self.samples = []
        self.error_types = ["omissions", "distortions", "perseverations", "rotations", "displacements", "relative_size_errors"]
        
        self._prepare_samples()

    def _prepare_samples(self):
        # Iterujemy po pacjentach w przetworzonych danych
        for patient_dir in self.processed_dir.iterdir():
            if not patient_dir.is_dir(): continue
            patient_name = patient_dir.name
            
            # Szukamy odpowiadającego folderu w raw, aby pobrać labels.json
            # Pacjent w raw może mieć wiele testów, ale tutaj zakładamy strukturę data/raw/Pacjent/Test/labels.json
            raw_patient_path = self.raw_dir / patient_name
            if not raw_patient_path.exists(): continue
            
            for test_dir in raw_patient_path.iterdir():
                if not test_dir.is_dir(): continue
                labels_path = test_dir / "labels.json"
                if not labels_path.exists(): continue
                
                with open(labels_path, 'r') as f:
                    labels_data = json.load(f)
                
                # Tworzymy mapowanie drawing_id -> errors
                drawings_labels = {d['drawing_id']: d['errors'] for d in labels_data.get('drawings', [])}
                
                # Dopasowujemy przetworzone obrazy
                # Nazwa pliku: {patient_name}_{test_id}_p{pattern_idx}.png
                for img_path in patient_dir.glob(f"{patient_name}_{test_dir.name}_p*.png"):
                    import re
                    match = re.search(r'_p(\d+)\.png$', img_path.name)
                    if not match: continue
                    drawing_id = int(match.group(1))
                    
                    if drawing_id in drawings_labels:
                        self.samples.append({
                            'img_path': img_path,
                            'labels': drawings_labels[drawing_id],
                            'patient': patient_name
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Transformacja etykiet: z liczby błędów na format binarny (multi-label)
        # 1 if count > 0 else 0
        target = torch.tensor([
            1.0 if sample['labels'].get(et, 0) > 0 else 0.0 
            for et in self.error_types
        ], dtype=torch.float32)
        
        return image, target, sample['patient']

def get_model(num_classes):
    # Załadowanie ResNet z zamrożonymi warstwami
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # Zastąpienie ostatniej warstwy
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Szybki test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = BVRTDataset(
        processed_dir="data/processed/3d-input-data",
        raw_dir="data/raw",
        transform=transform
    )
    
    print(f"Liczba próbek: {len(dataset)}")
    if len(dataset) > 0:
        img, target, patient = dataset[0]
        print(f"Shape obrazu: {img.shape}")
        print(f"Etykiety (binary): {target}")
        print(f"Pacjent: {patient}")
        
        model = get_model(len(dataset.error_types))
        output = model(img.unsqueeze(0))
        print(f"Output modelu shape: {output.shape}")
