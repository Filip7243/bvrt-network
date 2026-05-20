import json
import torch
import re
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Tuple

class HybridBVRTDataset(Dataset):
    """
    Dataset dla hybrydowej architektury ViT-ResNet w teście BVRT.
    Obsługuje 3-kanałowe obrazy (Dziecko, Wzorzec, Różnica) przygotowane przez BVRTPreprocessor.
    """

    def __init__(self, root_dir: str, patient_ids: Optional[List[str]] = None, transform: Any = None):
        """
        Inicjalizuje dataset.

        Args:
            root_dir: Ścieżka do katalogu z przetworzonymi danymi (processed/vit-resnet-data).
            patient_ids: Opcjonalna lista nazw pacjentów do uwzględnienia (dla walidacji LOSO).
            transform: Transformacje torchvision do zastosowania na obrazach.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Kategorie błędów zgodnie z neuropsychologiczną oceną BVRT
        self.error_categories = [
            "omissions", "distortions", "perseverations",
            "rotations", "displacements", "relative_size_errors"
        ]

        # Skanowanie katalogów pacjentów
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Katalog danych nie istnieje: {self.root_dir}")

        patients = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if patient_ids:
            patients = [d for d in patients if d.name in patient_ids]

        for patient_dir in patients:
            labels_file = patient_dir / "labels.json"
            if not labels_file.exists():
                print(f"Ostrzeżenie: Brak pliku labels.json w {patient_dir}")
                continue

            with open(labels_file, "r", encoding='utf-8') as f:
                labels_data = json.load(f)

            # Mapowanie drawing_id na błędy (liczba wystąpień danego błędu)
            drawings_labels = {d["drawing_id"]: d["errors"] for d in labels_data["drawings"]}

            # Wyszukiwanie obrazów PNG dla danego pacjenta
            # Format pliku: {pacjent}_{test_id}_p{indeks}.png
            for img_path in patient_dir.glob("*.png"):
                match = re.search(r'_p(\d+)\.png$', img_path.name)
                if not match:
                    continue
                
                drawing_idx = int(match.group(1))

                if drawing_idx in drawings_labels:
                    self.samples.append({
                        "img_path": img_path,
                        "labels": drawings_labels[drawing_idx],
                        "patient": patient_dir.name,
                        "drawing_idx": drawing_idx
                    })

    def __len__(self) -> int:
        """Zwraca liczbę próbek w zbiorze danych."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pobiera próbkę ze zbioru danych.

        Args:
            idx: Indeks próbki.

        Returns:
            Krotka (obraz, etykiety), gdzie:
            - obraz: Tensor przetworzonego obrazu 3-kanałowego.
            - etykiety: Multi-label tensor (1.0 jeśli błąd wystąpił >= 1 raz, 0.0 w przeciwnym razie).
        """
        sample = self.samples[idx]
        
        # Otwieranie obrazu (RGB - 3 kanały: R=Child, G=Pattern, B=Diff)
        image = Image.open(sample["img_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Transformacja etykiet: 1.0 jeśli liczba błędów w danej kategorii >= 1, w przeciwnym razie 0.0
        target = torch.tensor([
            1.0 if sample["labels"].get(cat, 0) > 0 else 0.0
            for cat in self.error_categories
        ], dtype=torch.float)

        return image, target

    def get_pos_weights(self) -> torch.Tensor:
        """
        Oblicza wagi dla klas pozytywnych (pos_weight) do zastosowania w BCEWithLogitsLoss.
        Pomaga zrównoważyć niezbalansowany zbiór danych.
        """
        all_labels = []
        for sample in self.samples:
            target = [
                1.0 if sample["labels"].get(cat, 0) > 0 else 0.0
                for cat in self.error_categories
            ]
            all_labels.append(target)
        
        labels_tensor = torch.tensor(all_labels, dtype=torch.float)
        positives = labels_tensor.sum(dim=0)
        negatives = len(self.samples) - positives
        
        # Dodajemy małą stałą epsilon, aby uniknąć dzielenia przez zero
        pos_weight = negatives / (positives + 1e-6)
        return pos_weight
