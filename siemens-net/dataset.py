import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SiameseBVRTDataset(Dataset):
    """
    Dataset for Siamese BVRT Network.
    Each sample consists of a child's drawing image, a pattern image,
    and a multi-label vector of errors.
    """

    def __init__(self, root_dir, patient_ids=None, transform=None):
        """
        Initializes the dataset.

        @param root_dir Path to the directory containing processed data.
        @param patient_ids Optional list of patient names to include (for LOSO).
        @param transform Transformations to apply to the images.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        self.error_categories = [
            "omissions", "distortions", "perseverations",
            "rotations", "displacements", "relative_size_errors"
        ]

        # Scan patient directories
        patients = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if patient_ids:
            patients = [d for d in patients if d.name in patient_ids]

        for patient_dir in patients:
            # Patients may have multiple test subdirectories
            for test_dir in patient_dir.iterdir():
                if not test_dir.is_dir():
                    continue
                
                labels_file = test_dir / "labels.json"
                if not labels_file.exists():
                    continue

                with open(labels_file, "r", encoding='utf-8') as f:
                    labels_data = json.load(f)

                # Map drawing_id to errors
                drawings_labels = {d["drawing_id"]: d["errors"] for d in labels_data["drawings"]}

                # Iterate through p1, p2, ..., p10 directories
                for p_dir in test_dir.glob("p*"):
                    try:
                        drawing_idx = int(p_dir.name[1:])
                    except ValueError:
                        continue

                    child_path = p_dir / "child.png"
                    pattern_path = p_dir / "pattern.png"

                    if child_path.exists() and pattern_path.exists() and drawing_idx in drawings_labels:
                        self.samples.append({
                            "child_path": child_path,
                            "pattern_path": pattern_path,
                            "labels": drawings_labels[drawing_idx],
                            "patient": patient_dir.name
                        })

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        @param idx Index of the sample to retrieve.
        @return A tuple (img_child, img_pattern, target).
        """
        sample = self.samples[idx]
        
        img_child = Image.open(sample["child_path"]).convert("RGB")
        img_pattern = Image.open(sample["pattern_path"]).convert("RGB")

        if self.transform:
            img_child = self.transform(img_child)
            img_pattern = self.transform(img_pattern)

        # Multi-label target: 1 if count > 0, else 0
        target = torch.tensor([
            1.0 if sample["labels"].get(cat, 0) > 0 else 0.0
            for cat in self.error_categories
        ], dtype=torch.float)

        return img_child, img_pattern, target

    def get_labels(self):
        """
        Returns all labels in the dataset as a single tensor.
        Useful for calculating pos_weight.
        """
        all_labels = []
        for sample in self.samples:
            target = [
                1.0 if sample["labels"].get(cat, 0) > 0 else 0.0
                for cat in self.error_categories
            ]
            all_labels.append(target)
        return torch.tensor(all_labels, dtype=torch.float)
