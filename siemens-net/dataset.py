import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class SiameseBVRTDataset(Dataset):
    """
    Dataset for Siamese BVRT Network.
    Each sample consists of a child's drawing image, a pattern image,
    and a multi-label vector of errors.
    """

    def __init__(self, root_dir, patient_ids=None, transform=None, geometry_feature_fn=None):
        """
        Initializes the dataset.

        @param root_dir Path to the directory containing processed data.
        @param patient_ids Optional list of patient names to include (for LOSO).
        @param transform Transformations to apply to the images.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.geometry_feature_fn = geometry_feature_fn
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
                        sample = {
                            "child_path": child_path,
                            "pattern_path": pattern_path,
                            "labels": drawings_labels[drawing_idx],
                            "patient": patient_dir.name,
                            # The same BVRT pattern numbers are shown to every patient.
                            # Keeping this metadata makes it possible to build a strict
                            # pattern-only baseline that checks whether a neural network
                            # is really using the drawing, not only memorizing that a
                            # given BVRT card is usually associated with specific errors.
                            "drawing_id": drawing_idx,
                            "test_id": test_dir.name,
                        }
                        if self.geometry_feature_fn is not None:
                            # Geometry features are deterministic descriptors
                            # of the original child-pattern pair. They are not
                            # recomputed after online augmentation, because
                            # their role is to provide stable, interpretable
                            # shape information to the neural model.
                            sample["geometry_features"] = self.geometry_feature_fn(sample)
                        self.samples.append(sample)

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

        if self.transform and hasattr(self.transform, "apply_pair"):
            img_child, img_pattern = self.transform.apply_pair(img_child, img_pattern)
        elif self.transform:
            img_child = self.transform(img_child)
            img_pattern = self.transform(img_pattern)

        # Multi-label target: 1 if at least one error of a given category was
        # annotated, otherwise 0. Count/ordinal prediction can be added later
        # without changing the image loading logic.
        target = torch.tensor(self.label_vector_for_sample(sample), dtype=torch.float)

        if self.geometry_feature_fn is not None:
            geometry_features = torch.tensor(sample["geometry_features"], dtype=torch.float)
            return img_child, img_pattern, geometry_features, target

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

    def label_vector_for_sample(self, sample):
        """
        Converts the raw count annotations for one drawing into the binary
        multi-label representation used by the current experiments.
        """
        return [
            1.0 if sample["labels"].get(cat, 0) > 0 else 0.0
            for cat in self.error_categories
        ]
