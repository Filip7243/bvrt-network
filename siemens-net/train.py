import copy
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from baselines import (
    GEOMETRY_FEATURE_DIM,
    constant_predictions,
    geometric_features_for_sample,
    geometric_logreg_predictions,
    majority_baseline_vector,
    pattern_majority_predictions,
)
from dataset import SiameseBVRTDataset
from model import (
    SiameseEfficientNet,
    SiameseEfficientNetGeometryFusion,
    SiameseEfficientNetLateFusion,
    SiameseResNet18GeometryFusion,
)


ERROR_CATEGORIES = [
    "omissions",
    "distortions",
    "perseverations",
    "rotations",
    "displacements",
    "relative_size_errors",
]


class ResizeLongSideAndPad:
    """Preserve BVRT geometry by resizing the long side and padding to a square."""

    def __init__(self, size: int = 224, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


class SiameseEvalTransform:
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose(
            [
                ResizeLongSideAndPad(size=image_size, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    def apply_pair(self, child: Image.Image, pattern: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(child), self.transform(pattern)


class SiameseTrainTransform:
    """
    Paired, semantics-preserving augmentation for BVRT drawings.

    The key rule is that geometric transforms are applied to child and pattern
    together. This changes camera/canvas placement but does not create or
    remove BVRT errors such as rotations or displacements between the two
    images. Label-changing augmentations, for example rotating only the child
    drawing, are intentionally avoided.
    """

    def __init__(
        self,
        image_size: int = 224,
        affine_probability: float = 0.75,
        blur_probability: float = 0.15,
        translate: Tuple[float, float] = (0.02, 0.02),
        scale: Tuple[float, float] = (0.97, 1.03),
        degrees: Tuple[float, float] = (-2.0, 2.0),
    ):
        self.resize = ResizeLongSideAndPad(size=image_size, fill=0)
        self.affine_probability = affine_probability
        self.blur_probability = blur_probability
        self.translate = translate
        self.scale = scale
        self.degrees = degrees
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _maybe_apply_joint_affine(
        self, child: Image.Image, pattern: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() >= self.affine_probability:
            return child, pattern

        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            degrees=self.degrees,
            translate=self.translate,
            scale_ranges=self.scale,
            shears=None,
            img_size=[child.height, child.width],
        )
        kwargs = {
            "angle": angle,
            "translate": translations,
            "scale": scale,
            "shear": shear,
            "interpolation": InterpolationMode.BILINEAR,
            "fill": 0,
        }
        return TF.affine(child, **kwargs), TF.affine(pattern, **kwargs)

    def _maybe_apply_joint_blur(
        self, child: Image.Image, pattern: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() >= self.blur_probability:
            return child, pattern

        # Light blur simulates rasterization/tablet variability. It is applied
        # to both images, so the clinical relation between child and pattern is
        # preserved.
        radius = random.uniform(0.2, 0.6)
        return child.filter(ImageFilter.GaussianBlur(radius)), pattern.filter(ImageFilter.GaussianBlur(radius))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.to_tensor(self.resize(image))

    def apply_pair(self, child: Image.Image, pattern: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        child = self.resize(child)
        pattern = self.resize(pattern)
        child, pattern = self._maybe_apply_joint_affine(child, pattern)
        child, pattern = self._maybe_apply_joint_blur(child, pattern)
        return self.to_tensor(child), self.to_tensor(pattern)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_pos_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
    """
    Class-balancing weights for BCEWithLogitsLoss.

    With very small folds, a label can be absent or always present in the
    training split. In that case the theoretical neg/pos ratio is either
    undefined or unhelpfully extreme, so the weight falls back to 1.0. A clamp
    also prevents a single rare label from dominating the whole optimization.
    """
    positives = labels_tensor.sum(dim=0)
    negatives = labels_tensor.shape[0] - positives
    raw_weights = negatives / (positives + 1e-6)
    safe_weights = torch.where((positives == 0) | (negatives == 0), torch.ones_like(raw_weights), raw_weights)
    return torch.clamp(safe_weights, min=0.1, max=20.0)


def unpack_batch(batch, device):
    """
    Supports both dataset formats:
    - image-only siamese batches: child, pattern, labels
    - geometry-assisted batches: child, pattern, geometry_features, labels
    """
    if len(batch) == 4:
        img_child, img_pattern, geometry_features, labels = batch
        return (
            img_child.to(device),
            img_pattern.to(device),
            geometry_features.to(device),
            labels.to(device),
        )

    img_child, img_pattern, labels = batch
    return img_child.to(device), img_pattern.to(device), None, labels.to(device)


def model_forward(model, img_child, img_pattern, geometry_features):
    """
    Calls the correct model signature. Geometry-aware models require a third
    tensor, while the original siamese models use only the image pair.
    """
    if geometry_features is None:
        return model(img_child, img_pattern)
    return model(img_child, img_pattern, geometry_features)


def train_one_epoch(model, loader, criterion, optimizer, device, freeze_backbone_bn=True):
    model.train()
    if freeze_backbone_bn and hasattr(model, "freeze_backbone_batchnorm"):
        model.freeze_backbone_batchnorm()

    running_loss = 0.0
    for batch in loader:
        img_child, img_pattern, geometry_features, labels = unpack_batch(batch, device)

        optimizer.zero_grad()
        outputs = model_forward(model, img_child, img_pattern, geometry_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_child.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            img_child, img_pattern, geometry_features, labels = unpack_batch(batch, device)

            logits = model_forward(model, img_child, img_pattern, geometry_features)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * img_child.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return running_loss / len(loader.dataset), np.vstack(all_probs), np.vstack(all_labels)


def labels_from_dataset(dataset: SiameseBVRTDataset) -> np.ndarray:
    return dataset.get_labels().numpy()


def select_validation_patient(train_candidates: Sequence[str], fold_idx: int) -> str:
    candidates = sorted(train_candidates)
    if not candidates:
        raise ValueError("Cannot select a validation patient from an empty candidate list.")
    return candidates[fold_idx % len(candidates)]


def sanity_check_splits(train_patients: Sequence[str], val_patient: str, test_patient: str) -> None:
    train_set = set(train_patients)
    val_set = {val_patient}
    test_set = {test_patient}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError(
            f"Patient leakage detected: train={train_patients}, val={val_patient}, test={test_patient}"
        )


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds.reshape(1, -1)).astype(int)


def tune_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    tuned = np.full(labels.shape[1], 0.5, dtype=np.float32)
    details: Dict[str, Any] = {}

    for idx, category in enumerate(ERROR_CATEGORIES):
        support = int(labels[:, idx].sum())
        if support == 0:
            details[category] = {
                "threshold": 0.5,
                "best_f1": 0.0,
                "fallback": "no_positive_validation_samples",
            }
            continue

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            pred = (probs[:, idx] >= threshold).astype(int)
            score = f1_score(labels[:, idx], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        tuned[idx] = float(best_threshold)
        details[category] = {
            "threshold": float(best_threshold),
            "best_f1": float(best_f1),
            "fallback": None,
        }

    return tuned, details


def compute_multilabel_metrics(
    labels: np.ndarray,
    preds_binary: np.ndarray,
    thresholds: np.ndarray,
    strategy: str,
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds_binary,
        average=None,
        zero_division=0,
    )

    per_label = {}
    for idx, category in enumerate(ERROR_CATEGORIES):
        per_label[category] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "actual_positives": int(labels[:, idx].sum()),
            "predicted_positives": int(preds_binary[:, idx].sum()),
            "true_positive_matches": int(((labels[:, idx] == 1) & (preds_binary[:, idx] == 1)).sum()),
            "threshold": float(thresholds[idx]),
        }

    return {
        "strategy": strategy,
        "macro_f1": float(f1_score(labels, preds_binary, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds_binary, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds_binary, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds_binary)),
        "subset_accuracy": float(accuracy_score(labels, preds_binary)),
        "per_label": per_label,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for fold in fold_metrics:
        for strategy, metrics in fold["test_metrics"].items():
            rows.append(
                {
                    "fold": fold["fold"],
                    "test_patient": fold["test_patient"],
                    "val_patient": fold["val_patient"],
                    "strategy": strategy,
                    "macro_f1": metrics["macro_f1"],
                    "micro_f1": metrics["micro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "hamming_loss": metrics["hamming_loss"],
                    "subset_accuracy": metrics["subset_accuracy"],
                    "best_val_loss": fold["best_val_loss"],
                    "epochs_trained": fold["epochs_trained"],
                }
            )
    return rows


def flatten_per_label_metrics(fold_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for fold in fold_metrics:
        for strategy, metrics in fold["test_metrics"].items():
            for label, label_metrics in metrics["per_label"].items():
                row = {
                    "fold": fold["fold"],
                    "test_patient": fold["test_patient"],
                    "val_patient": fold["val_patient"],
                    "strategy": strategy,
                    "label": label,
                }
                row.update(label_metrics)
                rows.append(row)
    return rows


def summarize_results(fold_metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"config": config, "strategies": {}}
    strategies = sorted(fold_metrics[0]["test_metrics"].keys()) if fold_metrics else []

    for strategy in strategies:
        strategy_metrics = [fold["test_metrics"][strategy] for fold in fold_metrics]
        summary["strategies"][strategy] = {}
        for key in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "subset_accuracy"]:
            values = np.asarray([metrics[key] for metrics in strategy_metrics], dtype=np.float32)
            summary["strategies"][strategy][f"{key}_mean"] = float(values.mean())
            summary["strategies"][strategy][f"{key}_std"] = float(values.std())

        per_label = {}
        for label in ERROR_CATEGORIES:
            f1_values = np.asarray(
                [metrics["per_label"][label]["f1"] for metrics in strategy_metrics],
                dtype=np.float32,
            )
            per_label[label] = {
                "f1_mean": float(f1_values.mean()),
                "f1_std": float(f1_values.std()),
            }
        summary["strategies"][strategy]["per_label"] = per_label

    return summary


def build_transforms(image_size: int = 224, use_augmentation: bool = True) -> Tuple[Any, Any]:
    """
    Builds separate train/eval transforms.

    Validation and test data must remain deterministic. Augmentation is used
    only for training folds and is paired across child/pattern images.
    """
    eval_transform = SiameseEvalTransform(image_size=image_size)
    train_transform = SiameseTrainTransform(image_size=image_size) if use_augmentation else eval_transform
    return train_transform, eval_transform


def build_model(
    model_arch: str,
    num_classes: int,
    spatial_dropout: float,
    include_raw_features: bool,
    pretrained: bool,
    geometry_feature_dim: int = GEOMETRY_FEATURE_DIM,
) -> nn.Module:
    """Factory for the available siamese architectures."""
    if model_arch == "vector_fusion":
        return SiameseEfficientNet(
            num_classes=num_classes,
            spatial_dropout_rate=spatial_dropout,
            include_raw_features=include_raw_features,
            pretrained=pretrained,
        )
    if model_arch == "vector_geometry_fusion":
        return SiameseEfficientNetGeometryFusion(
            num_classes=num_classes,
            geometry_feature_dim=geometry_feature_dim,
            spatial_dropout_rate=spatial_dropout,
            include_raw_features=include_raw_features,
            pretrained=pretrained,
        )
    if model_arch == "resnet18_vector_geometry_fusion":
        return SiameseResNet18GeometryFusion(
            num_classes=num_classes,
            geometry_feature_dim=geometry_feature_dim,
            spatial_dropout_rate=spatial_dropout,
            include_raw_features=include_raw_features,
            pretrained=pretrained,
        )
    if model_arch == "late_fusion":
        return SiameseEfficientNetLateFusion(
            num_classes=num_classes,
            spatial_dropout_rate=spatial_dropout,
            include_raw_features=include_raw_features,
            pretrained=pretrained,
        )
    raise ValueError(
        f"Unknown model_arch={model_arch!r}. "
        "Use 'resnet18_vector_geometry_fusion', 'vector_geometry_fusion', "
        "'late_fusion' or 'vector_fusion'."
    )


def run_loso_training(
    root_dir,
    num_epochs=20,
    results_dir="results/siamese-efficientnet-vector-geometry",
    patient_list=None,
    spatial_dropout=0.0,
    early_stopping_patience=4,
    batch_size=8,
    learning_rate=3e-4,
    weight_decay=1e-4,
    image_size=224,
    seed=42,
    include_raw_features=False,
    pretrained=True,
    fine_tune_backbone=False,
    model_arch="vector_geometry_fusion",
    use_semantic_augmentation=True,
    max_folds=None,
):
    set_seed(seed)

    root_path = Path(root_dir)
    patient_dirs = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    if patient_list is not None:
        patient_dirs = sorted(patient_list)
    if max_folds is not None:
        patient_dirs = patient_dirs[:max_folds]
    if len(patient_dirs) < 3:
        raise ValueError("Nested LOSO requires at least 3 patients.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms(
        image_size=image_size,
        use_augmentation=use_semantic_augmentation,
    )
    threshold_grid = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_geometry_features = model_arch in {"vector_geometry_fusion", "resnet18_vector_geometry_fusion"}

    config = {
        "model": (
            "SiameseResNet18GeometryFusion"
            if model_arch == "resnet18_vector_geometry_fusion"
            else
            "SiameseEfficientNetGeometryFusion"
            if model_arch == "vector_geometry_fusion"
            else "SiameseEfficientNetLateFusion"
            if model_arch == "late_fusion"
            else "SiameseEfficientNet"
        ),
        "pretrained": pretrained,
        "backbone": "resnet18" if model_arch == "resnet18_vector_geometry_fusion" else "efficientnet_b0",
        "model_arch": model_arch,
        "fusion": (
            "resnet18_vector_absdiff_multiply_plus_geometry_mlp"
            if model_arch == "resnet18_vector_geometry_fusion" and not include_raw_features
            else "resnet18_vector_child_pattern_absdiff_multiply_plus_geometry_mlp"
            if model_arch == "resnet18_vector_geometry_fusion"
            else "vector_absdiff_multiply_plus_geometry_mlp"
            if model_arch == "vector_geometry_fusion" and not include_raw_features
            else "vector_child_pattern_absdiff_multiply_plus_geometry_mlp"
            if model_arch == "vector_geometry_fusion"
            else "feature_map_late_fusion_child_pattern_absdiff_multiply"
            if model_arch == "late_fusion" and include_raw_features
            else "feature_map_late_fusion_absdiff_multiply"
            if model_arch == "late_fusion"
            else "vector_concat_child_pattern_absdiff_multiply"
            if include_raw_features
            else "vector_concat_absdiff_multiply"
        ),
        "geometry_features": {
            "enabled": use_geometry_features,
            "feature_dim": GEOMETRY_FEATURE_DIM if use_geometry_features else 0,
            "source": "deterministic_child_pattern_mask_descriptors",
            "augmentation_applied_to_geometry": False,
        },
        "training": "head_only" if not fine_tune_backbone else "optional_backbone_finetuning",
        "use_semantic_augmentation": use_semantic_augmentation,
        "augmentation": {
            "scope": "train_only",
            "paired_child_pattern_affine": True,
            "paired_child_pattern_blur": True,
            "label_changing_single_branch_transforms": False,
        },
        "image_resize": {
            "method": "resize_long_side_then_black_pad",
            "image_size": image_size,
            "preserve_aspect_ratio": True,
        },
        "num_epochs": num_epochs,
        "early_stopping_patience": early_stopping_patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "threshold_grid": threshold_grid,
        "patients": patient_dirs,
        "labels": ERROR_CATEGORIES,
        "device": str(device),
    }
    write_json(output_dir / "experiment_config.json", config)

    geometry_feature_fn = geometric_features_for_sample if use_geometry_features else None

    full_ds = SiameseBVRTDataset(
        root_dir,
        patient_ids=patient_dirs,
        transform=eval_transform,
        geometry_feature_fn=geometry_feature_fn,
    )
    print(f"Dataset sanity check: {len(full_ds)} samples, {len(patient_dirs)} patients.")

    fold_metrics: List[Dict[str, Any]] = []
    print(f"Starting nested LOSO on {len(patient_dirs)} patients. Device: {device}")

    for fold_idx, test_patient in enumerate(patient_dirs):
        candidates = [p for p in patient_dirs if p != test_patient]
        val_patient = select_validation_patient(candidates, fold_idx)
        train_patients = [p for p in candidates if p != val_patient]
        sanity_check_splits(train_patients, val_patient, test_patient)

        print("\n" + "=" * 72)
        print(f"Fold {fold_idx + 1}/{len(patient_dirs)}")
        print(f"Train patients: {train_patients}")
        print(f"Validation patient: {val_patient}")
        print(f"Test patient: {test_patient}")

        train_ds = SiameseBVRTDataset(
            root_dir,
            patient_ids=train_patients,
            transform=train_transform,
            geometry_feature_fn=geometry_feature_fn,
        )
        val_ds = SiameseBVRTDataset(
            root_dir,
            patient_ids=[val_patient],
            transform=eval_transform,
            geometry_feature_fn=geometry_feature_fn,
        )
        test_ds = SiameseBVRTDataset(
            root_dir,
            patient_ids=[test_patient],
            transform=eval_transform,
            geometry_feature_fn=geometry_feature_fn,
        )

        generator = torch.Generator()
        generator.manual_seed(seed + fold_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        pos_weight = calculate_pos_weights(train_ds.get_labels()).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(
            model_arch=model_arch,
            num_classes=len(ERROR_CATEGORIES),
            spatial_dropout=spatial_dropout,
            include_raw_features=include_raw_features,
            pretrained=pretrained,
            geometry_feature_dim=GEOMETRY_FEATURE_DIM,
        ).to(device)

        model.freeze_backbone()
        if fine_tune_backbone:
            model.unfreeze_blocks([6, 7])

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        best_model_wts: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        history = []

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                freeze_backbone_bn=not fine_tune_backbone,
            )
            val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
            val_preds_05 = apply_thresholds(val_probs, np.full(len(ERROR_CATEGORIES), 0.5))
            val_macro_f1 = f1_score(val_labels, val_preds_05, average="macro", zero_division=0)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_macro_f1_threshold_0_5": float(val_macro_f1),
                    "checkpoint_selected": improved,
                }
            )
            print(
                f"E{epoch + 1:02d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_macro_f1@0.5={val_macro_f1:.4f} "
                f"{'*New Best*' if improved else ''}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

        if best_model_wts is None:
            raise RuntimeError("No checkpoint was selected.")

        model.load_state_dict(best_model_wts)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        tuned_thresholds, threshold_details = tune_thresholds(val_probs, val_labels, threshold_grid)

        test_loss, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
        fixed_thresholds = np.full(len(ERROR_CATEGORIES), 0.5, dtype=np.float32)
        train_reference_labels = labels_from_dataset(train_ds)
        majority_vector = majority_baseline_vector(train_reference_labels)
        baseline_thresholds = np.full(len(ERROR_CATEGORIES), 0.5, dtype=np.float32)

        pattern_majority_preds = pattern_majority_predictions(train_ds, test_ds)
        geometric_logreg_preds = geometric_logreg_predictions(train_ds, test_ds, seed + fold_idx)

        test_metrics = {
            "model_tuned_thresholds": compute_multilabel_metrics(
                test_labels,
                apply_thresholds(test_probs, tuned_thresholds),
                tuned_thresholds,
                "model_tuned_thresholds",
            ),
            "model_threshold_0_5": compute_multilabel_metrics(
                test_labels,
                apply_thresholds(test_probs, fixed_thresholds),
                fixed_thresholds,
                "model_threshold_0_5",
            ),
            "majority_baseline": compute_multilabel_metrics(
                test_labels,
                constant_predictions(test_labels, majority_vector),
                majority_vector,
                "majority_baseline",
            ),
            "pattern_majority_baseline": compute_multilabel_metrics(
                test_labels,
                pattern_majority_preds,
                baseline_thresholds,
                "pattern_majority_baseline",
            ),
            "geometric_logreg_baseline": compute_multilabel_metrics(
                test_labels,
                geometric_logreg_preds,
                baseline_thresholds,
                "geometric_logreg_baseline",
            ),
            "always_positive_baseline": compute_multilabel_metrics(
                test_labels,
                np.ones_like(test_labels, dtype=int),
                np.ones(len(ERROR_CATEGORIES), dtype=np.float32),
                "always_positive_baseline",
            ),
        }

        fold_result = {
            "fold": fold_idx + 1,
            "train_patients": train_patients,
            "val_patient": val_patient,
            "test_patient": test_patient,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "epochs_trained": len(history),
            "history": history,
            "threshold_details": threshold_details,
            "tuned_thresholds": {
                label: float(tuned_thresholds[i]) for i, label in enumerate(ERROR_CATEGORIES)
            },
            "majority_baseline_vector": {
                label: int(majority_vector[i]) for i, label in enumerate(ERROR_CATEGORIES)
            },
            "test_metrics": test_metrics,
        }
        fold_metrics.append(fold_result)

        print(
            f"Test macro F1 tuned={test_metrics['model_tuned_thresholds']['macro_f1']:.4f} | "
            f"@0.5={test_metrics['model_threshold_0_5']['macro_f1']:.4f} | "
            f"pattern={test_metrics['pattern_majority_baseline']['macro_f1']:.4f} | "
            f"geom={test_metrics['geometric_logreg_baseline']['macro_f1']:.4f} | "
            f"majority={test_metrics['majority_baseline']['macro_f1']:.4f} | "
            f"always-positive={test_metrics['always_positive_baseline']['macro_f1']:.4f}"
        )

    summary = summarize_results(fold_metrics, config)
    write_json(output_dir / "fold_metrics.json", fold_metrics)
    write_json(output_dir / "summary_metrics.json", summary)

    write_csv(
        output_dir / "fold_metrics.csv",
        flatten_fold_metrics(fold_metrics),
        [
            "fold",
            "test_patient",
            "val_patient",
            "strategy",
            "macro_f1",
            "micro_f1",
            "weighted_f1",
            "hamming_loss",
            "subset_accuracy",
            "best_val_loss",
            "epochs_trained",
        ],
    )
    write_csv(
        output_dir / "per_label_metrics.csv",
        flatten_per_label_metrics(fold_metrics),
        [
            "fold",
            "test_patient",
            "val_patient",
            "strategy",
            "label",
            "precision",
            "recall",
            "f1",
            "support",
            "actual_positives",
            "predicted_positives",
            "true_positive_matches",
            "threshold",
        ],
    )

    tuned = summary["strategies"]["model_tuned_thresholds"]
    print(f"\nSiamese {config['backbone']} summary:")
    print(f"Macro F1: {tuned['macro_f1_mean']:.4f} +/- {tuned['macro_f1_std']:.4f}")
    print(f"Micro F1: {tuned['micro_f1_mean']:.4f} +/- {tuned['micro_f1_std']:.4f}")
    print(f"Results saved in: {output_dir.resolve()}")
    return {"fold_metrics": fold_metrics, "summary": summary}


if __name__ == "__main__":
    data_path = "data/processed/siemens-net-data"
    if not Path(data_path).exists():
        data_path = "../data/processed/siemens-net-data"

    run_loso_training(
        root_dir=data_path,
        num_epochs=20,
        results_dir="results/siamese-efficientnet-vector-geometry",
        spatial_dropout=0.0,
        early_stopping_patience=4,
        batch_size=8,
        learning_rate=3e-4,
        weight_decay=1e-4,
        image_size=224,
        seed=42,
        include_raw_features=False,
        pretrained=True,
        fine_tune_backbone=False,
        model_arch="vector_geometry_fusion",
        use_semantic_augmentation=True,
    )
