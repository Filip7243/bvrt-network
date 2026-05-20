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
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import HybridBVRTDataset
from models.hybrid_model import HybridBVRTModel


ERROR_CATEGORIES = [
    "omissions",
    "distortions",
    "perseverations",
    "rotations",
    "displacements",
    "relative_size_errors",
]


class ResizeLongSideAndPad:
    """
    Preserves BVRT geometry by resizing the long side and padding to a square.
    """

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


class BVRTTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        # Backbones are frozen in the main protocol; keep their BN/dropout deterministic.
        if hasattr(self.model, "cnn"):
            self.model.cnn.eval()
        if hasattr(self.model, "vit"):
            self.model.vit.eval()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / len(train_loader.dataset)

    def evaluate(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)

                running_loss += loss.item() * images.size(0)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return (
            running_loss / len(loader.dataset),
            np.vstack(all_probs),
            np.vstack(all_labels),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 224) -> Tuple[Any, Any]:
    base = [ResizeLongSideAndPad(size=image_size, fill=0)]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        base
        + [
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.RandomAffine(
                degrees=1,
                translate=(0.01, 0.01),
                scale=(0.99, 1.01),
                fill=0,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(base + [transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def labels_from_dataset(dataset: HybridBVRTDataset) -> np.ndarray:
    labels = []
    for sample in dataset.samples:
        labels.append(
            [
                1.0 if sample["labels"].get(category, 0) > 0 else 0.0
                for category in ERROR_CATEGORIES
            ]
        )
    return np.asarray(labels, dtype=np.float32)


def select_validation_patient(train_candidates: Sequence[str], fold_idx: int) -> str:
    if not train_candidates:
        raise ValueError("Cannot select validation patient from an empty candidate list.")
    return sorted(train_candidates)[fold_idx % len(train_candidates)]


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
            f1 = f1_score(labels[:, idx], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        tuned[idx] = float(best_threshold)
        details[category] = {
            "threshold": float(best_threshold),
            "best_f1": float(best_f1),
            "fallback": None,
        }

    return tuned, details


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds.reshape(1, -1)).astype(int)


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


def majority_baseline_thresholds(reference_labels: np.ndarray) -> np.ndarray:
    positives = reference_labels.mean(axis=0)
    return (positives >= 0.5).astype(np.float32)


def make_constant_predictions(labels: np.ndarray, label_vector: np.ndarray) -> np.ndarray:
    return np.tile(label_vector.reshape(1, -1), (labels.shape[0], 1)).astype(int)


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


def sanity_check_splits(train_patients: Sequence[str], val_patient: str, test_patient: str) -> None:
    train_set = set(train_patients)
    val_set = {val_patient}
    test_set = {test_patient}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError(
            f"Patient leakage detected: train={train_patients}, val={val_patient}, test={test_patient}"
        )


def run_loso_training(
    data_root: str,
    cnn_type: str = "efficientnet_b0",
    vit_type: str = "vit_b_16",
    pretrained: bool = True,
    num_epochs: int = 20,
    early_stopping_patience: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    seed: int = 42,
    device_name: str = "cuda",
    results_dir: str = "results",
) -> Dict[str, Any]:
    set_seed(seed)

    data_path = Path(data_root)
    patients = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if len(patients) < 3:
        raise ValueError("Nested LOSO requires at least 3 patients.")

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms(image_size=image_size)
    threshold_grid = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "HybridBVRTModel",
        "cnn_type": cnn_type,
        "vit_type": vit_type,
        "pretrained": pretrained,
        "training": "single_phase_frozen_backbones_train_fusion_head",
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
        "patients": patients,
        "device": str(device),
        "labels": ERROR_CATEGORIES,
    }
    write_json(output_dir / "experiment_config.json", config)

    full_ds = HybridBVRTDataset(data_root, transform=eval_transform)
    if len(full_ds) != len(patients) * 10:
        print(f"Ostrzeżenie: dataset ma {len(full_ds)} próbek dla {len(patients)} pacjentów.")
    else:
        print(f"Dataset sanity check: {len(full_ds)} próbek dla {len(patients)} pacjentów.")

    fold_metrics: List[Dict[str, Any]] = []

    print(f"Rozpoczynanie nested LOSO dla {len(patients)} pacjentów na urządzeniu: {device}")
    for fold_idx, test_patient in enumerate(patients):
        candidates = [patient for patient in patients if patient != test_patient]
        val_patient = select_validation_patient(candidates, fold_idx)
        train_patients = [patient for patient in candidates if patient != val_patient]
        sanity_check_splits(train_patients, val_patient, test_patient)

        print("\n" + "=" * 72)
        print(f"Fold {fold_idx + 1}/{len(patients)}")
        print(f"Train patients: {train_patients}")
        print(f"Validation patient: {val_patient}")
        print(f"Test patient: {test_patient}")

        train_ds = HybridBVRTDataset(data_root, patient_ids=train_patients, transform=train_transform)
        val_ds = HybridBVRTDataset(data_root, patient_ids=[val_patient], transform=eval_transform)
        test_ds = HybridBVRTDataset(data_root, patient_ids=[test_patient], transform=eval_transform)

        generator = torch.Generator()
        generator.manual_seed(seed + fold_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        pos_weights = train_ds.get_pos_weights().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        model = HybridBVRTModel(
            cnn_type=cnn_type,
            vit_type=vit_type,
            pretrained=pretrained,
            num_classes=len(ERROR_CATEGORIES),
        )
        model.set_train_phase(1)

        optimizer = optim.AdamW(
            model.get_trainable_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        trainer = BVRTTrainer(model, criterion, optimizer, device)

        best_val_loss = float("inf")
        best_model_wts: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        history = []

        for epoch in range(num_epochs):
            train_loss = trainer.train_one_epoch(train_loader)
            val_loss, val_probs, val_labels = trainer.evaluate(val_loader)
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
            status = "*New Best*" if improved else ""
            print(
                f"E{epoch + 1:02d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | val_macro_f1@0.5={val_macro_f1:.4f} {status}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping po {epoch + 1} epokach.")
                break

        if best_model_wts is None:
            raise RuntimeError("No checkpoint was selected during training.")

        model.load_state_dict(best_model_wts)
        val_loss, val_probs, val_labels = trainer.evaluate(val_loader)
        tuned_thresholds, threshold_details = tune_thresholds(
            val_probs,
            val_labels,
            threshold_grid,
        )

        test_loss, test_probs, test_labels = trainer.evaluate(test_loader)
        fixed_thresholds = np.full(len(ERROR_CATEGORIES), 0.5, dtype=np.float32)
        train_reference_labels = labels_from_dataset(train_ds)
        majority_vector = majority_baseline_thresholds(train_reference_labels)

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
                make_constant_predictions(test_labels, majority_vector),
                majority_vector,
                "majority_baseline",
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
                label: float(tuned_thresholds[idx]) for idx, label in enumerate(ERROR_CATEGORIES)
            },
            "majority_baseline_vector": {
                label: int(majority_vector[idx]) for idx, label in enumerate(ERROR_CATEGORIES)
            },
            "test_metrics": test_metrics,
        }
        fold_metrics.append(fold_result)

        print(
            f"Test macro F1 tuned={test_metrics['model_tuned_thresholds']['macro_f1']:.4f} "
            f"| @0.5={test_metrics['model_threshold_0_5']['macro_f1']:.4f} "
            f"| majority={test_metrics['majority_baseline']['macro_f1']:.4f}"
        )

    summary = summarize_results(fold_metrics, config)
    write_json(output_dir / "fold_metrics.json", fold_metrics)
    write_json(output_dir / "summary_metrics.json", summary)

    fold_rows = flatten_fold_metrics(fold_metrics)
    write_csv(
        output_dir / "fold_metrics.csv",
        fold_rows,
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

    per_label_rows = flatten_per_label_metrics(fold_metrics)
    write_csv(
        output_dir / "per_label_metrics.csv",
        per_label_rows,
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

    print("\nPodsumowanie model_tuned_thresholds:")
    tuned_summary = summary["strategies"]["model_tuned_thresholds"]
    print(f"Macro F1: {tuned_summary['macro_f1_mean']:.4f} +/- {tuned_summary['macro_f1_std']:.4f}")
    print(f"Micro F1: {tuned_summary['micro_f1_mean']:.4f} +/- {tuned_summary['micro_f1_std']:.4f}")
    print(f"Wyniki zapisane w: {output_dir.resolve()}")
    return {"fold_metrics": fold_metrics, "summary": summary}


def plot_results_from_csv(results_dir: str = "results") -> None:
    # Kept intentionally optional: CSV/JSON are the authoritative outputs for paper tables.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fold_csv = Path(results_dir) / "fold_metrics.csv"
    if not fold_csv.exists():
        return

    rows = []
    with open(fold_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["strategy"] == "model_tuned_thresholds":
                rows.append(row)

    if not rows:
        return

    plt.figure(figsize=(12, 5))
    plt.bar([row["test_patient"] for row in rows], [float(row["macro_f1"]) for row in rows])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Macro F1")
    plt.title("LOSO Macro F1 per Test Patient")
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "macro_f1_per_fold.png")
    plt.close()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    data_root = project_root / "data" / "processed" / "vit-resnet-data"
    results_root = project_root / "results"

    results = run_loso_training(
        data_root=str(data_root),
        cnn_type="efficientnet_b0",
        vit_type="vit_b_16",
        pretrained=True,
        num_epochs=20,
        early_stopping_patience=4,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        image_size=224,
        seed=42,
        results_dir=str(results_root),
    )
    plot_results_from_csv(str(results_root))
