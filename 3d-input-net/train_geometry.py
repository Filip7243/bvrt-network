import argparse
import copy
import csv
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


ERROR_CATEGORIES = [
    "omissions",
    "distortions",
    "perseverations",
    "rotations",
    "displacements",
    "relative_size_errors",
]
SCRIPT_DIR = Path(__file__).resolve().parent

# Nazwy są jawne, żeby w razie potrzeby dało się łatwo opisać w pracy magisterskiej,
# co dokładnie dostaje gałąź geometryczna. Wektor jest mały i celowo prosty:
# opisuje globalny kształt rysunku dziecka, wzorca i ich wzajemne położenie.
GEOMETRY_FEATURE_NAMES = [
    "child_area_ratio",
    "child_bbox_area_ratio",
    "child_bbox_width_ratio",
    "child_bbox_height_ratio",
    "child_centroid_x",
    "child_centroid_y",
    "child_component_count_norm",
    "child_aspect_ratio",
    "pattern_area_ratio",
    "pattern_bbox_area_ratio",
    "pattern_bbox_width_ratio",
    "pattern_bbox_height_ratio",
    "pattern_centroid_x",
    "pattern_centroid_y",
    "pattern_component_count_norm",
    "pattern_aspect_ratio",
    "centroid_distance_norm",
    "mask_iou",
    "child_only_area_ratio",
    "pattern_only_area_ratio",
]
GEOMETRY_FEATURE_DIM = len(GEOMETRY_FEATURE_NAMES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    """Obsługuje ścieżki uruchamiane z notebooka, katalogu skryptu albo root repo."""

    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    script_relative = SCRIPT_DIR / candidate
    if script_relative.exists():
        return script_relative

    return candidate


def resolve_output_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "..":
        return SCRIPT_DIR / candidate
    if candidate.parent.exists():
        return candidate
    return SCRIPT_DIR / candidate


class ResizeLongSideAndPad:
    """Skaluje obraz z zachowaniem proporcji i dopadduje czarnym tłem do kwadratu."""

    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = image.resize((new_width, new_height), Image.BILINEAR)

        canvas = Image.new(image.mode, (self.size, self.size), color=self.fill)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(image, (left, top))
        return canvas


def build_transforms(image_size=224, use_augmentation=True):
    # Uwaga: to nie jest zwykłe zdjęcie RGB. Kanały niosą znaczenie: rysunek,
    # wzorzec i mapa różnic. Dlatego nie używamy ColorJitter, bo zmieniałby
    # intensywności kanałów semantycznych i mógłby sztucznie osłabić np. diff-mapę.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_steps = [ResizeLongSideAndPad(size=image_size, fill=0)]
    if use_augmentation:
        # Ta augmentacja jest semantycznie bezpieczna dla 3-kanałowego wejścia,
        # bo ta sama transformacja przestrzenna działa jednocześnie na wszystkie
        # kanały. Nie zmienia więc relacji rysunek-wzorzec-diff.
        train_steps.append(
            transforms.RandomAffine(
                degrees=1.0,
                translate=(0.01, 0.01),
                scale=(0.99, 1.01),
                shear=0.0,
                fill=0,
            )
        )
    train_steps.extend([transforms.ToTensor(), normalize])

    eval_transform = transforms.Compose(
        [
            ResizeLongSideAndPad(size=image_size, fill=0),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def _labels_to_vector(labels):
    return np.array(
        [1 if labels.get(category, 0) > 0 else 0 for category in ERROR_CATEGORIES],
        dtype=np.int64,
    )


def labels_for_samples(samples):
    return np.stack([_labels_to_vector(sample["labels"]) for sample in samples]).astype(np.int64)


def majority_baseline_vector(reference_labels):
    positives = reference_labels.sum(axis=0)
    negatives = reference_labels.shape[0] - positives
    return (positives >= negatives).astype(np.int64)


def constant_predictions(labels, vector):
    return np.tile(vector.reshape(1, -1), (labels.shape[0], 1)).astype(np.int64)


def pattern_majority_predictions(train_samples, test_samples):
    """Baseline: dla każdego wzorca p1..p10 przewiduje większość etykiet z treningu.

    Ten baseline jest ważniejszy niż globalna większość, bo BVRT ma różne wzorce
    o różnym poziomie trudności. Jeśli model nie przebija tego punktu odniesienia,
    może tylko uczyć się, że np. dla danego wzorca często występują przemieszczenia.
    """

    global_labels = labels_for_samples(train_samples)
    fallback = majority_baseline_vector(global_labels)

    labels_by_pattern = {}
    for sample in train_samples:
        labels_by_pattern.setdefault(sample["drawing_idx"], []).append(_labels_to_vector(sample["labels"]))

    majority_by_pattern = {}
    for drawing_idx, labels in labels_by_pattern.items():
        majority_by_pattern[drawing_idx] = majority_baseline_vector(np.stack(labels))

    predictions = []
    for sample in test_samples:
        predictions.append(majority_by_pattern.get(sample["drawing_idx"], fallback))
    return np.stack(predictions).astype(np.int64)


def _foreground_mask(channel, threshold=20):
    return np.asarray(channel > threshold, dtype=bool)


def _mask_geometry(mask):
    height, width = mask.shape
    canvas_area = float(max(height * width, 1))
    area = float(mask.sum())
    area_ratio = area / canvas_area

    if area == 0:
        # Puste maski nie powinny często wystąpić, ale jawny wektor zerowy jest
        # stabilniejszy niż NaN-y w małym zbiorze.
        return np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    ys, xs = np.where(mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_width = float(x_max - x_min + 1)
    bbox_height = float(y_max - y_min + 1)
    bbox_area_ratio = (bbox_width * bbox_height) / canvas_area
    bbox_width_ratio = bbox_width / max(float(width), 1.0)
    bbox_height_ratio = bbox_height / max(float(height), 1.0)
    centroid_x = float(xs.mean()) / max(float(width - 1), 1.0)
    centroid_y = float(ys.mean()) / max(float(height - 1), 1.0)

    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    component_count = max(num_labels - 1, 0)
    component_count_norm = min(component_count, 10) / 10.0
    aspect_ratio = bbox_width / max(bbox_height, 1.0)

    return np.array(
        [
            area_ratio,
            bbox_area_ratio,
            bbox_width_ratio,
            bbox_height_ratio,
            centroid_x,
            centroid_y,
            component_count_norm,
            aspect_ratio,
        ],
        dtype=np.float32,
    )


def geometric_features_for_image(img_path):
    """Wyciąga cechy geometryczne z pliku 3-kanałowego utworzonego przez OpenCV.

    Pliki w `3d-input-data` są zapisywane przez `cv2.merge([child, pattern, diff])`.
    Po stronie OpenCV oznacza to kanały B=child, G=pattern, R=diff. Do geometrii
    czytamy obraz przez `cv2.imread`, żeby zachować tę kolejność bez niejawnej
    konwersji PIL RGB.
    """

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {img_path}")

    child_mask = _foreground_mask(image_bgr[:, :, 0])
    pattern_mask = _foreground_mask(image_bgr[:, :, 1])

    child_geom = _mask_geometry(child_mask)
    pattern_geom = _mask_geometry(pattern_mask)

    child_centroid = child_geom[4:6]
    pattern_centroid = pattern_geom[4:6]
    centroid_distance = float(np.linalg.norm(child_centroid - pattern_centroid) / np.sqrt(2.0))

    intersection = np.logical_and(child_mask, pattern_mask).sum()
    union = np.logical_or(child_mask, pattern_mask).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    canvas_area = float(max(child_mask.size, 1))
    child_only = float(np.logical_and(child_mask, np.logical_not(pattern_mask)).sum() / canvas_area)
    pattern_only = float(np.logical_and(pattern_mask, np.logical_not(child_mask)).sum() / canvas_area)

    features = np.concatenate(
        [
            child_geom,
            pattern_geom,
            np.array([centroid_distance, iou, child_only, pattern_only], dtype=np.float32),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def geometric_feature_matrix(samples):
    return np.stack([geometric_features_for_image(sample["img_path"]) for sample in samples]).astype(np.float32)


def geometric_logreg_predictions(train_samples, test_samples):
    """Klasyczny baseline geometryczny: 6 niezależnych regresji logistycznych.

    Przy 14 pacjentach ten baseline jest bardzo przydatny diagnostycznie. Jeśli
    prosta geometria wygrywa z CNN, to CNN najpewniej nie wykorzystuje dobrze
    struktury przestrzennej albo wariancja treningu jest za duża.
    """

    x_train = geometric_feature_matrix(train_samples)
    y_train = labels_for_samples(train_samples)
    x_test = geometric_feature_matrix(test_samples)

    predictions = np.zeros((len(test_samples), len(ERROR_CATEGORIES)), dtype=np.int64)
    for label_idx in range(len(ERROR_CATEGORIES)):
        y_label = y_train[:, label_idx]
        if len(np.unique(y_label)) < 2:
            predictions[:, label_idx] = int(y_label[0])
            continue

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=42,
            ),
        )
        classifier.fit(x_train, y_label)
        predictions[:, label_idx] = classifier.predict(x_test)

    return predictions


class BVRTDataset(Dataset):
    """Dataset dla przetworzonych obrazów 3-kanałowych BVRT.

    Jeśli `use_geometry_features=True`, zwracamy trójkę:
    `(image_tensor, geometry_tensor, target_tensor)`.
    W przeciwnym razie zachowujemy standardowe `(image_tensor, target_tensor)`.
    """

    def __init__(self, root_dir, patient_ids=None, transform=None, use_geometry_features=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_geometry_features = use_geometry_features
        self.samples = []

        patient_filter = set(patient_ids) if patient_ids is not None else None
        patient_dirs = [path for path in sorted(self.root_dir.iterdir()) if path.is_dir()]

        for patient_dir in patient_dirs:
            if patient_filter is not None and patient_dir.name not in patient_filter:
                continue

            labels_file = patient_dir / "labels.json"
            if not labels_file.exists():
                continue

            with labels_file.open("r", encoding="utf-8") as handle:
                labels_data = json.load(handle)

            drawings_labels = {
                int(drawing["drawing_id"]): drawing.get("errors", {})
                for drawing in labels_data.get("drawings", [])
            }

            for img_path in sorted(patient_dir.glob("*.png")):
                match = re.search(r"_p(\d+)\.png$", img_path.name)
                if not match:
                    continue

                drawing_idx = int(match.group(1))
                if drawing_idx not in drawings_labels:
                    continue

                self.samples.append(
                    {
                        "img_path": img_path,
                        "labels": drawings_labels[drawing_idx],
                        "patient": patient_dir.name,
                        "drawing_idx": drawing_idx,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(_labels_to_vector(sample["labels"]), dtype=torch.float32)

        if not self.use_geometry_features:
            return image, target

        geometry = torch.tensor(geometric_features_for_image(sample["img_path"]), dtype=torch.float32)
        return image, geometry, target

    def get_labels(self):
        return labels_for_samples(self.samples)


class ResNet18Transfer(nn.Module):
    """Bazowy wariant obrazowy, zgodny z poprzednim eksperymentem 3D."""

    def __init__(
        self,
        num_classes=6,
        freeze_backbone=True,
        unfreeze_layer4=False,
        dropout=0.35,
        pretrained=True,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone

        if unfreeze_layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, geometry=None):
        return self.model(image)


class ResNet18GeometryFusion(nn.Module):
    """ResNet18 + mały MLP dla geometrii, fuzja na wektorach cech.

    ResNet produkuje 512 cech obrazu. Geometria ma tylko 20 wymiarów, więc
    przechodzi przez bardzo mały MLP. Głowa klasyfikacyjna jest celowo mała,
    bo przy LOSO i n=14 większa głowa szybciej uczy się przypadkowych wzorców.
    """

    def __init__(
        self,
        num_classes=6,
        geometry_dim=GEOMETRY_FEATURE_DIM,
        freeze_backbone=True,
        unfreeze_layer4=False,
        dropout=0.35,
        pretrained=True,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        if unfreeze_layer4:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        image_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.image_projection = nn.Sequential(
            nn.Linear(image_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, 16),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, geometry):
        image_features = self.image_projection(self.backbone(image))
        geometry_features = self.geometry_projection(geometry)
        fused_features = torch.cat([image_features, geometry_features], dim=1)
        return self.classifier(fused_features)


def calculate_pos_weights(labels):
    positives = labels.sum(axis=0).astype(np.float32)
    negatives = labels.shape[0] - positives
    weights = np.ones_like(positives, dtype=np.float32)

    valid = positives > 0
    weights[valid] = negatives[valid] / np.maximum(positives[valid], 1.0)
    # Mały zbiór potrafi generować skrajne wagi. Clamp ogranicza niestabilne
    # skoki gradientu dla rzadkich etykiet, zwłaszcza rotacji i perseweracji.
    return torch.tensor(np.clip(weights, 0.25, 10.0), dtype=torch.float32)


def unpack_batch(batch, device):
    if len(batch) == 2:
        images, labels = batch
        return images.to(device), None, labels.to(device)

    images, geometry, labels = batch
    return images.to(device), geometry.to(device), labels.to(device)


class BVRTTrainer:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def _forward(self, images, geometry):
        if geometry is None:
            return self.model(images)
        return self.model(images, geometry)

    def train_one_epoch(self, train_loader):
        self.model.train()
        keep_frozen_batchnorm_eval(self.model)
        running_loss = 0.0

        for batch in train_loader:
            images, geometry, labels = unpack_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward(images, geometry)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)

        return running_loss / max(len(train_loader.dataset), 1)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        losses = []
        probabilities = []
        labels_all = []

        for batch in loader:
            images, geometry, labels = unpack_batch(batch, self.device)
            logits = self._forward(images, geometry)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()) * labels.size(0))
            probabilities.append(probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

        mean_loss = float(np.sum(losses) / max(len(loader.dataset), 1))
        return mean_loss, np.vstack(probabilities), np.vstack(labels_all).astype(np.int64)


def apply_thresholds(probs, thresholds):
    return (probs >= thresholds.reshape(1, -1)).astype(np.int64)


def tune_thresholds(probs, labels, threshold_grid):
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    details = {}

    for idx, label_name in enumerate(ERROR_CATEGORIES):
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in threshold_grid:
            pred = (probs[:, idx] >= threshold).astype(np.int64)
            f1 = f1_score(labels[:, idx], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        thresholds[idx] = best_threshold
        details[label_name] = {
            "threshold": float(best_threshold),
            "val_f1": float(best_f1),
            "val_support": int(labels[:, idx].sum()),
        }

    return thresholds, details


def compute_multilabel_metrics(labels, preds_binary, thresholds, strategy):
    per_label = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds_binary,
        average=None,
        zero_division=0,
    )

    for idx, label_name in enumerate(ERROR_CATEGORIES):
        per_label[label_name] = {
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
        "macro_precision": float(precision_score(labels, preds_binary, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds_binary, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds_binary)),
        "subset_accuracy": float(accuracy_score(labels, preds_binary)),
        "per_label": per_label,
    }


def should_replace_checkpoint(
    checkpoint_metric,
    val_loss,
    val_macro_f1,
    best_val_loss,
    best_val_macro_f1,
):
    if checkpoint_metric == "val_loss":
        return val_loss < best_val_loss

    # Domyślnie wybieramy model po macro F1, bo cel pracy jest multi-label i
    # interesują nas słabsze klasy, nie tylko dominujące etykiety. Val loss jest
    # tie-breakerem, żeby wybór był deterministyczny przy takim samym F1.
    if val_macro_f1 > best_val_macro_f1 + 1e-8:
        return True
    if abs(val_macro_f1 - best_val_macro_f1) <= 1e-8 and val_loss < best_val_loss:
        return True
    return False


def build_model(model_arch, pretrained, freeze_backbone, unfreeze_layer4, dropout):
    if model_arch == "resnet18":
        return ResNet18Transfer(
            num_classes=len(ERROR_CATEGORIES),
            freeze_backbone=freeze_backbone,
            unfreeze_layer4=unfreeze_layer4,
            dropout=dropout,
            pretrained=pretrained,
        )
    if model_arch == "resnet18_geometry":
        return ResNet18GeometryFusion(
            num_classes=len(ERROR_CATEGORIES),
            freeze_backbone=freeze_backbone,
            unfreeze_layer4=unfreeze_layer4,
            dropout=dropout,
            pretrained=pretrained,
        )
    raise ValueError(f"Nieznany model_arch: {model_arch}")


def collect_trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def keep_frozen_batchnorm_eval(model):
    # Przy zamrożonym backbone nie chcemy aktualizować statystyk BatchNorm na
    # kilkudziesięciu obrazach. To stabilizuje transfer learning w LOSO.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            params = list(module.parameters(recurse=False))
            if params and not any(param.requires_grad for param in params):
                module.eval()


def summarize_results(fold_results):
    strategies = sorted(fold_results[0]["test_metrics"].keys())
    summary = {"strategies": {}}

    for strategy in strategies:
        strategy_metrics = [fold["test_metrics"][strategy] for fold in fold_results]
        metric_summary = {}
        for metric_name in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "subset_accuracy"]:
            values = np.array([metrics[metric_name] for metrics in strategy_metrics], dtype=np.float32)
            metric_summary[f"{metric_name}_mean"] = float(values.mean())
            metric_summary[f"{metric_name}_std"] = float(values.std(ddof=0))

        per_label = {}
        for label_name in ERROR_CATEGORIES:
            values = np.array(
                [metrics["per_label"][label_name]["f1"] for metrics in strategy_metrics],
                dtype=np.float32,
            )
            per_label[label_name] = {
                "f1_mean": float(values.mean()),
                "f1_std": float(values.std(ddof=0)),
            }
        metric_summary["per_label"] = per_label
        summary["strategies"][strategy] = metric_summary

    return summary


def save_results(results_dir, config, fold_results, summary):
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    with (results_dir / "fold_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_results, handle, ensure_ascii=False, indent=2)

    with (results_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, **summary}, handle, ensure_ascii=False, indent=2)

    fold_rows = []
    per_label_rows = []
    for fold_result in fold_results:
        for strategy, metrics in fold_result["test_metrics"].items():
            fold_rows.append(
                {
                    "fold": fold_result["fold"],
                    "test_patient": fold_result["test_patient"],
                    "val_patient": fold_result["val_patient"],
                    "strategy": strategy,
                    "macro_f1": metrics["macro_f1"],
                    "micro_f1": metrics["micro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "hamming_loss": metrics["hamming_loss"],
                    "subset_accuracy": metrics["subset_accuracy"],
                    "best_val_loss": fold_result["best_val_loss"],
                    "best_val_macro_f1": fold_result["best_val_macro_f1"],
                    "epochs_trained": fold_result["epochs_trained"],
                }
            )

            for label_name, label_metrics in metrics["per_label"].items():
                per_label_rows.append(
                    {
                        "fold": fold_result["fold"],
                        "test_patient": fold_result["test_patient"],
                        "val_patient": fold_result["val_patient"],
                        "strategy": strategy,
                        "label": label_name,
                        **label_metrics,
                    }
                )

    with (results_dir / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    with (results_dir / "per_label_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_label_rows)


def run_loso_training_scientific(
    root_dir="../data/processed/3d-input-data",
    results_dir="../results/3d-input-resnet-geometry",
    model_arch="resnet18_geometry",
    num_epochs=25,
    early_stopping_patience=4,
    checkpoint_metric="val_macro_f1",
    batch_size=8,
    learning_rate=3e-4,
    weight_decay=1e-4,
    dropout=0.35,
    image_size=224,
    pretrained=True,
    freeze_backbone=True,
    unfreeze_layer4=False,
    use_augmentation=True,
    seed=42,
    device=None,
    num_workers=0,
    max_folds=None,
):
    set_seed(seed)
    root_dir = resolve_path(root_dir)
    results_dir = resolve_output_path(results_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    threshold_grid = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)]

    patients = sorted([path.name for path in root_dir.iterdir() if path.is_dir()])
    if max_folds is not None:
        patients = patients[:max_folds]

    train_transform, eval_transform = build_transforms(image_size=image_size, use_augmentation=use_augmentation)
    use_geometry_features = model_arch == "resnet18_geometry"

    config = {
        "model": model_arch,
        "pretrained": pretrained,
        "training": "head_only_transfer_learning" if freeze_backbone and not unfreeze_layer4 else "partial_finetuning",
        "freeze_backbone": freeze_backbone,
        "unfreeze_layer4": unfreeze_layer4,
        "input_channels": "processed_png_cv2_merge_child_pattern_diff_read_by_PIL_for_CNN",
        "geometry_features": GEOMETRY_FEATURE_NAMES if use_geometry_features else [],
        "geometry_channel_source": "cv2_BGR_channels_B_child_G_pattern_R_diff",
        "image_resize": {
            "method": "resize_long_side_then_black_pad",
            "image_size": image_size,
            "preserve_aspect_ratio": True,
        },
        "augmentation": {
            "enabled": use_augmentation,
            "spatial_affine": "degrees=1, translate=1%, scale=99-101%",
            "color_jitter": "disabled_semantic_channels",
        },
        "num_epochs": num_epochs,
        "early_stopping_patience": early_stopping_patience,
        "checkpoint_metric": checkpoint_metric,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "seed": seed,
        "threshold_grid": threshold_grid,
        "patients": patients,
        "labels": ERROR_CATEGORIES,
        "device": device,
    }

    fold_results = []
    for fold_idx, test_patient in enumerate(patients, start=1):
        val_patient = patients[fold_idx % len(patients)]
        train_patients = [patient for patient in patients if patient not in {test_patient, val_patient}]

        print(f"\nFold {fold_idx}/{len(patients)} | test={test_patient} | val={val_patient}")

        train_ds = BVRTDataset(
            root_dir,
            patient_ids=train_patients,
            transform=train_transform,
            use_geometry_features=use_geometry_features,
        )
        val_ds = BVRTDataset(
            root_dir,
            patient_ids=[val_patient],
            transform=eval_transform,
            use_geometry_features=use_geometry_features,
        )
        test_ds = BVRTDataset(
            root_dir,
            patient_ids=[test_patient],
            transform=eval_transform,
            use_geometry_features=use_geometry_features,
        )

        generator = torch.Generator()
        generator.manual_seed(seed + fold_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        pos_weights = calculate_pos_weights(train_ds.get_labels()).to(device)
        model = build_model(
            model_arch=model_arch,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_layer4=unfreeze_layer4,
            dropout=dropout,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = torch.optim.AdamW(
            collect_trainable_parameters(model),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        trainer = BVRTTrainer(model, device, criterion, optimizer)

        best_state = None
        best_val_loss = float("inf")
        best_val_macro_f1 = -1.0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, num_epochs + 1):
            train_loss = trainer.train_one_epoch(train_loader)
            val_loss, val_probs, val_labels = trainer.predict(val_loader)
            val_preds_05 = apply_thresholds(val_probs, np.full(len(ERROR_CATEGORIES), 0.5, dtype=np.float32))
            val_macro_f1 = f1_score(val_labels, val_preds_05, average="macro", zero_division=0)

            improved = should_replace_checkpoint(
                checkpoint_metric=checkpoint_metric,
                val_loss=val_loss,
                val_macro_f1=val_macro_f1,
                best_val_loss=best_val_loss,
                best_val_macro_f1=best_val_macro_f1,
            )
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss
                best_val_macro_f1 = float(val_macro_f1)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_macro_f1_threshold_0_5": float(val_macro_f1),
                    "checkpoint": bool(improved),
                }
            )

            marker = " *New Best*" if improved else ""
            print(
                f"E{epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| val_macro_f1@0.5={val_macro_f1:.4f}{marker}"
            )

            if epochs_without_improvement >= early_stopping_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_loss, val_probs, val_labels = trainer.predict(val_loader)
        test_loss, test_probs, test_labels = trainer.predict(test_loader)
        tuned_thresholds, threshold_details = tune_thresholds(val_probs, val_labels, threshold_grid)
        fixed_thresholds = np.full(len(ERROR_CATEGORIES), 0.5, dtype=np.float32)

        train_reference_labels = train_ds.get_labels()
        majority_vector = majority_baseline_vector(train_reference_labels)
        pattern_majority_preds = pattern_majority_predictions(train_ds.samples, test_ds.samples)
        geometric_logreg_preds = geometric_logreg_predictions(train_ds.samples, test_ds.samples)

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
                majority_vector.astype(np.float32),
                "majority_baseline",
            ),
            "pattern_majority_baseline": compute_multilabel_metrics(
                test_labels,
                pattern_majority_preds,
                np.full(len(ERROR_CATEGORIES), np.nan, dtype=np.float32),
                "pattern_majority_baseline",
            ),
            "geometric_logreg_baseline": compute_multilabel_metrics(
                test_labels,
                geometric_logreg_preds,
                np.full(len(ERROR_CATEGORIES), np.nan, dtype=np.float32),
                "geometric_logreg_baseline",
            ),
            "always_positive_baseline": compute_multilabel_metrics(
                test_labels,
                np.ones_like(test_labels, dtype=np.int64),
                np.ones(len(ERROR_CATEGORIES), dtype=np.float32),
                "always_positive_baseline",
            ),
        }

        fold_result = {
            "fold": fold_idx,
            "test_patient": test_patient,
            "val_patient": val_patient,
            "train_patients": train_patients,
            "best_val_loss": float(best_val_loss),
            "best_val_macro_f1": float(best_val_macro_f1),
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
        fold_results.append(fold_result)

        print(
            f"Test macro F1 tuned={test_metrics['model_tuned_thresholds']['macro_f1']:.4f} | "
            f"@0.5={test_metrics['model_threshold_0_5']['macro_f1']:.4f} | "
            f"pattern={test_metrics['pattern_majority_baseline']['macro_f1']:.4f} | "
            f"geometry-logreg={test_metrics['geometric_logreg_baseline']['macro_f1']:.4f}"
        )

    summary = summarize_results(fold_results)
    save_results(results_dir, config, fold_results, summary)

    tuned = summary["strategies"]["model_tuned_thresholds"]
    print("\nPodsumowanie 3D ResNet18 + geometria model_tuned_thresholds:")
    print(f"Macro F1: {tuned['macro_f1_mean']:.4f} +/- {tuned['macro_f1_std']:.4f}")
    print(f"Micro F1: {tuned['micro_f1_mean']:.4f} +/- {tuned['micro_f1_std']:.4f}")
    print(f"Results saved in: {results_dir.resolve()}")
    return {"config": config, "fold_results": fold_results, "summary": summary}


def parse_args():
    parser = argparse.ArgumentParser(description="LOSO training for 3D BVRT input with geometry fusion.")
    parser.add_argument("--root-dir", default="../data/processed/3d-input-data")
    parser.add_argument("--results-dir", default="../results/3d-input-resnet-geometry")
    parser.add_argument("--model-arch", default="resnet18_geometry", choices=["resnet18", "resnet18_geometry"])
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--checkpoint-metric", default="val_macro_f1", choices=["val_macro_f1", "val_loss"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loso_training_scientific(
        root_dir=args.root_dir,
        results_dir=args.results_dir,
        model_arch=args.model_arch,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_metric=args.checkpoint_metric,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
        unfreeze_layer4=args.unfreeze_layer4,
        use_augmentation=not args.no_augmentation,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        max_folds=args.max_folds,
    )
