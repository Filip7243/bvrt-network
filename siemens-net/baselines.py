from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import SiameseBVRTDataset


GEOMETRY_FEATURE_DIM = 19


def labels_for_samples(dataset: SiameseBVRTDataset) -> np.ndarray:
    """
    Returns binary labels in the same order as dataset.samples.

    Baselines operate directly on sample metadata and image paths. Keeping label
    extraction here avoids going through Dataset.__getitem__, which would apply
    neural-network image transforms that the baselines do not need.
    """
    return np.asarray([dataset.label_vector_for_sample(sample) for sample in dataset.samples], dtype=np.float32)


def majority_baseline_vector(reference_labels: np.ndarray) -> np.ndarray:
    """Predicts each label as positive if it is positive in at least half of training samples."""
    return (reference_labels.mean(axis=0) >= 0.5).astype(np.float32)


def constant_predictions(labels: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Repeats one fixed multi-label vector for every test sample."""
    return np.tile(vector.reshape(1, -1), (labels.shape[0], 1)).astype(int)


def pattern_majority_predictions(
    train_ds: SiameseBVRTDataset,
    test_ds: SiameseBVRTDataset,
) -> np.ndarray:
    """
    Pattern-only baseline.

    For every BVRT card number p1..p10, this baseline predicts the majority
    error vector observed for the same card in training patients. It never reads
    the tested child's drawing. This is a necessary control because all patients
    solve the same BVRT cards, and some cards may naturally induce specific
    error categories more often than others.
    """
    labels_by_pattern: Dict[int, List[np.ndarray]] = {}
    global_vector = majority_baseline_vector(labels_for_samples(train_ds))

    for sample in train_ds.samples:
        drawing_id = int(sample["drawing_id"])
        labels_by_pattern.setdefault(drawing_id, []).append(
            np.asarray(train_ds.label_vector_for_sample(sample), dtype=np.float32)
        )

    majority_by_pattern = {
        drawing_id: (np.vstack(labels).mean(axis=0) >= 0.5).astype(int)
        for drawing_id, labels in labels_by_pattern.items()
    }

    predictions = []
    for sample in test_ds.samples:
        drawing_id = int(sample["drawing_id"])
        predictions.append(majority_by_pattern.get(drawing_id, global_vector).astype(int))
    return np.vstack(predictions)


def _foreground_mask(path: Path) -> np.ndarray:
    """
    Loads a processed BVRT image as a binary foreground mask.

    The preprocessing stores white strokes on a black background. A low
    threshold is enough and remains robust to anti-aliasing from SVG rendering
    or image resizing.
    """
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8) > 20


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _mask_geometry(mask: np.ndarray) -> Dict[str, float]:
    """
    Computes simple shape descriptors for one binary drawing.

    These descriptors intentionally stay interpretable: foreground area,
    centroid, bounding-box geometry and connected-component count. They are a
    classical baseline, not an attempt to hand-engineer the full BVRT task.
    """
    height, width = mask.shape
    area = float(mask.sum())
    total = float(height * width)

    if area == 0:
        return {
            "area": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "bbox_area": 0.0,
            "aspect": 0.0,
            "components": 0.0,
        }

    ys, xs = np.nonzero(mask)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox_w = x_max - x_min + 1.0
    bbox_h = y_max - y_min + 1.0

    # OpenCV is already part of preprocessing. If it is unavailable in a small
    # runtime, the baseline still works and simply omits component information.
    try:
        import cv2

        components = float(max(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1, 0))
    except Exception:
        components = 0.0

    return {
        "area": area / total,
        "cx": float(xs.mean() / max(width - 1, 1)),
        "cy": float(ys.mean() / max(height - 1, 1)),
        "bbox_w": bbox_w / width,
        "bbox_h": bbox_h / height,
        "bbox_area": (bbox_w * bbox_h) / total,
        "aspect": _safe_div(bbox_w, bbox_h),
        "components": components,
    }


def geometric_features_for_sample(sample: Dict[str, Any]) -> np.ndarray:
    """
    Extracts interpretable geometric comparison features for one BVRT pair.

    The feature vector compares child and pattern masks through overlap,
    child-only strokes, pattern-only strokes, centroid displacement, bounding-box
    changes, aspect-ratio changes and fragmentation. These signals correspond
    directly to BVRT categories such as omissions, displacements and relative
    size errors.
    """
    child = _foreground_mask(sample["child_path"])
    pattern = _foreground_mask(sample["pattern_path"])

    child_g = _mask_geometry(child)
    pattern_g = _mask_geometry(pattern)
    overlap = child & pattern
    union = child | pattern
    child_only = child & ~pattern
    pattern_only = pattern & ~child

    features = [
        child_g["area"],
        pattern_g["area"],
        child_g["bbox_area"],
        pattern_g["bbox_area"],
        _safe_div(child_g["area"], pattern_g["area"]),
        _safe_div(child_g["bbox_area"], pattern_g["bbox_area"]),
        abs(child_g["cx"] - pattern_g["cx"]),
        abs(child_g["cy"] - pattern_g["cy"]),
        abs(child_g["bbox_w"] - pattern_g["bbox_w"]),
        abs(child_g["bbox_h"] - pattern_g["bbox_h"]),
        _safe_div(float(overlap.sum()), float(union.sum())),
        _safe_div(float(child_only.sum()), float(union.sum())),
        _safe_div(float(pattern_only.sum()), float(union.sum())),
        child_g["components"],
        pattern_g["components"],
        abs(child_g["components"] - pattern_g["components"]),
        child_g["aspect"],
        pattern_g["aspect"],
        abs(child_g["aspect"] - pattern_g["aspect"]),
    ]
    if len(features) != GEOMETRY_FEATURE_DIM:
        raise RuntimeError(f"Expected {GEOMETRY_FEATURE_DIM} geometry features, got {len(features)}.")
    return np.asarray(features, dtype=np.float32)


def geometric_feature_matrix(dataset: SiameseBVRTDataset) -> np.ndarray:
    """Builds a matrix of classical geometric features for all samples."""
    return np.vstack([geometric_features_for_sample(sample) for sample in dataset.samples])


def geometric_logreg_predictions(
    train_ds: SiameseBVRTDataset,
    test_ds: SiameseBVRTDataset,
    seed: int,
) -> np.ndarray:
    """
    Classical geometry baseline: independent logistic regression per label.

    The problem is multi-label, so every BVRT error category gets its own
    balanced logistic regression. If a training fold contains only one class for
    a label, the baseline falls back to that constant class rather than failing.
    """
    x_train = geometric_feature_matrix(train_ds)
    y_train = labels_for_samples(train_ds).astype(int)
    x_test = geometric_feature_matrix(test_ds)

    predictions = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=int)
    for label_idx in range(y_train.shape[1]):
        label_values = y_train[:, label_idx]
        unique_values = np.unique(label_values)
        if len(unique_values) == 1:
            predictions[:, label_idx] = int(unique_values[0])
            continue

        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            ),
        )
        classifier.fit(x_train, label_values)
        predictions[:, label_idx] = classifier.predict(x_test).astype(int)

    return predictions
