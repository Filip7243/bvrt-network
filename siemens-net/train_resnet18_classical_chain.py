import argparse
import csv
import json
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from baselines import (
    GEOMETRY_FEATURE_DIM,
    constant_predictions,
    geometric_features_for_sample,
    labels_for_samples,
    majority_baseline_vector,
    pattern_majority_predictions,
)
from dataset import SiameseBVRTDataset
from train import ERROR_CATEGORIES, ResizeLongSideAndPad, set_seed


STROKE_FEATURE_NAMES = [
    "duration_s",
    "actual_drawing_duration_s",
    "planning_latency_s",
    "interruptions_count",
    "interruption_total_s",
    "interruption_mean_s",
    "undo_count",
    "redo_count",
    "overdrawing_score",
    "revisits_count",
    "shading_detected",
    "direction_changes_count",
    "directional_reversals_count",
    "rapid_velocity_changes_count",
    "efficiency_ratio",
    "max_local_density",
    "avg_velocity",
    "max_velocity",
    "velocity_ratio",
    "velocity_std",
    "velocity_p25",
    "velocity_p50",
    "velocity_p75",
    "velocity_iqr",
    "stroke_count",
    "point_count",
    "total_path_length_norm",
    "straight_line_distance_norm",
    "path_efficiency",
    "stroke_bbox_area_ratio",
    "stroke_bbox_width_ratio",
    "stroke_bbox_height_ratio",
    "stroke_centroid_x",
    "stroke_centroid_y",
    "mean_stroke_length_norm",
    "std_stroke_length_norm",
    "mean_points_per_stroke_norm",
]

STRONG_GEOMETRY_FEATURE_NAMES = [
    "child_perimeter_ratio",
    "child_contour_count_norm",
    "child_largest_contour_area_ratio",
    "child_hull_area_ratio",
    "child_solidity",
    "child_extent",
    "child_circularity",
    "pattern_perimeter_ratio",
    "pattern_contour_count_norm",
    "pattern_largest_contour_area_ratio",
    "pattern_hull_area_ratio",
    "pattern_solidity",
    "pattern_extent",
    "pattern_circularity",
    "contour_area_absdiff_ratio",
    "perimeter_absdiff_ratio",
    "bbox_width_absdiff_ratio",
    "bbox_height_absdiff_ratio",
    "bbox_area_absdiff_ratio",
    "aspect_absdiff",
    "symmetric_chamfer_norm",
    "child_to_pattern_chamfer_norm",
    "pattern_to_child_chamfer_norm",
] + [f"child_hu_{idx}" for idx in range(1, 8)] + [f"pattern_hu_{idx}" for idx in range(1, 8)] + [
    f"hu_absdiff_{idx}" for idx in range(1, 8)
]


def resolve_input_path(path):
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    script_relative = Path(__file__).resolve().parent / candidate
    if script_relative.exists():
        return script_relative
    return candidate


def resolve_output_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "..":
        return Path(__file__).resolve().parent / candidate
    return candidate


def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def value_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return float(values.std()), float(p25), float(p50), float(p75), float(p75 - p25)


@lru_cache(maxsize=None)
def summary_by_drawing(test_dir):
    summary_path = Path(test_dir) / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {int(drawing["index"]): drawing for drawing in summary.get("drawings", [])}


def drawing_for_sample(sample):
    return summary_by_drawing(str(Path(sample["child_path"]).parent.parent)).get(int(sample["drawing_id"]), {})


def flatten_stroke_points(strokes_data):
    points = []
    stroke_lengths = []
    points_per_stroke = []
    for stroke in strokes_data or []:
        if not stroke:
            continue
        arr = np.asarray(stroke, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            continue
        xy = arr[:, 2:4]
        points.append(xy)
        points_per_stroke.append(len(xy))
        if len(xy) > 1:
            stroke_lengths.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
        else:
            stroke_lengths.append(0.0)
    if not points:
        return np.zeros((0, 2), dtype=np.float32), [], []
    return np.vstack(points).astype(np.float32), stroke_lengths, points_per_stroke


def stroke_features_for_sample(sample):
    """Behavioral/vector features from summary.json strokes_data."""

    drawing = drawing_for_sample(sample)
    display = drawing.get("display_info", {})
    width = safe_float(display.get("window_width"), 1.0)
    height = safe_float(display.get("window_height"), 1.0)
    canvas_area = max(width * height, 1.0)
    diag = max(float(np.hypot(width, height)), 1.0)

    started_at = safe_float(drawing.get("started_at"))
    first_stroke_at = safe_float(drawing.get("first_stroke_at"), started_at)
    planning_latency = max(first_stroke_at - started_at, 0.0)

    interruption_durations = [safe_float(value) for value in drawing.get("interruption_durations", [])]
    interruption_total = float(np.sum(interruption_durations)) if interruption_durations else 0.0
    interruption_mean = float(np.mean(interruption_durations)) if interruption_durations else 0.0

    velocity_std, velocity_p25, velocity_p50, velocity_p75, velocity_iqr = value_stats(
        [safe_float(value) for value in drawing.get("velocities", [])]
    )

    points, stroke_lengths, points_per_stroke = flatten_stroke_points(drawing.get("strokes_data", []))
    total_path_length = float(np.sum(stroke_lengths)) if stroke_lengths else 0.0
    if len(points) > 0:
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        bbox_width = max(float(x_max - x_min), 0.0)
        bbox_height = max(float(y_max - y_min), 0.0)
        centroid_x = float(points[:, 0].mean() / max(width, 1.0))
        centroid_y = float(points[:, 1].mean() / max(height, 1.0))
        straight_distance = float(np.linalg.norm(points[-1] - points[0]))
    else:
        bbox_width = 0.0
        bbox_height = 0.0
        centroid_x = 0.5
        centroid_y = 0.5
        straight_distance = 0.0

    mean_stroke_length = float(np.mean(stroke_lengths)) if stroke_lengths else 0.0
    std_stroke_length = float(np.std(stroke_lengths)) if stroke_lengths else 0.0
    mean_points_per_stroke = float(np.mean(points_per_stroke)) if points_per_stroke else 0.0
    path_efficiency = straight_distance / total_path_length if total_path_length > 0 else 0.0

    features = np.asarray(
        [
            safe_float(drawing.get("duration_s")),
            safe_float(drawing.get("actual_drawing_duration_s")),
            planning_latency,
            safe_float(drawing.get("interruptions_count")),
            interruption_total,
            interruption_mean,
            safe_float(drawing.get("undo_count")),
            safe_float(drawing.get("redo_count")),
            safe_float(drawing.get("overdrawing_score")),
            safe_float(drawing.get("revisits_count")),
            1.0 if drawing.get("shading_detected") else 0.0,
            safe_float(drawing.get("direction_changes_count")),
            safe_float(drawing.get("directional_reversals_count")),
            safe_float(drawing.get("rapid_velocity_changes_count")),
            safe_float(drawing.get("efficiency_ratio")),
            safe_float(drawing.get("max_local_density")),
            safe_float(drawing.get("avg_velocity")),
            safe_float(drawing.get("max_velocity")),
            safe_float(drawing.get("velocity_ratio")),
            velocity_std,
            velocity_p25,
            velocity_p50,
            velocity_p75,
            velocity_iqr,
            float(len(stroke_lengths)),
            float(len(points)) / 1000.0,
            total_path_length / diag,
            straight_distance / diag,
            path_efficiency,
            (bbox_width * bbox_height) / canvas_area,
            bbox_width / max(width, 1.0),
            bbox_height / max(height, 1.0),
            centroid_x,
            centroid_y,
            mean_stroke_length / diag,
            std_stroke_length / diag,
            mean_points_per_stroke / 1000.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def foreground_mask(path):
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > 20).astype(np.uint8)


def contour_stats(mask):
    height, width = mask.shape
    canvas_area = float(max(height * width, 1))
    diag = float(max(np.hypot(width, height), 1.0))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)

    areas = np.asarray([cv2.contourArea(contour) for contour in contours], dtype=np.float32)
    perimeters = np.asarray([cv2.arcLength(contour, True) for contour in contours], dtype=np.float32)
    largest = contours[int(np.argmax(areas))]
    largest_area = float(areas.max())
    perimeter = float(perimeters.sum())
    _, _, bbox_width, bbox_height = cv2.boundingRect(largest)
    bbox_area = float(max(bbox_width * bbox_height, 1))
    hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
    solidity = largest_area / hull_area if hull_area > 0 else 0.0
    extent = largest_area / bbox_area
    circularity = (4.0 * np.pi * largest_area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8))).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    shape = np.asarray(
        [
            perimeter / diag,
            min(len(contours), 20) / 20.0,
            largest_area / canvas_area,
            hull_area / canvas_area,
            solidity,
            extent,
            circularity,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(shape), np.nan_to_num(hu.astype(np.float32))


def bbox_stats(mask):
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    bbox_width = float(xs.max() - xs.min() + 1) / max(float(width), 1.0)
    bbox_height = float(ys.max() - ys.min() + 1) / max(float(height), 1.0)
    bbox_area = bbox_width * bbox_height
    return bbox_width, bbox_height, bbox_area, bbox_width / max(bbox_height, 1e-6)


def chamfer_distances(child_mask, pattern_mask):
    height, width = child_mask.shape
    diag = float(max(np.hypot(width, height), 1.0))
    child_points = child_mask > 0
    pattern_points = pattern_mask > 0
    if child_points.sum() == 0 or pattern_points.sum() == 0:
        return 1.0, 1.0, 1.0

    child_to_pattern_map = cv2.distanceTransform((1 - pattern_mask).astype(np.uint8), cv2.DIST_L2, 3)
    pattern_to_child_map = cv2.distanceTransform((1 - child_mask).astype(np.uint8), cv2.DIST_L2, 3)
    child_to_pattern = float(child_to_pattern_map[child_points].mean() / diag)
    pattern_to_child = float(pattern_to_child_map[pattern_points].mean() / diag)
    return (child_to_pattern + pattern_to_child) / 2.0, child_to_pattern, pattern_to_child


def strong_geometry_features_for_sample(sample):
    child = foreground_mask(sample["child_path"])
    pattern = foreground_mask(sample["pattern_path"])
    child_shape, child_hu = contour_stats(child)
    pattern_shape, pattern_hu = contour_stats(pattern)
    child_bw, child_bh, child_ba, child_aspect = bbox_stats(child)
    pattern_bw, pattern_bh, pattern_ba, pattern_aspect = bbox_stats(pattern)
    symmetric_chamfer, child_to_pattern, pattern_to_child = chamfer_distances(child, pattern)

    comparison = np.asarray(
        [
            abs(child_shape[2] - pattern_shape[2]),
            abs(child_shape[0] - pattern_shape[0]),
            abs(child_bw - pattern_bw),
            abs(child_bh - pattern_bh),
            abs(child_ba - pattern_ba),
            abs(child_aspect - pattern_aspect),
            symmetric_chamfer,
            child_to_pattern,
            pattern_to_child,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(
        np.concatenate([child_shape, pattern_shape, comparison, child_hu, pattern_hu, np.abs(child_hu - pattern_hu)]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def matrix(samples, feature_fn):
    return np.vstack([feature_fn(sample) for sample in samples]).astype(np.float32)


class PairImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        child = Image.open(sample["child_path"]).convert("RGB")
        pattern = Image.open(sample["pattern_path"]).convert("RGB")
        return self.transform(child), self.transform(pattern)


def eval_transform(image_size):
    return transforms.Compose(
        [
            ResizeLongSideAndPad(size=image_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def extract_resnet18_pair_features(samples, pretrained, image_size, batch_size, device):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)

    loader = DataLoader(PairImageDataset(samples, eval_transform(image_size)), batch_size=batch_size, shuffle=False)
    features = []
    for child, pattern in loader:
        child_features = model(child.to(device)).cpu().numpy()
        pattern_features = model(pattern.to(device)).cpu().numpy()
        diff = np.abs(child_features - pattern_features)
        mul = child_features * pattern_features
        features.append(np.concatenate([diff, mul], axis=1))
    return np.vstack(features).astype(np.float32)


def make_classifier(kind, seed):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed),
        )
    if kind == "linearsvm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.05, class_weight="balanced", max_iter=10000, random_state=seed),
        )
    raise ValueError(kind)


def predict_one_vs_rest(x_train, y_train, x_test, kind, seed):
    preds = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=int)
    thresholds = np.full(y_train.shape[1], 0.5 if kind == "logreg" else np.nan, dtype=np.float32)
    for label_idx in range(y_train.shape[1]):
        y = y_train[:, label_idx]
        if len(np.unique(y)) < 2:
            preds[:, label_idx] = int(y[0])
            continue
        clf = make_classifier(kind, seed + label_idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(x_train, y)
        if kind == "logreg":
            preds[:, label_idx] = (clf.predict_proba(x_test)[:, 1] >= 0.5).astype(int)
        else:
            preds[:, label_idx] = clf.predict(x_test).astype(int)
    return preds, thresholds


def chain_orders(y_train, seed):
    label_count = y_train.shape[1]
    original = np.arange(label_count)
    prevalence = y_train.mean(axis=0)
    rng = np.random.default_rng(seed)
    random_order = original.copy()
    rng.shuffle(random_order)
    orders = []
    for order in [original, original[::-1], np.argsort(-prevalence), np.argsort(prevalence), random_order]:
        order_tuple = tuple(int(idx) for idx in order)
        if order_tuple not in orders:
            orders.append(order_tuple)
    return [np.asarray(order, dtype=np.int64) for order in orders]


def predict_chain(x_train, y_train, x_test, kind, order, seed):
    """Classifier chain z klasycznym teacher forcing.

    W treningu kolejne modele dostają prawdziwe poprzednie etykiety, a podczas
    testowania dostają predykcje wcześniejszych modeli w łańcuchu. To standardowy
    wariant classifier chain, ale przy n=14 może zawyżać zależność od idealnych
    poprzednich etykiet i pogarszać odporność na propagację błędów.
    """

    preds = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=int)
    scores = np.zeros_like(preds, dtype=np.float32)
    train_chain = x_train
    test_chain = x_test
    for pos, label_idx in enumerate(order):
        y = y_train[:, label_idx]
        if len(np.unique(y)) < 2:
            pred = np.full(x_test.shape[0], int(y[0]), dtype=int)
            score = pred.astype(np.float32)
        else:
            clf = make_classifier(kind, seed + int(label_idx) + 100 * pos)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(train_chain, y)
            if kind == "logreg":
                score = clf.predict_proba(test_chain)[:, 1].astype(np.float32)
                pred = (score >= 0.5).astype(int)
            else:
                pred = clf.predict(test_chain).astype(int)
                score = pred.astype(np.float32)
        preds[:, label_idx] = pred
        scores[:, label_idx] = score
        train_chain = np.concatenate([train_chain, y.reshape(-1, 1)], axis=1)
        test_chain = np.concatenate([test_chain, pred.reshape(-1, 1)], axis=1)
    return preds, scores


def predict_chain_oof(x_train, y_train, x_test, groups, order, seed):
    """Classifier chain z OOF-predykcjami poprzednich etykiet w treningu.

    Dla każdej etykiety generujemy predykcje out-of-fold na zbiorze treningowym
    przez `GroupKFold` po pacjentach. Te predykcje, a nie prawdziwe etykiety, są
    doklejane jako kolejne cechy treningowe dla następnych ogniw łańcucha. Dzięki
    temu trening i test widzą podobny typ sygnału: poprzednie etykiety są
    predykcjami modelu, nie idealną informacją z anotacji.

    Wariant jest celowo ograniczony do LogisticRegression, bo zachowujemy próg
    `predict_proba >= 0.5` i porównywalność z pozostałymi eksperymentami.
    """

    preds = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=int)
    scores = np.zeros_like(preds, dtype=np.float32)
    train_chain = x_train
    test_chain = x_test
    groups = np.asarray(groups)

    for pos, label_idx in enumerate(order):
        y = y_train[:, label_idx].astype(int)
        unique_y = np.unique(y)

        if len(unique_y) < 2:
            train_oof_pred = np.full(y_train.shape[0], int(unique_y[0]), dtype=int)
            pred = np.full(x_test.shape[0], int(unique_y[0]), dtype=int)
            score = pred.astype(np.float32)
        else:
            train_oof_pred = np.zeros(y_train.shape[0], dtype=int)
            unique_groups = np.unique(groups)

            if len(unique_groups) < 2:
                # OOF wymaga co najmniej dwóch pacjentów treningowych. Ten
                # fallback pozwala uruchamiać bardzo krótkie smoke-testy.
                train_oof_pred = y.copy()
            else:
                n_splits = min(5, len(unique_groups))
                inner_cv = GroupKFold(n_splits=n_splits)
                for inner_idx, (inner_train_idx, inner_valid_idx) in enumerate(
                    inner_cv.split(train_chain, y, groups)
                ):
                    inner_y = y[inner_train_idx]
                    if len(np.unique(inner_y)) < 2:
                        train_oof_pred[inner_valid_idx] = int(inner_y[0])
                        continue

                    inner_clf = make_classifier("logreg", seed + int(label_idx) + 100 * pos + 1000 * inner_idx)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        inner_clf.fit(train_chain[inner_train_idx], inner_y)
                    inner_score = inner_clf.predict_proba(train_chain[inner_valid_idx])[:, 1]
                    train_oof_pred[inner_valid_idx] = (inner_score >= 0.5).astype(int)

            clf = make_classifier("logreg", seed + int(label_idx) + 100 * pos)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(train_chain, y)
            score = clf.predict_proba(test_chain)[:, 1].astype(np.float32)
            pred = (score >= 0.5).astype(int)

        preds[:, label_idx] = pred
        scores[:, label_idx] = score
        train_chain = np.concatenate([train_chain, train_oof_pred.reshape(-1, 1)], axis=1)
        test_chain = np.concatenate([test_chain, pred.reshape(-1, 1)], axis=1)

    return preds, scores


def predict_chain_ensemble(x_train, y_train, x_test, seed):
    scores = []
    for idx, order in enumerate(chain_orders(y_train, seed)):
        _, order_scores = predict_chain(x_train, y_train, x_test, "logreg", order, seed + 1000 * idx)
        scores.append(order_scores)
    mean_scores = np.mean(np.stack(scores, axis=0), axis=0)
    return (mean_scores >= 0.5).astype(int), np.full(y_train.shape[1], 0.5, dtype=np.float32)


def compute_metrics(labels, preds, thresholds, strategy):
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    per_label = {}
    for idx, category in enumerate(ERROR_CATEGORIES):
        per_label[category] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "actual_positives": int(labels[:, idx].sum()),
            "predicted_positives": int(preds[:, idx].sum()),
            "true_positive_matches": int(((labels[:, idx] == 1) & (preds[:, idx] == 1)).sum()),
            "threshold": float(thresholds[idx]),
        }
    return {
        "strategy": strategy,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
        "subset_accuracy": float(accuracy_score(labels, preds)),
        "per_label": per_label,
    }


def summarize(folds, config):
    summary = {"config": config, "strategies": {}}
    for strategy in sorted(folds[0]["test_metrics"].keys()):
        metrics = [fold["test_metrics"][strategy] for fold in folds]
        out = {}
        for key in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "subset_accuracy"]:
            values = np.asarray([metric[key] for metric in metrics], dtype=np.float32)
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std())
        out["per_label"] = {}
        for label in ERROR_CATEGORIES:
            values = np.asarray([metric["per_label"][label]["f1"] for metric in metrics], dtype=np.float32)
            out["per_label"][label] = {"f1_mean": float(values.mean()), "f1_std": float(values.std())}
        summary["strategies"][strategy] = out
    return summary


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    root_dir="../data/processed/siemens-net-data",
    results_dir="../results/siamese-resnet18-classical-chain",
    image_size=224,
    batch_size=16,
    seed=42,
    pretrained=True,
    device=None,
    max_folds=None,
):
    set_seed(seed)
    root_dir = resolve_input_path(root_dir)
    results_dir = resolve_output_path(results_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    patients = sorted([path.name for path in root_dir.iterdir() if path.is_dir()])
    if max_folds is not None:
        patients = patients[:max_folds]

    config = {
        "experiment": "siamese_resnet18_pair_features_geometry_stroke_classifier_chain",
        "root_dir": str(root_dir),
        "patients": patients,
        "labels": ERROR_CATEGORIES,
        "split_policy": "LOSO; train on all non-test patients because no validation tuning is used",
        "threshold_policy": (
            "LogisticRegression predict_proba >= 0.5; LinearSVM hard decisions; "
            "classifier_chain_logreg is a legacy teacher-forcing alias; "
            "classifier_chain_logreg_oof uses patient-grouped OOF previous-label predictions during training"
        ),
        "features": {
            "resnet18_pair_features": "absdiff + multiply of frozen child/pattern embeddings",
            "basic_geometry_dim": GEOMETRY_FEATURE_DIM,
            "strong_geometry": STRONG_GEOMETRY_FEATURE_NAMES,
            "stroke": STROKE_FEATURE_NAMES,
            "pretrained": pretrained,
        },
    }

    fold_results = []
    for fold_idx, test_patient in enumerate(patients, start=1):
        train_patients = [patient for patient in patients if patient != test_patient]
        print(f"\nFold {fold_idx}/{len(patients)} | test={test_patient}")
        train_ds = SiameseBVRTDataset(root_dir, patient_ids=train_patients)
        test_ds = SiameseBVRTDataset(root_dir, patient_ids=[test_patient])
        y_train = labels_for_samples(train_ds).astype(int)
        y_test = labels_for_samples(test_ds).astype(int)

        train_basic = matrix(train_ds.samples, geometric_features_for_sample)
        test_basic = matrix(test_ds.samples, geometric_features_for_sample)
        train_strong = matrix(train_ds.samples, strong_geometry_features_for_sample)
        test_strong = matrix(test_ds.samples, strong_geometry_features_for_sample)
        train_stroke = matrix(train_ds.samples, stroke_features_for_sample)
        test_stroke = matrix(test_ds.samples, stroke_features_for_sample)
        train_structured = np.concatenate([train_basic, train_strong, train_stroke], axis=1)
        test_structured = np.concatenate([test_basic, test_strong, test_stroke], axis=1)
        train_resnet = extract_resnet18_pair_features(
            train_ds.samples, pretrained=pretrained, image_size=image_size, batch_size=batch_size, device=device
        )
        test_resnet = extract_resnet18_pair_features(
            test_ds.samples, pretrained=pretrained, image_size=image_size, batch_size=batch_size, device=device
        )

        feature_sets = {
            "stroke_only": (train_stroke, test_stroke),
            "geometry_stroke": (train_structured, test_structured),
            "resnet18_pair_features": (train_resnet, test_resnet),
            "resnet18_pair_features_plus_geometry_stroke": (
                np.concatenate([train_resnet, train_structured], axis=1),
                np.concatenate([test_resnet, test_structured], axis=1),
            ),
        }

        majority_vector = majority_baseline_vector(y_train)
        baseline_thresholds = np.full(len(ERROR_CATEGORIES), 0.5, dtype=np.float32)
        test_metrics = {
            "majority_baseline": compute_metrics(
                y_test, constant_predictions(y_test, majority_vector), majority_vector, "majority_baseline"
            ),
            "pattern_majority_baseline": compute_metrics(
                y_test, pattern_majority_predictions(train_ds, test_ds), baseline_thresholds, "pattern_majority_baseline"
            ),
            "always_positive_baseline": compute_metrics(
                y_test, np.ones_like(y_test), np.ones(len(ERROR_CATEGORIES), dtype=np.float32), "always_positive_baseline"
            ),
        }

        for feature_name, (x_train, x_test) in feature_sets.items():
            for kind in ["logreg", "linearsvm"]:
                preds, thresholds = predict_one_vs_rest(x_train, y_train, x_test, kind, seed + fold_idx)
                strategy = f"{feature_name}_{kind}"
                test_metrics[strategy] = compute_metrics(y_test, preds, thresholds, strategy)

        chain_name = "resnet18_pair_features_plus_geometry_stroke"
        x_train, x_test = feature_sets[chain_name]
        for kind in ["logreg", "linearsvm"]:
            preds, scores = predict_chain(x_train, y_train, x_test, kind, np.arange(y_train.shape[1]), seed + fold_idx)
            thresholds = np.full(y_train.shape[1], 0.5 if kind == "logreg" else np.nan, dtype=np.float32)
            strategy = f"{chain_name}_classifier_chain_{kind}"
            test_metrics[strategy] = compute_metrics(y_test, preds, thresholds, strategy)
        preds, scores = predict_chain(x_train, y_train, x_test, "logreg", np.arange(y_train.shape[1]), seed + fold_idx)
        thresholds = np.full(y_train.shape[1], 0.5, dtype=np.float32)
        strategy = f"{chain_name}_classifier_chain_logreg_teacher_forcing"
        test_metrics[strategy] = compute_metrics(y_test, preds, thresholds, strategy)
        train_groups = np.asarray([sample["patient"] for sample in train_ds.samples])
        preds, scores = predict_chain_oof(
            x_train,
            y_train,
            x_test,
            train_groups,
            np.arange(y_train.shape[1]),
            seed + fold_idx,
        )
        strategy = f"{chain_name}_classifier_chain_logreg_oof"
        test_metrics[strategy] = compute_metrics(y_test, preds, thresholds, strategy)
        preds, thresholds = predict_chain_ensemble(x_train, y_train, x_test, seed + fold_idx)
        strategy = f"{chain_name}_classifier_chain_logreg_ensemble"
        test_metrics[strategy] = compute_metrics(y_test, preds, thresholds, strategy)

        fold_results.append(
            {
                "fold": fold_idx,
                "test_patient": test_patient,
                "val_patient": None,
                "train_patients": train_patients,
                "best_val_loss": np.nan,
                "best_val_macro_f1": np.nan,
                "epochs_trained": 0,
                "feature_dimensions": {name: int(values[0].shape[1]) for name, values in feature_sets.items()},
                "test_metrics": test_metrics,
            }
        )
        best_strategy = max(test_metrics.items(), key=lambda item: item[1]["macro_f1"])
        print(f"Best fold strategy={best_strategy[0]} | macro F1={best_strategy[1]['macro_f1']:.4f}")

    summary = summarize(fold_results, config)
    write_json(results_dir / "experiment_config.json", config)
    write_json(results_dir / "fold_metrics.json", fold_results)
    write_json(results_dir / "summary_metrics.json", summary)

    fold_rows = []
    per_label_rows = []
    for fold in fold_results:
        for strategy, metrics in fold["test_metrics"].items():
            fold_rows.append(
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
            for label, label_metrics in metrics["per_label"].items():
                per_label_rows.append(
                    {
                        "fold": fold["fold"],
                        "test_patient": fold["test_patient"],
                        "val_patient": fold["val_patient"],
                        "strategy": strategy,
                        "label": label,
                        **label_metrics,
                    }
                )
    write_csv(results_dir / "fold_metrics.csv", fold_rows)
    write_csv(results_dir / "per_label_metrics.csv", per_label_rows)

    print("\nSiamese ResNet18 classical-chain summary:")
    for strategy_name, metrics in sorted(summary["strategies"].items()):
        print(
            f"{strategy_name}: macro F1={metrics['macro_f1_mean']:.4f} +/- {metrics['macro_f1_std']:.4f}, "
            f"micro F1={metrics['micro_f1_mean']:.4f} +/- {metrics['micro_f1_std']:.4f}"
        )
    print(f"Results saved in: {results_dir.resolve()}")
    return {"fold_results": fold_results, "summary": summary}


def parse_args():
    parser = argparse.ArgumentParser(description="Siamese ResNet18 pair-feature classical chain experiments.")
    parser.add_argument("--root-dir", default="../data/processed/siemens-net-data")
    parser.add_argument("--results-dir", default="../results/siamese-resnet18-classical-chain")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        root_dir=args.root_dir,
        results_dir=args.results_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        device=args.device,
        max_folds=args.max_folds,
    )
