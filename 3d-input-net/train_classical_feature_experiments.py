import argparse
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from train_geometry import (
    BVRTDataset,
    ERROR_CATEGORIES,
    GEOMETRY_FEATURE_NAMES,
    ResizeLongSideAndPad,
    compute_multilabel_metrics,
    constant_predictions,
    geometric_features_for_image,
    labels_for_samples,
    majority_baseline_vector,
    pattern_majority_predictions,
    resolve_output_path,
    resolve_path,
    save_results,
    set_seed,
    summarize_results,
)


SCRIPT_DIR = Path(__file__).resolve().parent

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

SHAPE_FEATURE_NAMES = [
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
]

HU_FEATURE_NAMES = (
    [f"child_hu_{idx}" for idx in range(1, 8)]
    + [f"pattern_hu_{idx}" for idx in range(1, 8)]
    + [f"hu_absdiff_{idx}" for idx in range(1, 8)]
)

STRONG_GEOMETRY_FEATURE_NAMES = SHAPE_FEATURE_NAMES + HU_FEATURE_NAMES
STRUCTURED_FEATURE_NAMES = GEOMETRY_FEATURE_NAMES + STRONG_GEOMETRY_FEATURE_NAMES + STROKE_FEATURE_NAMES


def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def stats(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return (
        float(np.std(values)),
        float(p25),
        float(p50),
        float(p75),
        float(p75 - p25),
    )


@lru_cache(maxsize=None)
def load_summary_by_drawing(patient_dir):
    summary_path = Path(patient_dir) / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {int(drawing["index"]): drawing for drawing in summary.get("drawings", [])}


def drawing_for_sample(sample):
    patient_dir = Path(sample["img_path"]).parent
    return load_summary_by_drawing(patient_dir).get(int(sample["drawing_idx"]), {})


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
            deltas = np.diff(xy, axis=0)
            stroke_lengths.append(float(np.linalg.norm(deltas, axis=1).sum()))
        else:
            stroke_lengths.append(0.0)

    if not points:
        return np.zeros((0, 2), dtype=np.float32), [], []
    return np.vstack(points).astype(np.float32), stroke_lengths, points_per_stroke


def stroke_features_for_drawing(drawing):
    """Cechy behawioralne i wektorowe z danych rysowania.

    Te cechy nie opisują tylko końcowego obrazu. Dodają informację o procesie:
    czas planowania, płynność ruchu, liczbę poprawek, długość trajektorii i
    zagęszczenie punktów. To może być wartościowe klinicznie dla BVRT, bo błędy
    wykonawcze i organizacja przestrzenna nie zawsze są widoczne w samym PNG.
    """

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

    velocities = [safe_float(value) for value in drawing.get("velocities", [])]
    velocity_std, velocity_p25, velocity_p50, velocity_p75, velocity_iqr = stats(velocities)

    points, stroke_lengths, points_per_stroke = flatten_stroke_points(drawing.get("strokes_data", []))
    stroke_count = len(stroke_lengths)
    point_count = len(points)
    total_path_length = float(np.sum(stroke_lengths)) if stroke_lengths else 0.0

    if point_count > 0:
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

    path_efficiency = straight_distance / total_path_length if total_path_length > 0 else 0.0
    mean_stroke_length = float(np.mean(stroke_lengths)) if stroke_lengths else 0.0
    std_stroke_length = float(np.std(stroke_lengths)) if stroke_lengths else 0.0
    mean_points_per_stroke = float(np.mean(points_per_stroke)) if points_per_stroke else 0.0

    features = np.array(
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
            float(stroke_count),
            float(point_count) / 1000.0,
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
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def masks_from_processed_image(img_path):
    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {img_path}")
    child_mask = (image_bgr[:, :, 0] > 20).astype(np.uint8)
    pattern_mask = (image_bgr[:, :, 1] > 20).astype(np.uint8)
    return child_mask, pattern_mask


def contour_stats(mask):
    height, width = mask.shape
    canvas_area = float(max(height * width, 1))
    diag = float(max(np.hypot(width, height), 1.0))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)

    contour_areas = np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float32)
    contour_perimeters = np.array([cv2.arcLength(contour, closed=True) for contour in contours], dtype=np.float32)
    largest_idx = int(np.argmax(contour_areas))
    largest = contours[largest_idx]
    largest_area = float(contour_areas[largest_idx])
    perimeter = float(contour_perimeters.sum())

    x, y, bbox_width, bbox_height = cv2.boundingRect(largest)
    bbox_area = float(max(bbox_width * bbox_height, 1))
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = largest_area / hull_area if hull_area > 0 else 0.0
    extent = largest_area / bbox_area
    circularity = (4.0 * np.pi * largest_area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    shape = np.array(
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
    return np.nan_to_num(shape, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(
        hu.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def bbox_stats(mask):
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    bbox_width = float(xs.max() - xs.min() + 1) / max(float(width), 1.0)
    bbox_height = float(ys.max() - ys.min() + 1) / max(float(height), 1.0)
    bbox_area = bbox_width * bbox_height
    aspect = bbox_width / max(bbox_height, 1e-6)
    return bbox_width, bbox_height, bbox_area, aspect


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


def strong_geometry_features_for_image(img_path):
    """Rozszerzona geometria końcowego obrazu.

    Oprócz prostych pól i centroidów dodaje kontury, obwody, momenty Hu oraz
    odległość Chamfera między rysunkiem dziecka a wzorcem. To mocniejszy
    klasyczny baseline niż sama powierzchnia/bounding box.
    """

    child_mask, pattern_mask = masks_from_processed_image(img_path)
    child_shape, child_hu = contour_stats(child_mask)
    pattern_shape, pattern_hu = contour_stats(pattern_mask)

    child_bbox_width, child_bbox_height, child_bbox_area, child_aspect = bbox_stats(child_mask)
    pattern_bbox_width, pattern_bbox_height, pattern_bbox_area, pattern_aspect = bbox_stats(pattern_mask)
    symmetric_chamfer, child_to_pattern, pattern_to_child = chamfer_distances(child_mask, pattern_mask)

    comparison = np.array(
        [
            abs(child_shape[2] - pattern_shape[2]),
            abs(child_shape[0] - pattern_shape[0]),
            abs(child_bbox_width - pattern_bbox_width),
            abs(child_bbox_height - pattern_bbox_height),
            abs(child_bbox_area - pattern_bbox_area),
            abs(child_aspect - pattern_aspect),
            symmetric_chamfer,
            child_to_pattern,
            pattern_to_child,
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            child_shape,
            pattern_shape,
            comparison,
            child_hu,
            pattern_hu,
            np.abs(child_hu - pattern_hu),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def structured_features_for_sample(sample):
    drawing = drawing_for_sample(sample)
    return np.concatenate(
        [
            geometric_features_for_image(sample["img_path"]),
            strong_geometry_features_for_image(sample["img_path"]),
            stroke_features_for_drawing(drawing),
        ]
    ).astype(np.float32)


def feature_matrix(samples, feature_fn):
    return np.stack([feature_fn(sample) for sample in samples]).astype(np.float32)


class ImageOnlyDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]["img_path"]).convert("RGB")
        return self.transform(image)


def build_eval_transform(image_size):
    return transforms.Compose(
        [
            ResizeLongSideAndPad(size=image_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_feature_extractor(backbone_name, pretrained):
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        feature_dim = 512
    elif backbone_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        feature_dim = 1280
    else:
        raise ValueError(f"Nieznany backbone: {backbone_name}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, feature_dim


@torch.no_grad()
def extract_cnn_features(samples, backbone_name, pretrained, image_size, batch_size, device):
    transform = build_eval_transform(image_size)
    dataset = ImageOnlyDataset(samples, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model, feature_dim = build_feature_extractor(backbone_name, pretrained)
    model = model.to(device)

    features = []
    for images in loader:
        embeddings = model(images.to(device))
        features.append(embeddings.cpu().numpy())

    if not features:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.vstack(features).astype(np.float32)


def make_classifier(kind, seed):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if kind == "linearsvm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.05,
                class_weight="balanced",
                max_iter=10000,
                random_state=seed,
            ),
        )
    raise ValueError(f"Nieznany klasyfikator: {kind}")


def predict_multilabel_classical(x_train, y_train, x_test, classifier_kind, seed):
    """Trenuje 6 niezależnych klasyfikatorów one-vs-rest.

    LogisticRegression jest progowana jawnie przez `predict_proba >= 0.5`.
    LinearSVM zwraca twardą decyzję znaku marginesu, bo kalibracja probabilistyczna
    na kilkudziesięciu przykładach byłaby dodatkowym źródłem wariancji.
    """

    predictions = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=np.int64)
    thresholds = np.full(y_train.shape[1], 0.5 if classifier_kind == "logreg" else np.nan, dtype=np.float32)

    for label_idx in range(y_train.shape[1]):
        y_label = y_train[:, label_idx]
        if len(np.unique(y_label)) < 2:
            predictions[:, label_idx] = int(y_label[0])
            continue

        classifier = make_classifier(classifier_kind, seed + label_idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier.fit(x_train, y_label)

        if classifier_kind == "logreg":
            positive_probs = classifier.predict_proba(x_test)[:, 1]
            predictions[:, label_idx] = (positive_probs >= 0.5).astype(np.int64)
        else:
            predictions[:, label_idx] = classifier.predict(x_test).astype(np.int64)

    return predictions, thresholds


def evaluate_strategy(name, x_train, y_train, x_test, y_test, classifier_kind, seed):
    predictions, thresholds = predict_multilabel_classical(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        classifier_kind=classifier_kind,
        seed=seed,
    )
    return compute_multilabel_metrics(y_test, predictions, thresholds, name)


def chain_orders(y_train, seed):
    """Kilka deterministycznych kolejności dla classifier chain.

    Pojedynczy classifier chain jest zależny od kolejności etykiet. Dlatego dla
    wariantu ensemble używamy małego zestawu porządków: oryginalnego, odwrotnego,
    według częstości klas i jednego losowego ustalonego seedem.
    """

    label_count = y_train.shape[1]
    original = np.arange(label_count)
    prevalence = y_train.mean(axis=0)
    frequent_first = np.argsort(-prevalence)
    rare_first = np.argsort(prevalence)
    rng = np.random.default_rng(seed)
    random_order = original.copy()
    rng.shuffle(random_order)

    orders = []
    for order in [original, original[::-1], frequent_first, rare_first, random_order]:
        order_tuple = tuple(int(idx) for idx in order)
        if order_tuple not in orders:
            orders.append(order_tuple)
    return [np.array(order, dtype=np.int64) for order in orders]


def predict_classifier_chain(x_train, y_train, x_test, classifier_kind, order, seed):
    """Classifier chain z klasycznym teacher forcing.

    Dla każdej kolejnej etykiety model dostaje oryginalne cechy oraz poprzednie
    etykiety w łańcuchu. W treningu są to prawdziwe etykiety, w predykcji są to
    etykiety przewidziane wcześniej przez łańcuch. To standardowy wariant
    classifier chain, ale przy bardzo małym zbiorze może tworzyć rozbieżność
    dystrybucji: trening widzi idealne poprzednie etykiety, test widzi błędne
    predykcje poprzednich modeli.
    """

    label_count = y_train.shape[1]
    predictions = np.zeros((x_test.shape[0], label_count), dtype=np.int64)
    scores = np.zeros((x_test.shape[0], label_count), dtype=np.float32)
    x_train_chain = x_train
    x_test_chain = x_test

    for chain_pos, label_idx in enumerate(order):
        y_label = y_train[:, label_idx]
        if len(np.unique(y_label)) < 2:
            test_pred = np.full(x_test.shape[0], int(y_label[0]), dtype=np.int64)
            test_score = test_pred.astype(np.float32)
        else:
            classifier = make_classifier(classifier_kind, seed + int(label_idx) + 100 * chain_pos)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                classifier.fit(x_train_chain, y_label)

            if classifier_kind == "logreg":
                test_score = classifier.predict_proba(x_test_chain)[:, 1].astype(np.float32)
                test_pred = (test_score >= 0.5).astype(np.int64)
            else:
                test_pred = classifier.predict(x_test_chain).astype(np.int64)
                test_score = test_pred.astype(np.float32)

        predictions[:, label_idx] = test_pred
        scores[:, label_idx] = test_score
        x_train_chain = np.concatenate([x_train_chain, y_label.reshape(-1, 1)], axis=1)
        x_test_chain = np.concatenate([x_test_chain, test_pred.reshape(-1, 1)], axis=1)

    return predictions, scores


def predict_classifier_chain_oof(x_train, y_train, x_test, groups, order, seed):
    """Classifier chain z predykcjami out-of-fold jako cechami treningowymi.

    Ten wariant usuwa asymetrię teacher forcing dla poprzednich etykiet.
    Zamiast doklejać do `x_train_chain` prawdziwą poprzednią etykietę, doklejamy
    jej predykcję out-of-fold. Foldy wewnętrzne są grupowane po pacjencie, więc
    predykcja dla danego pacjenta pochodzi z modelu trenowanego bez tego
    pacjenta. Dzięki temu kolejne ogniwa łańcucha uczą się na cechach o podobnym
    poziomie szumu jak podczas testowania.

    Używamy tylko LogisticRegression, bo potrzebujemy stabilnego progu 0.5 na
    `predict_proba`. Przy etykiecie jednowartościowej model zastępujemy stałą
    predykcją, tak jak w pozostałych baseline'ach.
    """

    label_count = y_train.shape[1]
    predictions = np.zeros((x_test.shape[0], label_count), dtype=np.int64)
    scores = np.zeros((x_test.shape[0], label_count), dtype=np.float32)
    x_train_chain = x_train
    x_test_chain = x_test
    groups = np.asarray(groups)

    for chain_pos, label_idx in enumerate(order):
        y_label = y_train[:, label_idx].astype(np.int64)
        unique_labels = np.unique(y_label)

        if len(unique_labels) < 2:
            train_oof_pred = np.full(y_train.shape[0], int(unique_labels[0]), dtype=np.int64)
            test_pred = np.full(x_test.shape[0], int(unique_labels[0]), dtype=np.int64)
            test_score = test_pred.astype(np.float32)
        else:
            train_oof_pred = np.zeros(y_train.shape[0], dtype=np.int64)
            unique_groups = np.unique(groups)

            if len(unique_groups) < 2:
                # OOF wymaga co najmniej dwóch pacjentów treningowych. Ten
                # fallback dotyczy tylko ekstremalnych smoke-testów.
                train_oof_pred = y_label.copy()
            else:
                n_splits = min(5, len(unique_groups))
                inner_cv = GroupKFold(n_splits=n_splits)
                for inner_idx, (inner_train_idx, inner_valid_idx) in enumerate(
                    inner_cv.split(x_train_chain, y_label, groups)
                ):
                    inner_y = y_label[inner_train_idx]
                    if len(np.unique(inner_y)) < 2:
                        train_oof_pred[inner_valid_idx] = int(inner_y[0])
                        continue

                    inner_classifier = make_classifier(
                        "logreg",
                        seed + int(label_idx) + 100 * chain_pos + 1000 * inner_idx,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        inner_classifier.fit(x_train_chain[inner_train_idx], inner_y)
                    inner_score = inner_classifier.predict_proba(x_train_chain[inner_valid_idx])[:, 1]
                    train_oof_pred[inner_valid_idx] = (inner_score >= 0.5).astype(np.int64)

            classifier = make_classifier("logreg", seed + int(label_idx) + 100 * chain_pos)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                classifier.fit(x_train_chain, y_label)
            test_score = classifier.predict_proba(x_test_chain)[:, 1].astype(np.float32)
            test_pred = (test_score >= 0.5).astype(np.int64)

        predictions[:, label_idx] = test_pred
        scores[:, label_idx] = test_score
        x_train_chain = np.concatenate([x_train_chain, train_oof_pred.reshape(-1, 1)], axis=1)
        x_test_chain = np.concatenate([x_test_chain, test_pred.reshape(-1, 1)], axis=1)

    return predictions, scores


def predict_classifier_chain_ensemble(x_train, y_train, x_test, seed):
    """Mały ensemble classifier chains dla LogisticRegression.

    Uśredniamy prawdopodobieństwa z kilku kolejności i stosujemy próg 0.5.
    To ogranicza arbitralność kolejności etykiet bez dodawania walidacji.
    """

    all_scores = []
    for order_idx, order in enumerate(chain_orders(y_train, seed)):
        _, scores = predict_classifier_chain(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            classifier_kind="logreg",
            order=order,
            seed=seed + 1000 * order_idx,
        )
        all_scores.append(scores)

    mean_scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    predictions = (mean_scores >= 0.5).astype(np.int64)
    thresholds = np.full(y_train.shape[1], 0.5, dtype=np.float32)
    return predictions, thresholds


def evaluate_classifier_chain_strategy(name, x_train, y_train, x_test, y_test, classifier_kind, seed):
    order = np.arange(y_train.shape[1], dtype=np.int64)
    predictions, _ = predict_classifier_chain(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        classifier_kind=classifier_kind,
        order=order,
        seed=seed,
    )
    thresholds = np.full(
        y_train.shape[1],
        0.5 if classifier_kind == "logreg" else np.nan,
        dtype=np.float32,
    )
    return compute_multilabel_metrics(y_test, predictions, thresholds, name)


def evaluate_classifier_chain_oof_strategy(name, x_train, y_train, x_test, y_test, groups, seed):
    order = np.arange(y_train.shape[1], dtype=np.int64)
    predictions, _ = predict_classifier_chain_oof(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        groups=groups,
        order=order,
        seed=seed,
    )
    thresholds = np.full(y_train.shape[1], 0.5, dtype=np.float32)
    return compute_multilabel_metrics(y_test, predictions, thresholds, name)


def evaluate_classifier_chain_ensemble_strategy(name, x_train, y_train, x_test, y_test, seed):
    predictions, thresholds = predict_classifier_chain_ensemble(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        seed=seed,
    )
    return compute_multilabel_metrics(y_test, predictions, thresholds, name)


def build_feature_sets(train_samples, test_samples, args, device):
    train_basic_geometry = feature_matrix(train_samples, lambda sample: geometric_features_for_image(sample["img_path"]))
    test_basic_geometry = feature_matrix(test_samples, lambda sample: geometric_features_for_image(sample["img_path"]))

    train_strong_geometry = feature_matrix(train_samples, lambda sample: strong_geometry_features_for_image(sample["img_path"]))
    test_strong_geometry = feature_matrix(test_samples, lambda sample: strong_geometry_features_for_image(sample["img_path"]))

    train_stroke = feature_matrix(train_samples, lambda sample: stroke_features_for_drawing(drawing_for_sample(sample)))
    test_stroke = feature_matrix(test_samples, lambda sample: stroke_features_for_drawing(drawing_for_sample(sample)))

    train_structured = np.concatenate([train_basic_geometry, train_strong_geometry, train_stroke], axis=1)
    test_structured = np.concatenate([test_basic_geometry, test_strong_geometry, test_stroke], axis=1)

    feature_sets = {
        "strong_geometry": (train_strong_geometry, test_strong_geometry),
        "stroke_only": (train_stroke, test_stroke),
        "geometry_stroke": (train_structured, test_structured),
    }

    if args.include_cnn_features:
        for backbone_name in args.backbones:
            train_cnn = extract_cnn_features(
                train_samples,
                backbone_name=backbone_name,
                pretrained=not args.no_pretrained,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
            )
            test_cnn = extract_cnn_features(
                test_samples,
                backbone_name=backbone_name,
                pretrained=not args.no_pretrained,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
            )
            feature_sets[f"{backbone_name}_features"] = (train_cnn, test_cnn)
            feature_sets[f"{backbone_name}_features_plus_geometry_stroke"] = (
                np.concatenate([train_cnn, train_structured], axis=1),
                np.concatenate([test_cnn, test_structured], axis=1),
            )

    return feature_sets


def run_loso_classical_feature_experiments(
    root_dir="../data/processed/3d-input-data",
    results_dir="../results/3d-input-classical-chain-experiments",
    image_size=224,
    batch_size=16,
    seed=42,
    device=None,
    no_pretrained=False,
    include_cnn_features=True,
    backbones=None,
    max_folds=None,
):
    set_seed(seed)
    root_dir = resolve_path(root_dir)
    results_dir = resolve_output_path(results_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbones = backbones or ["resnet18", "efficientnet_b0"]

    patients = sorted([path.name for path in root_dir.iterdir() if path.is_dir()])
    if max_folds is not None:
        patients = patients[:max_folds]

    config = {
        "experiment": "classical_geometry_stroke_frozen_cnn_features",
        "root_dir": str(root_dir),
        "image_size": image_size,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "patients": patients,
        "labels": ERROR_CATEGORIES,
        "threshold_policy": {
            "logistic_regression": "predict_proba >= 0.5",
            "linear_svm": "hard margin decision, no probability calibration",
            "classifier_chain_logreg": "predict_proba >= 0.5; legacy alias for teacher forcing",
            "classifier_chain_logreg_teacher_forcing": "train chain uses true previous labels; test chain uses predicted previous labels",
            "classifier_chain_logreg_oof": "train chain uses patient-grouped out-of-fold predictions of previous labels",
            "classifier_chain_logreg_ensemble": "mean probability across deterministic chain orders >= 0.5",
        },
        "split_policy": "LOSO; classical models train on all non-test patients because no validation tuning is used",
        "chain_feature_set": "resnet18_features_plus_geometry_stroke",
        "feature_groups": {
            "basic_geometry": GEOMETRY_FEATURE_NAMES,
            "strong_geometry": STRONG_GEOMETRY_FEATURE_NAMES,
            "stroke": STROKE_FEATURE_NAMES,
            "cnn_backbones": backbones if include_cnn_features else [],
            "pretrained": not no_pretrained,
        },
        "classifiers": {
            "logreg": "StandardScaler + LogisticRegression(C=0.5, class_weight=balanced, liblinear)",
            "linearsvm": "StandardScaler + LinearSVC(C=0.05, class_weight=balanced)",
            "classifier_chain_logreg": "Classifier chain with LogisticRegression base models; legacy alias for teacher forcing",
            "classifier_chain_logreg_teacher_forcing": "Classifier chain with true previous labels appended during training",
            "classifier_chain_logreg_oof": "Classifier chain with patient-grouped OOF previous-label predictions appended during training",
            "classifier_chain_linearsvm": "Classifier chain with LinearSVC base models",
        },
    }

    fold_results = []
    for fold_idx, test_patient in enumerate(patients, start=1):
        val_patient = None
        train_patients = [patient for patient in patients if patient != test_patient]

        print(f"\nFold {fold_idx}/{len(patients)} | test={test_patient} | val=None")

        train_ds = BVRTDataset(root_dir, patient_ids=train_patients, transform=None, use_geometry_features=False)
        test_ds = BVRTDataset(root_dir, patient_ids=[test_patient], transform=None, use_geometry_features=False)
        y_train = labels_for_samples(train_ds.samples)
        y_test = labels_for_samples(test_ds.samples)

        feature_sets = build_feature_sets(train_ds.samples, test_ds.samples, argparse.Namespace(
            include_cnn_features=include_cnn_features,
            backbones=backbones,
            no_pretrained=no_pretrained,
            image_size=image_size,
            batch_size=batch_size,
        ), device)

        majority_vector = majority_baseline_vector(y_train)
        test_metrics = {
            "majority_baseline": compute_multilabel_metrics(
                y_test,
                constant_predictions(y_test, majority_vector),
                majority_vector.astype(np.float32),
                "majority_baseline",
            ),
            "pattern_majority_baseline": compute_multilabel_metrics(
                y_test,
                pattern_majority_predictions(train_ds.samples, test_ds.samples),
                np.full(len(ERROR_CATEGORIES), np.nan, dtype=np.float32),
                "pattern_majority_baseline",
            ),
            "always_positive_baseline": compute_multilabel_metrics(
                y_test,
                np.ones_like(y_test, dtype=np.int64),
                np.ones(len(ERROR_CATEGORIES), dtype=np.float32),
                "always_positive_baseline",
            ),
        }

        for feature_name, (x_train, x_test) in feature_sets.items():
            for classifier_kind in ["logreg", "linearsvm"]:
                strategy = f"{feature_name}_{classifier_kind}"
                test_metrics[strategy] = evaluate_strategy(
                    name=strategy,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    classifier_kind=classifier_kind,
                    seed=seed + fold_idx,
                )

        chain_feature_name = "resnet18_features_plus_geometry_stroke"
        if chain_feature_name in feature_sets:
            x_train, x_test = feature_sets[chain_feature_name]
            train_groups = np.asarray([sample["patient"] for sample in train_ds.samples])
            test_metrics[f"{chain_feature_name}_classifier_chain_logreg"] = evaluate_classifier_chain_strategy(
                name=f"{chain_feature_name}_classifier_chain_logreg",
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                classifier_kind="logreg",
                seed=seed + fold_idx,
            )
            test_metrics[f"{chain_feature_name}_classifier_chain_logreg_teacher_forcing"] = (
                evaluate_classifier_chain_strategy(
                    name=f"{chain_feature_name}_classifier_chain_logreg_teacher_forcing",
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    classifier_kind="logreg",
                    seed=seed + fold_idx,
                )
            )
            test_metrics[f"{chain_feature_name}_classifier_chain_logreg_oof"] = (
                evaluate_classifier_chain_oof_strategy(
                    name=f"{chain_feature_name}_classifier_chain_logreg_oof",
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    groups=train_groups,
                    seed=seed + fold_idx,
                )
            )
            test_metrics[f"{chain_feature_name}_classifier_chain_linearsvm"] = evaluate_classifier_chain_strategy(
                name=f"{chain_feature_name}_classifier_chain_linearsvm",
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                classifier_kind="linearsvm",
                seed=seed + fold_idx,
            )
            test_metrics[f"{chain_feature_name}_classifier_chain_logreg_ensemble"] = (
                evaluate_classifier_chain_ensemble_strategy(
                    name=f"{chain_feature_name}_classifier_chain_logreg_ensemble",
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    seed=seed + fold_idx,
                )
            )

        fold_result = {
            "fold": fold_idx,
            "test_patient": test_patient,
            "val_patient": val_patient,
            "train_patients": train_patients,
            "best_val_loss": np.nan,
            "best_val_macro_f1": np.nan,
            "epochs_trained": 0,
            "feature_dimensions": {name: int(values[0].shape[1]) for name, values in feature_sets.items()},
            "test_metrics": test_metrics,
        }
        fold_results.append(fold_result)

        best_strategy = max(test_metrics.items(), key=lambda item: item[1]["macro_f1"])
        print(f"Best fold strategy={best_strategy[0]} | macro F1={best_strategy[1]['macro_f1']:.4f}")

    summary = summarize_results(fold_results)
    save_results(results_dir, config, fold_results, summary)

    print("\nPodsumowanie klasycznych eksperymentów cech:")
    for strategy, metrics in sorted(summary["strategies"].items()):
        print(
            f"{strategy}: macro F1={metrics['macro_f1_mean']:.4f} +/- {metrics['macro_f1_std']:.4f}, "
            f"micro F1={metrics['micro_f1_mean']:.4f} +/- {metrics['micro_f1_std']:.4f}"
        )
    print(f"Results saved in: {results_dir.resolve()}")
    return {"config": config, "fold_results": fold_results, "summary": summary}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical BVRT experiments: geometry, stroke data, frozen CNN features."
    )
    parser.add_argument("--root-dir", default="../data/processed/3d-input-data")
    parser.add_argument("--results-dir", default="../results/3d-input-classical-chain-experiments")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-cnn-features", action="store_true")
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "efficientnet_b0"], choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loso_classical_feature_experiments(
        root_dir=args.root_dir,
        results_dir=args.results_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        no_pretrained=args.no_pretrained,
        include_cnn_features=not args.no_cnn_features,
        backbones=args.backbones,
        max_folds=args.max_folds,
    )
