"""
Utility functions for the AVE project.
Parsing annotations, creating labels, computing class weights, MLflow setup.
"""

import os
import numpy as np
import torch
import config


def parse_annotation_line(line):
    """
    Parse a single annotation line from the AVE dataset.

    Format: "Category&VideoID&Quality&StartTime&EndTime"
    Example: "Church bell&RUhOCu3LNXM&good&0&10"

    Returns:
        dict with keys: category, video_id, quality, start_time, end_time
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split("&")
    if len(parts) != 5:
        return None
    return {
        "category": parts[0],
        "video_id": parts[1],
        "quality": parts[2],
        "start_time": int(parts[3]),
        "end_time": int(parts[4]),
    }


def load_split_file(filepath):
    """
    Load a dataset split file (trainSet.txt, valSet.txt, testSet.txt).

    Returns:
        list of dicts, each with: category, video_id, quality, start_time, end_time
    """
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed is not None:
                samples.append(parsed)
    return samples


def create_second_labels(start_time, end_time, total_seconds=10):
    """Binary per-second labels (1 = event, 0 = background). Kept for compatibility."""
    labels = np.zeros(total_seconds, dtype=np.int64)
    for t in range(total_seconds):
        if start_time <= t < end_time:
            labels[t] = 1
    return labels


def create_multiclass_labels(category, start_time, end_time, total_seconds=10):
    """
    Create per-second 29-class labels for a video clip.

    Seconds in [start_time, end_time) get the event category index (0–27).
    All other seconds get config.BACKGROUND_IDX (28).

    Args:
        category: event category string (must be in config.ALL_CLASSES)
        start_time: event start second (inclusive)
        end_time: event end second (exclusive)
        total_seconds: total clip duration in seconds

    Returns:
        numpy int64 array of shape (total_seconds,) with class indices 0–28
    """
    cat_idx = config.ALL_CLASS_TO_IDX.get(category, config.BACKGROUND_IDX)
    labels = np.full(total_seconds, config.BACKGROUND_IDX, dtype=np.int64)
    for t in range(total_seconds):
        if start_time <= t < end_time:
            labels[t] = cat_idx
    return labels


def get_video_path(video_id):
    """
    Get the full path to a video file given its YouTube ID.

    Args:
        video_id: YouTube video ID string

    Returns:
        Full path to the .mp4 file
    """
    return os.path.join(config.VIDEO_DIR, f"{video_id}.mp4")


def compute_class_weights(dataset_samples):
    """
    Compute inverse-frequency weights for all 29 classes to handle class imbalance.

    Rare event categories (and the Background class) each get a weight
    inversely proportional to how often that label appears across all seconds
    in the given split. The weights are normalised so the mean weight ≈ 1.

    Args:
        dataset_samples: list of sample dicts from load_split_file

    Returns:
        torch.FloatTensor of shape (NUM_CLASSES,) = (29,)
    """
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for sample in dataset_samples:
        labels = create_multiclass_labels(
            sample["category"], sample["start_time"], sample["end_time"]
        )
        for lbl in labels:
            counts[lbl] += 1

    counts = np.maximum(counts, 1.0)          # avoid div-by-zero for unseen classes
    weights = 1.0 / counts
    weights /= weights.sum() / config.NUM_CLASSES   # normalise: mean weight ≈ 1
    return torch.tensor(weights, dtype=torch.float32)


def compute_category_distribution(dataset_samples):
    """
    Count clips per event category for data-exploration purposes.

    Returns:
        dict mapping category name → clip count
    """
    counts = {}
    for sample in dataset_samples:
        cat = sample["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def setup_mlflow():
    """
    Configure MLflow tracking for experiment logging.
    Uses local file-based tracking (no server needed).
    """
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    return mlflow


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.FEATURES_DIR, "audio"), exist_ok=True)
    os.makedirs(os.path.join(config.FEATURES_DIR, "video"), exist_ok=True)
