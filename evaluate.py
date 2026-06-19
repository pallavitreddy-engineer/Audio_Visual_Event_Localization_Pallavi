"""
Evaluation metrics for AVE per-second event localisation.

Metrics (Guide §6):
  • per_second_accuracy — overall % of seconds correctly classified
  • per_second_recall   — macro-averaged recall across event classes
                          (recall is prioritised over precision: §6.2)
  • temporal_iou        — per-clip intersection-over-union of predicted vs
                          ground-truth event segments (§6.3)
  • evaluate            — full evaluation pass over a DataLoader
  • baseline_scores     — sanity baselines: always-Background and random
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import config


# ──────────────────────────────────────────────
# Per-second metrics
# ──────────────────────────────────────────────

def per_second_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of seconds where the predicted class matches the ground truth.
    Both tensors must be 1-D LongTensors of the same length.
    """
    return (preds == labels).float().mean().item()


def per_second_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Macro-averaged recall across the 28 real event classes (Background excluded).

    Recall = TP / (TP + FN) per class, then averaged.
    Guide §6.2: missing a real event (FN) is worse than a false alarm, so
    recall is the primary single-number metric for this task.
    """
    recalls = []
    for c in range(config.NUM_CLASSES - 1):    # skip Background (index 28)
        gt_pos  = (labels == c)
        tp      = ((preds == c) & gt_pos).sum().item()
        fn      = ((preds != c) & gt_pos).sum().item()
        denom   = tp + fn
        recalls.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(recalls))


# ──────────────────────────────────────────────
# Temporal IoU
# ──────────────────────────────────────────────

def _labels_to_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a sequence of per-second labels to a list of (start, end) segments
    where end is exclusive and the label is not Background.
    """
    segments = []
    in_seg   = False
    start    = 0
    for t, lbl in enumerate(labels):
        is_event = int(lbl) != config.BACKGROUND_IDX
        if is_event and not in_seg:
            start  = t
            in_seg = True
        elif not is_event and in_seg:
            segments.append((start, t))
            in_seg = False
    if in_seg:
        segments.append((start, len(labels)))
    return segments


def temporal_iou_single(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Temporal IoU for a single clip (Guide §6.3).

    Computes the ratio of seconds where BOTH pred and gt indicate a (matching)
    event over the seconds where EITHER indicates an event.

    Returns:
        IoU in [0, 1]; 1.0 if both are all-Background.
    """
    pred_event = pred_labels != config.BACKGROUND_IDX
    gt_event   = gt_labels   != config.BACKGROUND_IDX

    intersection = int((pred_event & gt_event).sum())
    union        = int((pred_event | gt_event).sum())

    if union == 0:
        return 1.0          # both fully Background — perfect agreement
    return intersection / union


def temporal_iou_batch(
    all_preds: list[np.ndarray],
    all_labels: list[np.ndarray],
    threshold: float = 0.5,
) -> dict:
    """
    Compute mean temporal IoU and the fraction of clips with IoU ≥ threshold.

    Args:
        all_preds : list of (10,) prediction arrays per clip
        all_labels: list of (10,) ground-truth arrays per clip
        threshold : IoU threshold considered a correct detection (§6.3 → 0.5)

    Returns:
        dict with keys: mean_iou, correct_fraction
    """
    ious = [temporal_iou_single(p, g) for p, g in zip(all_preds, all_labels)]
    ious = np.array(ious)
    return {
        "mean_iou"        : float(ious.mean()),
        "correct_fraction": float((ious >= threshold).mean()),
        "threshold"       : threshold,
    }


# ──────────────────────────────────────────────
# Full evaluation pass
# ──────────────────────────────────────────────

def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    """
    Run inference over `loader` and compute all metrics.

    Returns dict with:
        accuracy, recall, mean_iou, correct_fraction,
        per_class_report (string)
    """
    import torch.nn as nn

    model.eval()
    all_preds_flat, all_labels_flat = [], []
    all_preds_clips, all_labels_clips = [], []

    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"].to(device)
            video  = batch["video"].to(device)
            labels = batch["labels"]              # (B, T) cpu

            logits = model(audio, video, modality_dropout_prob=0.0)
            preds  = logits.argmax(dim=-1).cpu()  # (B, T)

            for p, l in zip(preds.numpy(), labels.numpy()):
                all_preds_clips.append(p)
                all_labels_clips.append(l)
                all_preds_flat.extend(p.tolist())
                all_labels_flat.extend(l.tolist())

    preds_t  = torch.tensor(all_preds_flat,  dtype=torch.long)
    labels_t = torch.tensor(all_labels_flat, dtype=torch.long)

    acc    = per_second_accuracy(preds_t, labels_t)
    recall = per_second_recall(preds_t, labels_t)
    iou_d  = temporal_iou_batch(all_preds_clips, all_labels_clips)

    report = classification_report(
        all_labels_flat,
        all_preds_flat,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.ALL_CLASSES,
        zero_division=0,
    )

    return {
        "accuracy"        : acc,
        "recall"          : recall,
        "mean_iou"        : iou_d["mean_iou"],
        "correct_fraction": iou_d["correct_fraction"],
        "iou_threshold"   : iou_d["threshold"],
        "per_class_report": report,
    }


# ──────────────────────────────────────────────
# Baseline scores (Guide §6.4)
# ──────────────────────────────────────────────

def baseline_scores(loader: DataLoader) -> None:
    """
    Print sanity-check baselines.
    Level 1: always predict Background.
    Level 2: random uniform prediction across 29 classes.
    """
    all_labels = []
    for batch in loader:
        all_labels.extend(batch["labels"].reshape(-1).tolist())

    labels_t = torch.tensor(all_labels, dtype=torch.long)
    n        = len(labels_t)

    # Level 1 — always Background
    bg_preds = torch.full((n,), config.BACKGROUND_IDX, dtype=torch.long)
    print("=== Baseline 1: Always predict Background ===")
    print(f"  Accuracy : {per_second_accuracy(bg_preds, labels_t):.4f}")
    print(f"  Recall   : {per_second_recall(bg_preds, labels_t):.4f}")

    # Level 2 — random uniform
    rng        = torch.Generator()
    rng.manual_seed(42)
    rand_preds = torch.randint(0, config.NUM_CLASSES, (n,), generator=rng)
    print("\n=== Baseline 2: Random uniform prediction ===")
    print(f"  Accuracy : {per_second_accuracy(rand_preds, labels_t):.4f}")
    print(f"  Recall   : {per_second_recall(rand_preds, labels_t):.4f}")


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import torch
    from dataset import AVEDataset
    from models import AVEModel

    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = _get_device()

    test_set    = AVEDataset(split="test", use_preextracted=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=16, shuffle=False, num_workers=0,
    )

    ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found at {ckpt}. Run train.py first.")

    model = AVEModel().to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    print(f"Loaded checkpoint: {ckpt}\n")

    print("── Baselines ────────────────────────────────")
    baseline_scores(test_loader)

    print("\n── Model evaluation (test set) ──────────────")
    metrics = evaluate(model, test_loader, device)
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Macro Recall    : {metrics['recall']:.4f}")
    print(f"Mean Temporal IoU: {metrics['mean_iou']:.4f}")
    print(f"IoU≥0.5 fraction: {metrics['correct_fraction']:.4f}")
    print("\nPer-class report:")
    print(metrics["per_class_report"])
