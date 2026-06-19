"""
Pre-extract and save audio (VGGish) and video (R(2+1)D) features for every
clip in the AVE dataset.  Run once before training:

    python feature_extractor.py

Saved shapes per clip:
    features/audio/<video_id>.pt  →  torch.FloatTensor (10, 128)
    features/video/<video_id>.pt  →  torch.FloatTensor (10, 49, 512)

Audio pipeline  (Guide §3.3)
    MP4 → librosa.load(sr=16 000) → 10 one-second chunks → VGGish → (10, 128)
    librosa applies an anti-aliasing low-pass filter automatically before
    downsampling from the native 44 100 Hz to 16 000 Hz.

Video pipeline  (Guide §3.4)
    MP4 → cv2 frames (8 per second, 112×112) → R(2+1)D (no avgpool/fc)
        → (B, 512, T', 7, 7) → temporal mean → (512, 7, 7) → reshape (49, 512)
"""

import os
import warnings
import numpy as np
import torch
import cv2
import librosa

import config
import utils
from models import R2Plus1DEncoder


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# VGGish audio encoder
# ──────────────────────────────────────────────

def build_vggish(device: torch.device):
    """
    Load VGGish from harritaylor/torchvggish via torch.hub.
    postprocess=False: returns raw 128-dim float embeddings (no PCA/quantisation).
    Requires internet access on first call to download model weights (~100 MB).
    """
    model = torch.hub.load(
        "harritaylor/torchvggish", "vggish",
        postprocess=False,
        trust_repo=True,
    )
    model.eval()
    model = model.to(device)
    return model


def extract_audio_features(
    mp4_path: str,
    vggish,
    device: torch.device,
    n_segments: int = config.NUM_SEGMENTS,
    target_sr: int = config.AUDIO_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Extract VGGish features for a single video clip.

    Returns:
        FloatTensor (10, 128)
    """
    try:
        # librosa.load applies anti-aliasing automatically (Guide §3.3)
        y, _ = librosa.load(mp4_path, sr=target_sr, mono=True, duration=10.0)
    except Exception as e:
        warnings.warn(f"Audio load failed for {mp4_path}: {e} — using zeros")
        return torch.zeros(n_segments, config.AUDIO_EMBED_DIM)

    # Pad / truncate to exactly 10 seconds
    target_len = target_sr * n_segments
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # VGGish expects float32 waveform at target_sr
    y = y.astype(np.float32)

    try:
        with torch.no_grad():
            # Forward: waveform numpy → (N, 128), N ≈ n_segments
            embeddings = vggish.forward(y, target_sr)   # (N, 128)
    except Exception as e:
        warnings.warn(f"VGGish forward failed for {mp4_path}: {e} — using zeros")
        return torch.zeros(n_segments, config.AUDIO_EMBED_DIM)

    embeddings = embeddings.cpu().float()

    # Ensure exactly n_segments rows (pad with zeros if VGGish gives fewer)
    if embeddings.size(0) >= n_segments:
        embeddings = embeddings[:n_segments]
    else:
        pad = torch.zeros(n_segments - embeddings.size(0), config.AUDIO_EMBED_DIM)
        embeddings = torch.cat([embeddings, pad], dim=0)

    return embeddings   # (10, 128)


# ──────────────────────────────────────────────
# R(2+1)D video encoder
# ──────────────────────────────────────────────

# Kinetics normalisation constants (matched to R(2+1)D pretrained weights)
_VID_MEAN = torch.tensor([0.43216, 0.39466, 0.37645]).view(3, 1, 1, 1)
_VID_STD  = torch.tensor([0.22803, 0.22145, 0.21700]).view(3, 1, 1, 1)


def _load_segment_frames(
    cap: cv2.VideoCapture,
    seg_idx: int,
    fps: float,
    total_frames: int,
    n_frames: int = config.VIDEO_NUM_FRAMES_PER_SEGMENT,
    size: tuple = config.VIDEO_FRAME_SIZE,
) -> torch.Tensor:
    """
    Sample n_frames uniformly from second [seg_idx, seg_idx+1) of the video.

    Returns:
        FloatTensor (3, n_frames, H, W)  — Kinetics-normalised, ready for R(2+1)D
    """
    start_f = int(seg_idx * fps)
    end_f   = min(int((seg_idx + 1) * fps), total_frames)
    if end_f <= start_f:
        end_f = start_f + 1

    indices = np.linspace(start_f, end_f - 1, n_frames, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)

    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0    # (H, W, 3)
            frame = frame.permute(2, 0, 1)                     # (3, H, W)
        else:
            frame = torch.zeros(3, *size)
        frames.append(frame)

    clip = torch.stack(frames, dim=1)   # (3, n_frames, H, W)

    # Kinetics normalisation
    mean = _VID_MEAN.to(clip.device)
    std  = _VID_STD.to(clip.device)
    clip = (clip - mean) / std
    return clip


def extract_video_features(
    mp4_path: str,
    encoder: R2Plus1DEncoder,
    device: torch.device,
    n_segments: int = config.NUM_SEGMENTS,
    n_frames: int = config.VIDEO_NUM_FRAMES_PER_SEGMENT,
) -> torch.Tensor:
    """
    Extract R(2+1)D spatial features for a single video clip.

    Returns:
        FloatTensor (10, 49, 512)
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        warnings.warn(f"Cannot open video: {mp4_path} — using zeros")
        return torch.zeros(n_segments, config.VIDEO_NUM_REGIONS, config.VIDEO_FEATURE_DIM)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    seg_features = []
    for seg_idx in range(n_segments):
        clip = _load_segment_frames(cap, seg_idx, fps, total_frames, n_frames)
        clip = clip.unsqueeze(0).to(device)   # (1, 3, n_frames, H, W)

        try:
            with torch.no_grad():
                feat = encoder(clip)           # (1, 512, 7, 7)
        except Exception as e:
            warnings.warn(f"R(2+1)D forward failed: {e}")
            feat = torch.zeros(1, config.VIDEO_FEATURE_DIM,
                               config.VIDEO_SPATIAL_SIZE, config.VIDEO_SPATIAL_SIZE,
                               device=device)

        feat = feat.squeeze(0)                # (512, 7, 7)
        feat = feat.permute(1, 2, 0)          # (7, 7, 512)
        feat = feat.reshape(config.VIDEO_NUM_REGIONS, config.VIDEO_FEATURE_DIM)  # (49, 512)
        seg_features.append(feat.cpu())

    cap.release()
    return torch.stack(seg_features, dim=0)   # (10, 49, 512)


# ──────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────

def extract_all_features(
    split_files: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Pre-extract audio and video features for every unique video in the dataset.

    Args:
        split_files: list of split file paths to process (default: all three)
        overwrite  : re-extract even if the .pt file already exists
    """
    if split_files is None:
        split_files = [config.TRAIN_SET_FILE, config.VAL_SET_FILE, config.TEST_SET_FILE]

    utils.ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    print("Loading VGGish …")
    vggish = build_vggish(device)

    print("Loading R(2+1)D-18 …")
    encoder = R2Plus1DEncoder(pretrained=True).to(device)
    encoder.eval()

    # Collect all unique video IDs across all splits
    all_samples = []
    for sf in split_files:
        all_samples.extend(utils.load_split_file(sf))

    seen = set()
    unique_samples = []
    for s in all_samples:
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            unique_samples.append(s)

    print(f"Extracting features for {len(unique_samples)} unique videos …\n")

    audio_dir = os.path.join(config.FEATURES_DIR, "audio")
    video_dir = os.path.join(config.FEATURES_DIR, "video")

    for i, sample in enumerate(unique_samples, 1):
        vid = sample["video_id"]
        mp4_path = utils.get_video_path(vid)

        audio_out = os.path.join(audio_dir, f"{vid}.pt")
        video_out = os.path.join(video_dir, f"{vid}.pt")

        if not os.path.exists(mp4_path):
            print(f"  [{i}/{len(unique_samples)}] SKIP (missing) {vid}")
            continue

        if not overwrite and os.path.exists(audio_out) and os.path.exists(video_out):
            print(f"  [{i}/{len(unique_samples)}] SKIP (exists)  {vid}")
            continue

        print(f"  [{i}/{len(unique_samples)}] {vid}", end=" … ", flush=True)

        if overwrite or not os.path.exists(audio_out):
            af = extract_audio_features(mp4_path, vggish, device)
            torch.save(af, audio_out)

        if overwrite or not os.path.exists(video_out):
            vf = extract_video_features(mp4_path, encoder, device)
            torch.save(vf, video_out)

        print("done")

    print("\nFeature extraction complete.")
    print(f"  Audio saved to : {audio_dir}/")
    print(f"  Video saved to : {video_dir}/")


if __name__ == "__main__":
    extract_all_features()
