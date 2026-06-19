"""
AVE Dataset class for PyTorch.
Handles loading video/audio, segmentation into 1-second chunks,
and per-second binary label creation.

Supports two modes:
  1. Raw mode: loads raw audio waveforms and video frames (for feature extraction)
  2. Pre-extracted mode: loads pre-extracted VGGish/R(2+1)D features from disk
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import warnings

import config
import utils


class AVEDataset(Dataset):
    """
    PyTorch Dataset for Audio-Visual Event localization.

    Each sample returns:
        - audio: raw waveform chunks (10, 16000) OR pre-extracted VGGish (10, 128)
        - video: raw frame chunks (10, 8, 3, 112, 112) OR spatial R(2+1)D (10, 49, 512)
        - labels: per-second 29-class labels (10,) — category idx 0–27 or Background 28
        - category_idx: integer index of the event category (0–27)
        - video_id: YouTube video ID string
    """

    def __init__(self, split="train", use_preextracted=False):
        """
        Args:
            split: one of 'train', 'val', 'test'
            use_preextracted: if True, load pre-extracted features from disk
        """
        super().__init__()
        self.split = split
        self.use_preextracted = use_preextracted

        # Load the appropriate split file
        split_files = {
            "train": config.TRAIN_SET_FILE,
            "val": config.VAL_SET_FILE,
            "test": config.TEST_SET_FILE,
        }
        if split not in split_files:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_files.keys())}")

        self.samples = utils.load_split_file(split_files[split])

        # Filter out samples whose video files don't exist
        valid_samples = []
        for s in self.samples:
            video_path = utils.get_video_path(s["video_id"])
            if os.path.exists(video_path):
                valid_samples.append(s)
        
        skipped = len(self.samples) - len(valid_samples)
        if skipped > 0:
            warnings.warn(f"Skipped {skipped}/{len(self.samples)} samples with missing video files in {split} split")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        category = sample["category"]
        start_time = sample["start_time"]
        end_time = sample["end_time"]

        # Per-second 29-class labels: event category index or Background (28)
        labels = utils.create_multiclass_labels(category, start_time, end_time)
        labels = torch.from_numpy(labels).long()

        # Category index (0–27) for reference / analysis
        category_idx = config.CATEGORY_TO_IDX.get(category, 0)

        if self.use_preextracted:
            audio_features, video_features = self._load_preextracted(video_id)
        else:
            audio_features, video_features = self._load_raw(video_id)

        return {
            "audio": audio_features,
            "video": video_features,
            "labels": labels,
            "category_idx": category_idx,
            "video_id": video_id,
        }

    def _load_preextracted(self, video_id):
        """Load pre-extracted audio and video features from disk."""
        audio_path = os.path.join(config.FEATURES_DIR, "audio", f"{video_id}.pt")
        video_path = os.path.join(config.FEATURES_DIR, "video", f"{video_id}.pt")

        if os.path.exists(audio_path) and os.path.exists(video_path):
            audio_features = torch.load(audio_path, weights_only=True)
            video_features = torch.load(video_path, weights_only=True)
        else:
            # Fall back to raw loading if pre-extracted features don't exist
            warnings.warn(f"Pre-extracted features not found for {video_id}, loading raw")
            audio_features, video_features = self._load_raw(video_id)

        return audio_features, video_features

    def _load_raw(self, video_id):
        """
        Load raw audio waveform and video frames from the video file.

        Returns:
            audio: tensor of shape (10, 16000) — 10 one-second chunks at 16kHz
            video: tensor of shape (10, 8, 3, 112, 112) — 10 chunks of 8 frames
        """
        video_path = utils.get_video_path(video_id)
        audio = self._extract_audio(video_path)
        video = self._extract_video(video_path)
        return audio, video

    def _extract_audio(self, video_path):
        """
        Extract audio from video, resample to 16kHz, split into 10 one-second chunks.

        Uses torchaudio for loading with automatic anti-aliasing during resampling
        (as described in Section 3.3 of the team guide — the low-pass filter is
        applied automatically before sample-rate reduction).

        Returns:
            tensor of shape (NUM_SEGMENTS, SAMPLES_PER_SEGMENT) = (10, 16000)
        """
        try:
            import torchaudio

            waveform, sr = torchaudio.load(video_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz (torchaudio handles anti-aliasing internally)
            if sr != config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=config.AUDIO_SAMPLE_RATE,
                )
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)  # (total_samples,)

            # Ensure we have exactly 10 seconds of audio
            total_needed = config.NUM_SEGMENTS * config.SAMPLES_PER_SEGMENT
            if waveform.shape[0] >= total_needed:
                waveform = waveform[:total_needed]
            else:
                # Pad with zeros if too short
                padding = total_needed - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Reshape into 10 one-second chunks
            chunks = waveform.reshape(config.NUM_SEGMENTS, config.SAMPLES_PER_SEGMENT)
            return chunks

        except Exception as e:
            warnings.warn(f"Failed to extract audio from {video_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(config.NUM_SEGMENTS, config.SAMPLES_PER_SEGMENT)

    def _extract_video(self, video_path):
        """
        Extract video frames, resize, and split into 10 one-second chunks.
        Each chunk contains VIDEO_FPS (8) evenly-spaced frames.

        Returns:
            tensor of shape (NUM_SEGMENTS, VIDEO_FPS, 3, H, W) = (10, 8, 3, 112, 112)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 25.0  # Fallback FPS

            frames_per_second = fps
            all_chunks = []

            for seg_idx in range(config.NUM_SEGMENTS):
                # Calculate frame indices for this 1-second segment
                seg_start_time = seg_idx
                seg_end_time = seg_idx + 1

                start_frame = int(seg_start_time * frames_per_second)
                end_frame = int(seg_end_time * frames_per_second)
                end_frame = min(end_frame, total_frames)

                # Sample VIDEO_FPS evenly-spaced frames from this segment
                if end_frame > start_frame:
                    frame_indices = np.linspace(
                        start_frame, end_frame - 1,
                        num=config.VIDEO_NUM_FRAMES_PER_SEGMENT,
                        dtype=int,
                    )
                else:
                    frame_indices = np.array([start_frame] * config.VIDEO_NUM_FRAMES_PER_SEGMENT)

                segment_frames = []
                for fi in frame_indices:
                    fi = min(fi, total_frames - 1) if total_frames > 0 else 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap.read()
                    if ret:
                        # Resize to model input size and convert BGR → RGB
                        frame = cv2.resize(frame, config.VIDEO_FRAME_SIZE)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Normalize to [0, 1] and convert to (C, H, W)
                        frame = torch.from_numpy(frame).float() / 255.0
                        frame = frame.permute(2, 0, 1)  # (H,W,C) → (C,H,W)
                    else:
                        frame = torch.zeros(3, *config.VIDEO_FRAME_SIZE)
                    segment_frames.append(frame)

                # Stack frames for this segment: (VIDEO_FPS, 3, H, W)
                chunk = torch.stack(segment_frames, dim=0)
                all_chunks.append(chunk)

            cap.release()

            # Stack all segments: (NUM_SEGMENTS, VIDEO_FPS, 3, H, W)
            video_tensor = torch.stack(all_chunks, dim=0)
            return video_tensor

        except Exception as e:
            warnings.warn(f"Failed to extract video from {video_path}: {e}")
            return torch.zeros(
                config.NUM_SEGMENTS,
                config.VIDEO_NUM_FRAMES_PER_SEGMENT,
                3,
                *config.VIDEO_FRAME_SIZE,
            )


def get_dataloader(split, batch_size=None, use_preextracted=False, num_workers=0):
    """
    Create a DataLoader for the specified split.

    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch size (defaults to config.BATCH_SIZE)
        use_preextracted: whether to use pre-extracted features
        num_workers: number of data loading workers

    Returns:
        DataLoader instance
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = AVEDataset(split=split, use_preextracted=use_preextracted)
    shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
