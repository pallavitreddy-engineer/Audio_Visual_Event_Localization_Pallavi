# Audio-Visual Event Localization (AVE)

Per-second audio-visual event localization on the AVE dataset using a BiLSTM attention network.  
Given a 10-second video clip, the model predicts the event category for every second (28 categories + Background).

---

## Results

| Metric | Score |
|--------|-------|
| Per-second accuracy | 60.0% |
| **Macro recall** (primary) | **72.8%** |
| Mean Temporal IoU | 81.7% |
| Clips with IoU ≥ 0.5 | 83.1% |

Full per-class report: [`results.txt`](results.txt)

---

## Dataset

- **AVE dataset**: 4,143 video clips × 10 seconds, 28 event categories + Background = 29 classes
- Per-second labels derived from `[start_time, end_time)` annotations; all other seconds = Background (index 28)
- Splits: train (3,339) / val (402) / test (402 approx.)

---

## Architecture

```
Audio (10 × 16000 Hz)          Video (10 × 8 frames × 112×112)
       │                                      │
   VGGish                              R(2+1)D-18
       │                                      │
  (10, 128)                   temporal mean-pool → (10, 49, 512)
       │                                      │
       └──── AudioGuidedAttention ────────────┘
                       │
              attended video (10, 512)
                       │
       ┌───────────────┴───────────────┐
  BiLSTM (audio)                BiLSTM (video)
  128 → 256                     512 → 512
       │                              │
       └──────────── concat ──────────┘
                       │
                  768 → fc(256) → fc(29)
                       │
              per-second logits (10, 29)
```

**Key design choices**
- **Audio-guided attention**: audio embedding attends over 49 spatial video regions to select the relevant area
- **Modality dropout** (p=0.1): randomly zeros one modality per batch item during training to prevent one modality dominating
- **Class-weighted loss**: inverse-frequency weights handle the 29-class imbalance
- **Gradient clipping** at max norm 5.0 for BiLSTM stability

---

## Project Structure

```
Audio_visual_event_localization/
├── config.py              # All hyperparameters and paths
├── utils.py               # Annotation parsing, class weights
├── dataset.py             # AVEDataset (raw and pre-extracted modes)
├── models.py              # AVEModel, AudioGuidedAttention, R2Plus1DEncoder
├── feature_extractor.py   # VGGish + R(2+1)D feature extraction
├── train.py               # Training loop utilities
├── evaluate.py            # Accuracy, recall, temporal IoU metrics
├── run_pipeline.py        # End-to-end: extract → train → evaluate
│
├── features/
│   ├── audio/             # Pre-extracted VGGish features (10, 128) per video
│   └── video/             # Pre-extracted R(2+1)D features (10, 49, 512) per video
├── checkpoints/
│   └── best_model.pt      # Best model checkpoint (saved by val loss)
├── results.txt            # Final test-set evaluation report
└── pipeline.log           # Full run log
```

---

## Setup

```bash
conda create -n torch_env python=3.10
conda activate torch_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install librosa opencv-python resampy scikit-learn
```

**Requirements**: NVIDIA GPU with CUDA, ~1.2 GB disk for pre-extracted features.

---

## Run

```bash
conda activate torch_env
python run_pipeline.py
```

The pipeline runs 3 stages automatically:

| Stage | Description | Time |
|-------|-------------|------|
| 1 | Feature extraction (VGGish + R(2+1)D) for all videos | ~8 hrs (once only) |
| 2 | Training — 30 epochs max, early stopping patience=5 | ~10 min |
| 3 | Test-set evaluation — accuracy, recall, temporal IoU | ~1 min |

Features are cached — if Stage 1 was already run, it is skipped automatically on the next run.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Dropout | 0.3 |
| LSTM hidden (audio) | 128 → 256 (BiLSTM) |
| LSTM hidden (video) | 256 → 512 (BiLSTM) |

---

## Metrics

- **Macro recall**: average recall across 28 event classes (Background excluded) — primary metric
- **Temporal IoU**: intersection-over-union of predicted vs ground-truth event windows per clip
- **Accuracy**: fraction of seconds correctly classified

### Baseline comparison

| Baseline | Recall |
|----------|--------|
| Always predict Background | 0.0% |
| Random uniform (29 classes) | ~3.4% |
| **Our att_Net (this run)** | **72.8%** |

---

## GPU

Tested on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.4 / driver 591.74.
