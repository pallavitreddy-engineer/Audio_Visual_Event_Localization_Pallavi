"""
AVE model: att_Net / AV_att  (Tian et al., ECCV 2018)

Pipeline per clip (10 one-second segments):
  1. AudioGuidedAttention  — audio vec (128) attends to 7×7 video grid (49×512)
                             → attended video vec (512) per second
  2. audio_lstm            — BiLSTM over 10 audio vecs  → (10, 256)
  3. video_lstm            — BiLSTM over 10 attended vecs → (10, 512)
  4. Concat + dropout      — (10, 768)
  5. fc1 + relu            — (10, 256)
  6. fc2                   — (10, 29)   raw logits per second
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class AudioGuidedAttention(nn.Module):
    """
    Audio-guided spatial attention over the 7×7 video feature grid.

    The audio embedding is projected into the video feature space (512-dim),
    then dot-producted with each of the 49 spatial regions to produce attention
    weights.  The output is a single 'attended' video vector per time step.
    """

    def __init__(self, audio_dim: int = 128, video_dim: int = 512, n_regions: int = 49):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, video_dim, bias=False)

    def forward(self, audio: torch.Tensor, video_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio        : (B, 128)      audio embedding for one second
            video_spatial: (B, 49, 512)  spatial grid for the same second

        Returns:
            attended     : (B, 512)      attention-weighted video vector
        """
        proj = self.audio_proj(audio)              # (B, 512)
        proj = proj.unsqueeze(2)                   # (B, 512, 1)
        scores = torch.bmm(video_spatial, proj)    # (B, 49, 1)
        weights = F.softmax(scores, dim=1)         # (B, 49, 1)
        attended = (video_spatial * weights).sum(dim=1)  # (B, 512)
        return attended


class AVEModel(nn.Module):
    """
    Audio-Visual Event Localization model (att_Net).

    Inputs (per forward call — one full clip):
        audio : (B, T, 128)      VGGish features, T=10 seconds
        video : (B, T, 49, 512)  R(2+1)D spatial features, T=10 seconds

    Output:
        logits: (B, T, 29)       raw logits per second, 29 classes
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        audio_dim: int = config.AUDIO_EMBED_DIM,
        video_dim: int = config.VIDEO_FEATURE_DIM,
        n_regions: int = config.VIDEO_NUM_REGIONS,
        lstm_hidden: int = config.LSTM_HIDDEN_DIM,
        fc_hidden: int = config.FC_HIDDEN_DIM * 2,   # 256
        dropout: float = 0.3,
    ):
        super().__init__()

        self.attention = AudioGuidedAttention(audio_dim, video_dim, n_regions)

        # Bidirectional LSTM over audio: 128 → 256
        self.audio_lstm = nn.LSTM(
            input_size=audio_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        audio_lstm_out = lstm_hidden * 2   # 256

        # Bidirectional LSTM over attended video: 512 → 512
        self.video_lstm = nn.LSTM(
            input_size=video_dim,
            hidden_size=lstm_hidden * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        video_lstm_out = lstm_hidden * 4   # 512

        fused_dim = audio_lstm_out + video_lstm_out   # 768

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fused_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        modality_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            audio                : (B, T, 128)
            video                : (B, T, 49, 512)
            modality_dropout_prob: if > 0 and model is training, randomly zero
                                   one modality per batch to prevent dominance

        Returns:
            logits: (B, T, 29)
        """
        B, T, _ = audio.shape

        # Modality dropout (Guide §5 — prevents one modality dominating)
        if self.training and modality_dropout_prob > 0:
            audio, video = _apply_modality_dropout(audio, video, modality_dropout_prob)

        # Step 1: attention at every time step → attended video sequence
        attended_list = []
        for t in range(T):
            att = self.attention(audio[:, t, :], video[:, t, :, :])  # (B, 512)
            attended_list.append(att)
        attended = torch.stack(attended_list, dim=1)   # (B, T, 512)

        # Step 2: temporal modeling
        audio_out, _ = self.audio_lstm(audio)          # (B, T, 256)
        video_out, _ = self.video_lstm(attended)       # (B, T, 512)

        # Step 3: fusion
        fused = torch.cat([audio_out, video_out], dim=-1)   # (B, T, 768)
        fused = self.dropout(fused)

        x = F.relu(self.fc1(fused))                    # (B, T, 256)
        x = self.dropout(x)
        logits = self.fc2(x)                           # (B, T, 29)
        return logits


def _apply_modality_dropout(
    audio: torch.Tensor,
    video: torch.Tensor,
    prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly zero out one full modality for each batch item.

    For each item in the batch, with probability `prob`:
      - half of the time zero the audio  → model must use video
      - half of the time zero the video  → model must use audio
    This prevents the model from ignoring a modality entirely.
    """
    B = audio.size(0)
    r = torch.rand(B, device=audio.device)
    drop_audio = r < prob / 2
    drop_video = (r >= prob / 2) & (r < prob)

    if drop_audio.any():
        audio = audio.clone()
        audio[drop_audio] = 0.0
    if drop_video.any():
        video = video.clone()
        video[drop_video] = 0.0

    return audio, video


# ─────────────────────────────────────────────────────────
# Thin encoder wrappers used during feature pre-extraction
# (kept here so feature_extractor.py can import one place)
# ─────────────────────────────────────────────────────────

class R2Plus1DEncoder(nn.Module):
    """
    R(2+1)D-18 stripped of its avgpool and fc head.

    Input : (B, 3, T, 112, 112)  — RGB frames, Kinetics-normalised
    Output: (B, 512, 7, 7)       — spatial feature map, T-averaged
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models.video as vm
        weights = vm.R2Plus1D_18_Weights.DEFAULT if pretrained else None
        base = vm.r2plus1d_18(weights=weights)
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4   # out: (B, 512, T', 7, 7)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B, 512, T', 7, 7)
        x = x.mean(dim=2)           # (B, 512, 7, 7)  — temporal avg-pool
        return x
