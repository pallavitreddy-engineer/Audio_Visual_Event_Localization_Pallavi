"""
Full AVE pipeline: feature extraction -> training -> evaluation.
Run:  conda run -n torch_env python run_pipeline.py
Live log: pipeline.log
"""
import os, sys, time, logging, warnings
warnings.filterwarnings("ignore")

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger()

import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, utils
from models import AVEModel, R2Plus1DEncoder
from dataset import AVEDataset
import evaluate as ev
# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    props  = torch.cuda.get_device_properties(0)
    log.info(f"GPU : {props.name}  ({props.total_memory//1024**2} MB VRAM)")
else:
    DEVICE = torch.device("cpu")
    log.info("CPU mode")

log.info("=" * 60)
log.info("AVE FULL PIPELINE")
log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def stage1_extract():
    import librosa, cv2

    audio_dir = os.path.join(config.FEATURES_DIR, "audio")
    video_dir = os.path.join(config.FEATURES_DIR, "video")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    done_a = {f[:-3] for f in os.listdir(audio_dir) if f.endswith(".pt")}
    done_v = {f[:-3] for f in os.listdir(video_dir) if f.endswith(".pt")}

    all_samples = []
    for sf in [config.TRAIN_SET_FILE, config.VAL_SET_FILE, config.TEST_SET_FILE]:
        all_samples.extend(utils.load_split_file(sf))
    seen, unique = set(), []
    for s in all_samples:
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            unique.append(s)

    todo = [s for s in unique
            if s["video_id"] not in done_a or s["video_id"] not in done_v]

    log.info("")
    log.info("─" * 60)
    log.info("STAGE 1 — FEATURE EXTRACTION")
    log.info("─" * 60)
    log.info(f"Unique videos   : {len(unique)}")
    log.info(f"Already done    : {len(unique) - len(todo)}")
    log.info(f"To extract      : {len(todo)}")

    if not todo:
        log.info("All features already extracted — skipping.")
        return

    log.info("Loading VGGish ...")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish",
                             postprocess=False, trust_repo=True)
    vggish.eval().to(DEVICE)
    log.info("VGGish ready.")

    log.info("Loading R(2+1)D-18 ...")
    encoder = R2Plus1DEncoder(pretrained=True).to(DEVICE)
    encoder.eval()
    log.info("R(2+1)D ready.")

    _MEAN = torch.tensor([0.43216, 0.39466, 0.37645]).view(3, 1, 1, 1).to(DEVICE)
    _STD  = torch.tensor([0.22803, 0.22145, 0.21700]).view(3, 1, 1, 1).to(DEVICE)

    def audio_feat(mp4):
        try:
            y, _ = librosa.load(mp4, sr=config.AUDIO_SAMPLE_RATE, mono=True, duration=10.0)
            n = config.AUDIO_SAMPLE_RATE * config.NUM_SEGMENTS
            y = np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)
            with torch.no_grad():
                emb = vggish.forward(y, config.AUDIO_SAMPLE_RATE).cpu().float()
            if emb.size(0) >= config.NUM_SEGMENTS:
                return emb[:config.NUM_SEGMENTS]
            return torch.cat([emb, torch.zeros(config.NUM_SEGMENTS - emb.size(0), 128)], 0)
        except Exception as e:
            log.warning(f"    audio fail: {e}")
            return torch.zeros(config.NUM_SEGMENTS, config.AUDIO_EMBED_DIM)

    def video_feat(mp4):
        cap = cv2.VideoCapture(mp4)
        if not cap.isOpened():
            return torch.zeros(config.NUM_SEGMENTS, config.VIDEO_NUM_REGIONS, config.VIDEO_FEATURE_DIM)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segs = []
        for si in range(config.NUM_SEGMENTS):
            sf = int(si * fps)
            ef = max(min(int((si + 1) * fps), nf), sf + 1)
            idx = np.clip(np.linspace(sf, ef - 1, config.VIDEO_NUM_FRAMES_PER_SEGMENT, dtype=int), 0, nf - 1)
            frames = []
            for fi in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, fr = cap.read()
                if ret:
                    fr = cv2.resize(fr, config.VIDEO_FRAME_SIZE)
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    fr = torch.from_numpy(fr).float().div(255.0).permute(2, 0, 1)
                else:
                    fr = torch.zeros(3, *config.VIDEO_FRAME_SIZE)
                frames.append(fr)
            clip = torch.stack(frames, dim=1).unsqueeze(0).to(DEVICE)  # (1,3,T,H,W)
            clip = (clip - _MEAN) / _STD
            try:
                with torch.no_grad():
                    feat = encoder(clip).squeeze(0)  # (512,7,7)
            except Exception:
                feat = torch.zeros(config.VIDEO_FEATURE_DIM,
                                   config.VIDEO_SPATIAL_SIZE, config.VIDEO_SPATIAL_SIZE, device=DEVICE)
            segs.append(feat.cpu().permute(1,2,0).reshape(config.VIDEO_NUM_REGIONS, config.VIDEO_FEATURE_DIM))
        cap.release()
        return torch.stack(segs, 0)  # (10,49,512)

    t0 = time.time()
    for i, s in enumerate(todo, 1):
        vid = s["video_id"]
        mp4 = utils.get_video_path(vid)
        if not os.path.exists(mp4):
            log.info(f"  [{i}/{len(todo)}] SKIP (no mp4) {vid}")
            continue
        ap = os.path.join(audio_dir, f"{vid}.pt")
        vp = os.path.join(video_dir, f"{vid}.pt")
        t1 = time.time()
        if vid not in done_a:
            torch.save(audio_feat(mp4), ap)
        if vid not in done_v:
            torch.save(video_feat(mp4), vp)
        done_a.add(vid); done_v.add(vid)
        secs = time.time() - t1
        eta  = (time.time() - t0) / i * (len(todo) - i) / 60
        log.info(f"  [{i:4d}/{len(todo)}] {vid}  {secs:.1f}s  ETA {eta:.0f} min")

    log.info(f"Stage 1 done  ({(time.time()-t0)/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def stage2_train():
    log.info("")
    log.info("─" * 60)
    log.info("STAGE 2 — TRAINING")
    log.info("─" * 60)

    train_ds = AVEDataset("train", use_preextracted=True)
    val_ds   = AVEDataset("val",   use_preextracted=True)
    tr_ldr   = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    va_ldr   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
    log.info(f"Train : {len(train_ds)} clips  |  Val : {len(val_ds)} clips")

    cw        = utils.compute_class_weights(utils.load_split_file(config.TRAIN_SET_FILE)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    model     = AVEModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    log.info(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    t0         = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # train
        model.train()
        tl, tp, tlab = 0.0, [], []
        for b in tr_ldr:
            a = b["audio"].to(DEVICE); v = b["video"].to(DEVICE); l = b["labels"].to(DEVICE)
            optimizer.zero_grad()
            out  = model(a, v, modality_dropout_prob=config.MODALITY_DROPOUT_PROB)
            loss = criterion(out.reshape(-1, config.NUM_CLASSES), l.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tl += loss.item(); tp.append(out.argmax(-1).cpu()); tlab.append(l.cpu())
        tp  = torch.cat([x.reshape(-1) for x in tp])
        tlab= torch.cat([x.reshape(-1) for x in tlab])
        tr_loss = tl / len(tr_ldr)
        tr_acc  = ev.per_second_accuracy(tp, tlab)
        tr_rec  = ev.per_second_recall(tp, tlab)

        # val
        model.eval()
        vl, vp, vlab = 0.0, [], []
        with torch.no_grad():
            for b in va_ldr:
                a = b["audio"].to(DEVICE); v = b["video"].to(DEVICE); l = b["labels"].to(DEVICE)
                out  = model(a, v, modality_dropout_prob=0.0)
                loss = criterion(out.reshape(-1, config.NUM_CLASSES), l.reshape(-1))
                vl += loss.item(); vp.append(out.argmax(-1).cpu()); vlab.append(l.cpu())
        vp   = torch.cat([x.reshape(-1) for x in vp])
        vlab = torch.cat([x.reshape(-1) for x in vlab])
        va_loss = vl / len(va_ldr)
        va_acc  = ev.per_second_accuracy(vp, vlab)
        va_rec  = ev.per_second_recall(vp, vlab)

        scheduler.step(va_loss)
        elapsed = (time.time() - t0) / 60

        log.info(
            f"Ep {epoch:3d}/{config.NUM_EPOCHS} | "
            f"tr loss {tr_loss:.4f} acc {tr_acc:.3f} rec {tr_rec:.3f} | "
            f"va loss {va_loss:.4f} acc {va_acc:.3f} rec {va_rec:.3f} | "
            f"{elapsed:.0f} min"
        )

        if va_loss < best_loss:
            best_loss  = va_loss; no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            log.info(f"  -> best model saved  (val_loss={best_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOPPING_PATIENCE:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    log.info(f"Stage 2 done  best_val_loss={best_loss:.4f}  ({(time.time()-t0)/60:.1f} min)")
    return best_ckpt


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def stage3_evaluate(ckpt):
    from sklearn.metrics import classification_report
    log.info("")
    log.info("─" * 60)
    log.info("STAGE 3 — EVALUATION  (test set)")
    log.info("─" * 60)

    test_ds  = AVEDataset("test", use_preextracted=True)
    te_ldr   = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    log.info(f"Test set : {len(test_ds)} clips")

    model = AVEModel().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=DEVICE))
    model.eval()

    pf, lf, pc, lc = [], [], [], []
    with torch.no_grad():
        for b in te_ldr:
            a = b["audio"].to(DEVICE); v = b["video"].to(DEVICE); labs = b["labels"]
            out = model(a, v, modality_dropout_prob=0.0).argmax(-1).cpu()
            for p, l in zip(out.numpy(), labs.numpy()):
                pc.append(p); lc.append(l)
                pf.extend(p.tolist()); lf.extend(l.tolist())

    pt = torch.tensor(pf, dtype=torch.long)
    lt = torch.tensor(lf, dtype=torch.long)
    acc  = ev.per_second_accuracy(pt, lt)
    rec  = ev.per_second_recall(pt, lt)
    ious = np.array([ev.temporal_iou_single(p, g) for p, g in zip(pc, lc)])
    miou = float(ious.mean())
    iou5 = float((ious >= 0.5).mean())

    log.info("")
    log.info("══ TEST RESULTS ═════════════════════════════")
    log.info(f"  Per-second accuracy    : {acc:.4f}")
    log.info(f"  Macro recall (primary) : {rec:.4f}")
    log.info(f"  Mean Temporal IoU      : {miou:.4f}")
    log.info(f"  Clips with IoU >= 0.5  : {iou5:.4f}")
    log.info("══════════════════════════════════════════════")

    report = classification_report(lf, pf,
                labels=list(range(config.NUM_CLASSES)),
                target_names=config.ALL_CLASSES,
                zero_division=0)
    log.info("\nPer-class report:\n" + report)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Per-second accuracy    : {acc:.4f}\n")
        f.write(f"Macro recall           : {rec:.4f}\n")
        f.write(f"Mean Temporal IoU      : {miou:.4f}\n")
        f.write(f"Clips with IoU >= 0.5  : {iou5:.4f}\n\n")
        f.write(report)
    log.info(f"Results saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    stage1_extract()
    ckpt = stage2_train()
    stage3_evaluate(ckpt)
    log.info("")
    log.info(f"PIPELINE COMPLETE  ({(time.time()-t0)/60:.1f} min total)")
    log.info(f"  Log     -> pipeline.log")
    log.info(f"  Results -> results.txt")
    log.info(f"  Model   -> {ckpt}")
    log.info(f"  MLflow  -> run: mlflow ui")
