# common.py
# Shared utilities for HW2: Multiple Object Video Detection (VisDrone)
# UCU AI 2026
#
# Usage in any notebook:
#   import sys; sys.path.insert(0, "/kaggle/working")
#   from common import *
#
# Sections:
#   1. Constants & seeds
#   2. Image resize strategies (letterbox / plain / randompad)
#   3. Box coordinate transforms
#   4. Label loading (YOLO + ignore regions)
#   5. Dataset indexing & splits
#   6. Detector wrapper (torchvision-style)
#   7. COCO GT / predictions / evaluation
#   8. Submission writer
# ---------------------------------------------------------------------------

import os
import json
import random
import time
import hashlib
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

import torch
from torchvision.transforms import functional as TF
from torchvision.ops import box_iou
from torchvision.models.detection._utils import Matcher

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------------------------------------------------------------------
# 1. Constants & seeds
# ---------------------------------------------------------------------------

SEED = 28

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASS_NAMES)          # 8
# VisDrone YOLO class ids are 0-7 -> COCO category ids 1-8
YOLO_TO_COCO = {i: i + 1 for i in range(NUM_CLASSES)}
COCO_TO_YOLO = {v: k for k, v in YOLO_TO_COCO.items()}

# Default preprocessing decision (overwritten by 02_preprocessing_fixed.ipynb)
CHOSEN_PREPROCESSING = {"strategy": "letterbox", "size": 1280}

PAD_VALUE = 114  # gray, standard YOLO/letterbox value


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seeds(SEED)


# ---------------------------------------------------------------------------
# 2. Image resize strategies
# ---------------------------------------------------------------------------

def letterbox(img, new_size, pad_value=PAD_VALUE):
    """Keep aspect ratio, resize to fit new_size x new_size, pad with gray."""
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), pad_value, dtype=img.dtype)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, dict(scale=scale, pad_x=left, pad_y=top,
                        new_w=nw, new_h=nh, out_w=new_size, out_h=new_size)


def plain_resize(img, new_size):
    """Stretch to square (distorts aspect ratio). Kept for ablation."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    return resized, dict(scale_x=new_size / w, scale_y=new_size / h,
                         new_w=new_size, new_h=new_size,
                         out_w=new_size, out_h=new_size)


def _stable_seed_from_key(seed_key) -> int:
    """Deterministic integer from any key (str/int), independent of PYTHONHASHSEED.

    Replaces the old `hash(image_path)` which changes between Python sessions.
    """
    if isinstance(seed_key, int):
        return seed_key
    return int(hashlib.md5(str(seed_key).encode("utf-8")).hexdigest()[:8], 16)


def random_pad_canvas(img, canvas_size, seed_key, pad_value=PAD_VALUE):
    """Keep native scale, place at random offset on a fixed canvas.

    seed_key must be deterministic (use image path or (clip, frame) tuple).
    """
    h0, w0 = img.shape[:2]
    canvas_w, canvas_h = canvas_size

    if w0 > canvas_w or h0 > canvas_h:
        scale = min(canvas_w / w0, canvas_h / h0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        new_w, new_h = w0, h0

    rng = random.Random(SEED + _stable_seed_from_key(seed_key))
    max_x = canvas_w - new_w
    max_y = canvas_h - new_h
    pad_x = rng.randint(0, max_x) if max_x > 0 else 0
    pad_y = rng.randint(0, max_y) if max_y > 0 else 0

    canvas = np.full((canvas_h, canvas_w, 3), pad_value, dtype=img.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img

    return canvas, dict(scale=scale, pad_x=pad_x, pad_y=pad_y,
                        new_w=new_w, new_h=new_h,
                        out_w=canvas_w, out_h=canvas_h)


def apply_strategy(img, strategy, size, seed_key=None):
    if strategy == "letterbox":
        return letterbox(img, size)
    if strategy == "plain":
        return plain_resize(img, size)
    if strategy == "randompad":
        return random_pad_canvas(img, size, seed_key=seed_key)
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# 3. Box coordinate transforms
# ---------------------------------------------------------------------------

def scale_boxes_down(boxes, meta, mode):
    """Original-image boxes (x1,y1,x2,y2) -> network-canvas coordinates."""
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if mode in ("letterbox", "randompad"):
        s, px, py = meta["scale"], meta["pad_x"], meta["pad_y"]
        return np.stack([boxes[:, 0] * s + px, boxes[:, 1] * s + py,
                         boxes[:, 2] * s + px, boxes[:, 3] * s + py], axis=1)
    if mode == "plain":
        return np.stack([boxes[:, 0] * meta["scale_x"], boxes[:, 1] * meta["scale_y"],
                         boxes[:, 2] * meta["scale_x"], boxes[:, 3] * meta["scale_y"]],
                        axis=1)
    raise ValueError(mode)


def boxes_to_original(boxes, meta, mode, orig_w, orig_h):
    """Network-canvas boxes -> original-image pixels, clipped to image."""
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if mode in ("letterbox", "randompad"):
        s, px, py = meta["scale"], meta["pad_x"], meta["pad_y"]
        x1 = (boxes[:, 0] - px) / s
        y1 = (boxes[:, 1] - py) / s
        x2 = (boxes[:, 2] - px) / s
        y2 = (boxes[:, 3] - py) / s
    elif mode == "plain":
        x1 = boxes[:, 0] / meta["scale_x"]
        y1 = boxes[:, 1] / meta["scale_y"]
        x2 = boxes[:, 2] / meta["scale_x"]
        y2 = boxes[:, 3] / meta["scale_y"]
    else:
        raise ValueError(mode)
    return np.stack([x1, y1, x2, y2], axis=1).clip(0, [orig_w, orig_h, orig_w, orig_h])


# ---------------------------------------------------------------------------
# 4. Label loading
# ---------------------------------------------------------------------------

def load_yolo_labels(label_path, img_w, img_h):
    """YOLO (class cx cy w h) -> pixel (x1,y1,x2,y2) + int labels.

    Empty / missing file -> (0,4) and (0,) arrays.
    """
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    rows = np.loadtxt(label_path, ndmin=2)
    cls = rows[:, 0].astype(np.int64)
    cx, cy, w, h = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32), cls


def load_ignore_regions(ignore_path, img_w, img_h):
    """Ignore regions (cx cy w h, normalized) -> pixel (x1,y1,x2,y2)."""
    if not os.path.exists(ignore_path) or os.path.getsize(ignore_path) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rows = np.loadtxt(ignore_path, ndmin=2)
    cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Dataset indexing & splits
# ---------------------------------------------------------------------------

def index_split(split_dir):
    """Walk {split}/images/{clip}/{frame}.jpg and build an index DataFrame."""
    split_dir = Path(split_dir)
    rows = []
    for img_path in sorted((split_dir / "images").rglob("*.jpg")):
        clip = img_path.parent.name
        frame = img_path.stem
        rows.append({
            "clip": clip,
            "frame": frame,
            "image_path": str(img_path),
            "label_path": str(split_dir / "labels" / clip / f"{frame}.txt"),
            "ignore_path": str(split_dir / "ignore" / clip / f"{frame}.txt"),
        })
    return pd.DataFrame(rows)


def read_image_sizes(df_index, desc="reading sizes"):
    sizes = []
    for p in tqdm(df_index["image_path"], desc=desc):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        sizes.append((img.shape[1], img.shape[0]))
    df = df_index.copy()
    df["width"], df["height"] = zip(*sizes)
    return df


def save_splits(train_clips, val_clips, splits_dir):
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_clips.txt").write_text("\n".join(sorted(train_clips)))
    (splits_dir / "val_clips.txt").write_text("\n".join(sorted(val_clips)))


def load_splits(splits_dir):
    splits_dir = Path(splits_dir)
    train_clips = (splits_dir / "train_clips.txt").read_text().splitlines()
    val_clips = (splits_dir / "val_clips.txt").read_text().splitlines()
    return train_clips, val_clips


def filter_by_clips(df_index, clips):
    clips = set(clips)
    return df_index[df_index["clip"].isin(clips)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Detector wrapper
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def run_detector(detector, image_rgb, score_thr=0.001, device=None):
    """Run any torchvision-style detector on a single RGB uint8 image.

    Returns (boxes_xyxy_net, labels, scores) in the network canvas coords.
    """
    if device is None:
        device = next(detector.parameters()).device
    tensor = TF.to_tensor(image_rgb)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    out = detector([tensor.to(device)])[0]
    keep = out["scores"] >= score_thr
    return (out["boxes"][keep].cpu().numpy(),
            out["labels"][keep].cpu().numpy(),
            out["scores"][keep].cpu().numpy())


def predict_with_strategy(detector, image_path, strategy, size, score_thr=0.001,
                          device=None):
    """Full pipeline: read -> apply strategy -> detect -> map back to original px."""
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]

    seed_key = image_path  # deterministic via md5 inside random_pad_canvas
    inp, meta = apply_strategy(img, strategy, size, seed_key=seed_key)

    t0 = time.time()
    boxes, labels, scores = run_detector(detector, inp, score_thr=score_thr,
                                         device=device)
    dt = time.time() - t0

    boxes_orig = boxes_to_original(boxes, meta, strategy, w0, h0)
    return boxes_orig, labels, scores, dt, meta


# ---------------------------------------------------------------------------
# 7. COCO GT / predictions / evaluation
# ---------------------------------------------------------------------------

def build_coco_gt(df_index, out_json, include_ignore=True):
    """Build COCO GT from a YOLO-format index.

    - Normal annotations: iscrowd=0, category_id 1..8.
    - Ignore regions: iscrowd=1 (pycocotools will skip detections inside them).
    - Images are stored with absolute file paths in `file_name` so COCOeval
      can resolve them; width/height are read from disk once.
    """
    coco = {"images": [], "annotations": [], "categories": []}
    for i, name in enumerate(CLASS_NAMES, start=1):
        coco["categories"].append({"id": i, "name": name})

    ann_id = 1
    for img_id, (_, row) in enumerate(df_index.iterrows(), start=1):
        img = cv2.imread(row["image_path"], cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        coco["images"].append({
            "id": img_id,
            "file_name": row["image_path"],
            "width": int(w),
            "height": int(h),
        })

        boxes, cls = load_yolo_labels(row["label_path"], w, h)
        for (x1, y1, x2, y2), c in zip(boxes, cls):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(c) + 1,          # YOLO 0..7 -> COCO 1..8
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
            })
            ann_id += 1

        if include_ignore:
            ign = load_ignore_regions(row["ignore_path"], w, h)
            for (x1, y1, x2, y2) in ign:
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,               # dummy category
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 1,                   # <- ignored by COCOeval
                })
                ann_id += 1

    with open(out_json, "w") as f:
        json.dump(coco, f)


def predictions_to_coco(df_index, detector, strategy, size, out_json,
                        score_thr=0.001, device=None):
    """Run detector on df_index and dump COCO-format predictions (pixel coords)."""
    preds = []
    id_map = {row["image_path"]: i for i, (_, row) in enumerate(
        df_index.iterrows(), start=1)}
    valid_cat_ids = set(range(1, NUM_CLASSES + 1))

    for _, row in tqdm(df_index.iterrows(), total=len(df_index),
                       desc=f"pred {strategy}-{size}"):
        boxes, labels, scores, _, _ = predict_with_strategy(
            detector, row["image_path"], strategy, size,
            score_thr=score_thr, device=device)
        img_id = id_map[row["image_path"]]
        for (x1, y1, x2, y2), l, s in zip(boxes, labels, scores):
            if int(l) not in valid_cat_ids:
                continue
            preds.append({
                "image_id": int(img_id),
                "category_id": int(l),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

    with open(out_json, "w") as f:
        json.dump(preds, f)


def coco_eval_summary(gt_json, dt_json):
    """Return dict with mAP, AP50, AP75, AP_small, AP_medium, AP_large."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_json)
        if os.path.getsize(dt_json) == 0 or json.load(open(dt_json)) == []:
            return {k: 0.0 for k in ["mAP", "AP50", "AP75",
                                     "AP_small", "AP_medium", "AP_large"]}
        coco_dt = coco_gt.loadRes(dt_json)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    return {
        "mAP": float(e.stats[0]),
        "AP50": float(e.stats[1]),
        "AP75": float(e.stats[2]),
        "AP_small": float(e.stats[3]),
        "AP_medium": float(e.stats[4]),
        "AP_large": float(e.stats[5]),
    }


# ---------------------------------------------------------------------------
# 8. Submission writer
# ---------------------------------------------------------------------------

def predict_test_and_write_submission(test_csv, detector, strategy, size,
                                      out_csv, score_thr=0.05, max_dets=500,
                                      device=None, data_root=None):
    """Run detector on test set and write Kaggle submission.csv.

    test_csv columns: image_id, file_name, clip, frame, width, height
    file_name may be relative to data_root (e.g. test/images/clip_001/0000001.jpg).
    Output format: image_id,PredictionString where
        PredictionString = "<class> <score> <x> <y> <w> <h> ..." in ORIGINAL pixels.
    """
    df = pd.read_csv(test_csv)
    rows_out = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="test inference"):
        fname = row["file_name"]
        if data_root is not None and not os.path.isabs(fname):
            img_path = str(Path(data_root) / fname)
        else:
            img_path = fname

        boxes, labels, scores, _, _ = predict_with_strategy(
            detector, img_path, strategy, size,
            score_thr=score_thr, device=device)

        # Sort by score, keep top-k, cap to max_dets
        if len(scores) > 0:
            order = np.argsort(-scores)[:max_dets]
            boxes, labels, scores = boxes[order], labels[order], scores[order]

        parts = []
        for (x1, y1, x2, y2), l, s in zip(boxes, labels, scores):
            # Submission uses 0-based YOLO class ids (0..7)
            cls_id = int(l) - 1
            if cls_id < 0 or cls_id >= NUM_CLASSES:
                continue
            w_box = float(x2 - x1)
            h_box = float(y2 - y1)
            if w_box <= 0 or h_box <= 0:
                continue
            parts.append(f"{cls_id} {float(s):.4f} "
                         f"{float(x1):.1f} {float(y1):.1f} "
                         f"{w_box:.1f} {h_box:.1f}")

        rows_out.append({
            "image_id": row["image_id"],
            "PredictionString": " ".join(parts),
        })

    sub = pd.DataFrame(rows_out)
    sub.to_csv(out_csv, index=False)
    return sub


# ---------------------------------------------------------------------------
# Convenience: build val eval subset (for strategy comparison only)
# ---------------------------------------------------------------------------

def make_eval_subset(val_index, n=200, seed=SEED):
    """Small eval subset for fast strategy comparison.

    NOTE: this is only for comparing preprocessing strategies, NOT for model
    benchmarks. Final models are evaluated on the FULL val set.
    """
    return val_index.sample(min(n, len(val_index)), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------

def list_clips(split_dir):
    """Return sorted clip folder names under split_dir/images/."""
    images_root = Path(split_dir) / "images"
    assert images_root.exists(), f"missing images folder: {images_root}"
    return sorted(p.name for p in images_root.iterdir() if p.is_dir())

def build_fullval_gt(val_index, out_json, include_ignore=True):
    """Convenience wrapper: build GT for the full val split and cache it."""
    build_coco_gt(val_index, out_json, include_ignore=include_ignore)
    return out_json

# ---------------------------------------------------------------------------
# 9. Training dataset + collate
# ---------------------------------------------------------------------------

class VisDroneDetectionDataset(torch.utils.data.Dataset):
    """Returns (image_tensor, target) for torchvision detectors.

    target keys: boxes (float32, xyxy, network canvas), labels (int64, 1..8),
                 image_id, area, iscrowd, ignore_boxes (float32, xyxy, canvas).

    Augmentation is applied on ORIGINAL pixel boxes before the resize strategy.
    is_train=True enables augmentation; is_train=False returns clean samples.
    """
    def __init__(self, index_df, strategy, size, is_train=False, transforms=None):
        self.df        = index_df.reset_index(drop=True)
        self.strategy  = strategy
        self.size      = size
        self.is_train  = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        boxes, labels = load_yolo_labels(row["label_path"], w0, h0)
        ign           = load_ignore_regions(row["ignore_path"], w0, h0)

        if self.is_train and self.transforms is not None:
            out = self.transforms(
                image=img,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
                ignore_bboxes=ign.tolist(),
            )
            img    = out["image"]
            boxes  = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(out["labels"], dtype=np.int64).reshape(-1)
            ign    = np.array(out["ignore_bboxes"], dtype=np.float32).reshape(-1, 4)
            h0, w0 = img.shape[:2]

        img_net, meta = apply_strategy(img, self.strategy, self.size,
                                       seed_key=row["image_path"])
        boxes_net = scale_boxes_down(boxes, meta, self.strategy)
        ign_net   = scale_boxes_down(ign,   meta, self.strategy) if len(ign) else \
                    np.zeros((0, 4), dtype=np.float32)

        areas = (boxes_net[:, 2] - boxes_net[:, 0]) * (boxes_net[:, 3] - boxes_net[:, 1]) \
                if len(boxes_net) else np.zeros((0,), dtype=np.float32)

        target = {
            "boxes":        torch.as_tensor(boxes_net, dtype=torch.float32),
            "labels":       torch.as_tensor(labels + 1, dtype=torch.int64),  # bg=0, classes 1..8
            "image_id":     torch.tensor([idx], dtype=torch.int64),
            "area":         torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd":      torch.zeros((len(boxes_net),), dtype=torch.int64),
            "ignore_boxes": torch.as_tensor(ign_net, dtype=torch.float32),
        }

        img_t = TF.to_tensor(img_net)
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img_t, target


def detection_collate(batch):
    """torchvision detectors take a list of (image, target) tuples."""
    return tuple(zip(*batch))

# ---------------------------------------------------------------------------
# 10. Ignore-aware label assignment (training-time ignore masking)
# ---------------------------------------------------------------------------

def assign_labels_with_ignore(anchors, gt_boxes, gt_labels, ignore_boxes,
                              high_thresh=0.5, low_thresh=0.4,
                              ignore_iou_thresh=0.3):
    """Matcher + ignore mask.

    Returns (matched_labels, matched_boxes) with labels in {-1, 0, 1..8}.
    -1 = ignored (anchor overlaps an ignore region, contributes nothing to loss).
    """
    matcher = Matcher(high_thresh, low_thresh, allow_low_quality_matches=True)
    matched_labels, matched_boxes = matcher(anchors, gt_boxes)

    if len(gt_labels) == 0:
        return matched_labels, matched_boxes

    matched_labels = matched_labels.clone()
    matched_labels[matched_labels > 0] = gt_labels[matched_labels[matched_labels > 0] - 1]

    if ignore_boxes is not None and len(ignore_boxes) > 0:
        ignore_iou = box_iou(anchors, ignore_boxes).max(dim=1).values
        ignored    = (ignore_iou > ignore_iou_thresh) & (matched_labels >= 0)
        matched_labels[ignored] = -1

    return matched_labels, matched_boxes

# ---------------------------------------------------------------------------
# 11. Full-val benchmark helper
# ---------------------------------------------------------------------------

def evaluate_full_val(model, val_index, strategy, size, gt_json,
                     score_thr=0.001, device=None, out_json=None):
    """Predict on full val set, evaluate against gt_json, return metrics dict.

    Returns (metrics, pred_json_path). Also caches the JSON for TIDE / WBF reuse.
    """
    if out_json is None:
        out_json = str(Path("/kaggle/working") / "val_preds_tmp.json")
    predictions_to_coco(val_index, model, strategy, size, out_json,
                        score_thr=score_thr, device=device)
    metrics = coco_eval_summary(gt_json, out_json)
    return metrics, out_json

# ---------------------------------------------------------------------------
# 12. Speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed(model, index_df, strategy, size, n=100,
                    device=None, score_thr=0.5, warmup=5):
    """Return dict with GPU ms/image, CPU ms/image, GPU FPS."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    def _run(dev, n_iter):
        times = []
        for i in range(warmup + n_iter):
            img_path = index_df.iloc[i % len(index_df)]["image_path"]
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = predict_with_strategy(model, img_path, strategy, size,
                                      score_thr=score_thr, device=dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.time() - t0)
        return float(np.mean(times)), float(np.std(times))

    gpu_ms = None; gpu_fps = None
    if device != "cpu":
        m, _ = _run(device, n); gpu_ms = m * 1000; gpu_fps = 1.0 / m
    m_cpu, _ = _run("cpu", max(10, n // 5))   # fewer CPU iters

    return {
        "device": str(device),
        "gpu_ms_per_image": gpu_ms,
        "gpu_fps": gpu_fps,
        "cpu_ms_per_image": m_cpu * 1000,
    }

# ---------------------------------------------------------------------------
# 13. Artifact saving
# ---------------------------------------------------------------------------

def save_checkpoint(model, out_path):
    torch.save(model.state_dict(), str(out_path))
    return str(out_path)

def save_metrics(metrics_dict, out_path):
    with open(str(out_path), "w") as f:
        json.dump(metrics_dict, f, indent=2, default=float)
    return str(out_path)

