'''# Two-Stage Anchor-Based Baseline: Faster R-CNN

**Goal.** Train `fasterrcnn_resnet50_fpn_v2` from COCO weights with a 9-class head (8 VisDrone classes + background) on the clip-level train split, under the fixed preprocessing protocol locked in nb02. Evaluate on the **full val split** using **ignore-aware COCO metrics**. Produce a reproducible baseline against which nb04 (tuning), nb05–nb09 (other detector families), and nb11 (final pipeline) will be compared.

**What this notebook is:** the *untuned* baseline. Hyperparameter tuning is deliberately deferred to nb04.
**What this notebook is not:** the leaderboard submission. That is nb12.

**Protocol (locked):**
- Strategy + size: read from `chosen_preprocessing.json` (should be `letterbox-1280`).
- Split: read from `splits/train_clips.txt` and `splits/val_clips.txt`.
- Eval GT: **full val**, ignore-aware (`gt_fullval_ignore.json`, generated once here and reused downstream).
- Augmentation: HorizontalFlip (p=0.5), RandomScale (0.8–1.25), ColorJitter, RandomBrightnessContrast. No rotation, no random crops (both destroy small objects).
- Ignore masking during training: anchors overlapping ignore regions are assigned label `-1` via the custom RPN assigner from `common.py`.

**Artifacts to publish after training:**
- `fasterrcnn_baseline_best.pth`, `fasterrcnn_baseline_last.pth`
- `metrics_fasterrcnn_baseline.json` (metrics + config + W&B URL)
- `gt_fullval_ignore.json`, `gt_fullval_noignore.json`
- `val_preds_baseline_ignore.json` (reused by nb10 TIDE, nb11 WBF, nb13 temporal)
- `failure_case_1.png`, `failure_case_2.png`, `failure_case_3.png`
'''

import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm

COMP_DIR    = Path("/kaggle/input/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026")
DATASET_DIR = Path("/kaggle/input/datasets/suchok/my-dataset")
OUTPUTS_DIR = Path("/kaggle/input/datasets/suchok/outputs")
WORK_DIR    = Path("/kaggle/working")

assert COMP_DIR.exists(),    f"missing competition: {COMP_DIR}"
assert DATASET_DIR.exists(), f"missing my-dataset: {DATASET_DIR}"
assert OUTPUTS_DIR.exists(), f"missing outputs-2 (nb02 artifacts): {OUTPUTS_DIR}"

WORK_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATASET_DIR))
import common, importlib
importlib.reload(common)
from common import *

set_seeds(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE, "| torch:", torch.__version__)



'''2 load the locked protocol'''

chosen = json.load(open(OUTPUTS_DIR / "chosen_preprocessing.json"))
STRAT  = chosen["strategy"]
SIZE   = int(chosen["size"])
print("Locked preprocessing:", STRAT, SIZE)

train_clips, val_clips = load_splits(OUTPUTS_DIR / "splits")
print(f"clips: train={len(train_clips)}  val={len(val_clips)}")
assert set(train_clips).isdisjoint(set(val_clips))

TRAIN_DIR = COMP_DIR / "train"
VAL_DIR   = COMP_DIR / "val"
TEST_CSV  = COMP_DIR / "test_images.csv"

train_index = filter_by_clips(index_split(TRAIN_DIR), train_clips)
val_index   = filter_by_clips(index_split(VAL_DIR),   val_clips)

print(f"frames: train={len(train_index):,}  val={len(val_index):,}")
assert len(train_index) > 0 and len(val_index) > 0

'''
Locked preprocessing: letterbox 1280
clips: train=56  val=7
frames: train=24,201  val=2,846
'''


'''## 3. Build full-val GT files (once)

`nb02` produced GT for the 200-frame subset only — that was for the preprocessing ablation. For model benchmarking we need GT on the **full val set** (2,846 frames). We build both ignore-aware and no-ignore versions here and reuse them in every later notebook.'''

gt_fullval_ignore   = WORK_DIR / "gt_fullval_ignore.json"
gt_fullval_noignore = WORK_DIR / "gt_fullval_noignore.json"

if not gt_fullval_ignore.exists():
    build_fullval_gt(val_index, str(gt_fullval_ignore),   include_ignore=True)
if not gt_fullval_noignore.exists():
    build_fullval_gt(val_index, str(gt_fullval_noignore), include_ignore=False)

def _count(p):
    d = json.load(open(p))
    n0 = sum(a["iscrowd"] == 0 for a in d["annotations"])
    n1 = sum(a["iscrowd"] == 1 for a in d["annotations"])
    return n0, n1

m0, c0 = _count(gt_fullval_ignore)
print(f"gt_fullval_ignore  : {m0:,} normal, {c0:,} crowd")
print(f"gt_fullval_noignore: {len(json.load(open(gt_fullval_noignore))['annotations']):,} normal, 0 crowd")

'''gt_fullval_ignore  : 114,132 normal, 3,995 crowd
gt_fullval_noignore: 114,132 normal, 0 crowd'''


'''## 4. Dataset smoke test (must pass before training)

Instantiate the dataset on a few training frames, verify that boxes are inside the network canvas, labels are in `1..8`, and `ignore_boxes` carries through. Visualize one sample with GT + ignore regions drawn on the letterboxed canvas. This cell catches label-mapping and box-scaling bugs **before** spending GPU hours.'''

import albumentations as A

train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=(-0.2, 0.25), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,
    ),
)

train_ds = VisDroneDetectionDataset(
    train_index.head(8), STRAT, SIZE, is_train=True, transforms=train_transforms,
)
val_ds = VisDroneDetectionDataset(
    val_index.head(8), STRAT, SIZE, is_train=False, transforms=None,
)

img_t, target = train_ds[0]
print("image tensor:", tuple(img_t.shape), "| dtype:", img_t.dtype)
print("labels:", target["labels"].tolist())
print("boxes shape:", tuple(target["boxes"].shape),
      "| ignore_boxes shape:", tuple(target["ignore_boxes"].shape))

assert target["labels"].min() >= 1 and target["labels"].max() <= 8, \
    f"labels must be 1..8 (bg=0), got {target['labels'].tolist()}"
assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()
assert target["boxes"].max() <= SIZE + 1, "boxes must fit in canvas"
print("smoke OK")


