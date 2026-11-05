# Technical Report Draft: Multiple Object Video Detection (VisDrone)
**UCU AI 2026 — HW2**

> **Note on this document:** This is an internal working draft and knowledge map. It contains every insight extracted from EDA, preprocessing, and the baseline model. This is *not* the final submission report. It is meant to help you understand the exact state of your project, the meaning of your metrics, and the justifications for future decisions (Tuning, One-stage, Anchor-free, Query-based, and the Final Pipeline). Use this to write your final 1500-word report in your own words.

---

## 1. Data Understanding (EDA) & Motivation

### 1.1. Dataset Overview
The VisDrone2019-VID benchmark requires detecting 8 classes of objects in unlabelled drone footage. The training set contains 24,201 frames (1.61M annotations) across 56 clips, while the validation set contains 2,846 frames (146K annotations) across 7 clips. Image resolutions vary drastically, from 1344×756 up to 3840×2160.

### 1.2. Key EDA Insights
*   **Class Imbalance:** The dataset is heavily dominated by `car` (35.2%, 569K instances) and `person` (24.3%, 392K instances). The rarest classes are `bus` (4.5%, 73K instances) and `truck` (5.8%, 94K instances). *Impact:* This imbalance biases models toward the majority classes. We must use Focal Loss or class-weighted sampling. The low count of buses and trucks directly explains why models struggle to distinguish them from vans.
*   **Object Sizes:** Using COCO size thresholds, **24.9% of objects are small** (<32×32 px), 63.5% are medium (32–96 px), and only 11.6% are large (>=96 px). *Impact:* Standard detectors pool features to 7×7. A 20×20 pixel object gets reduced to 3×3 pixels of features. This justifies training at high resolution (1280×1280) and using slicing strategies (SAHI) at inference.
*   **Scene Density:** The average frame contains **66.8 objects**, but the maximum is **342**. *Impact:* Dense scenes cause standard NMS to accidentally suppress valid neighboring objects. This directly motivates the use of Soft-NMS in our final pipeline. For query-based models, this means we need at least 300 queries.
*   **Clip Variance:** Object sizes vary drastically per sequence (mean side length ranges from 28px to 156px). *Impact:* We must use a **clip-level split**, not a frame-level split. Consecutive frames share >95% of object instances; a frame-level split would cause extreme data leakage and falsely inflate our validation metrics.
*   **Recommended Anchors (K-Means):** We clustered the ground-truth widths and heights (k=9) to ground the anchor boxes in our specific data. The resulting anchors range from tiny (w=23.8, h=33.7) to massive (w=336.4, h=221.7), confirming that we need multi-scale feature pyramids to capture everything from single pedestrians to large buses.

---

## 2. Preprocessing Ablation

Before committing to any model training, we tested four resize and padding strategies using a zero-shot COCO-pretrained Faster R-CNN on a 200-frame validation subset. The decision metric was **mAP_ignore** (using `iscrowd=1`).

*   `letterbox-1280`: mAP 0.0125 (Winner)
*   `letterbox-1024`: mAP 0.0122
*   `randompad-(2720,1530)`: mAP 0.0116
*   `plain-1280`: mAP 0.0083

### Key Insights:
1.  **Aspect Distortion is Fatal:** `plain_resize` was ~34% worse than the others. Stretching a 3840×2160 aerial frame into a 1280×1280 square distorts object aspect ratios by roughly 3×, destroying the geometric features required to localize cars and pedestrians.
2.  **Randompad induced Distribution Shift:** `randompad` produced the most detections, but many were false positives (noise on road markings). Because the canvas was 2720×1530, `random_pad_canvas` downscaled the largest 4K images before padding, losing small cues. The off-center placement also shifted the distribution away from what the pretrained model expected.
3.  **Letterbox-1280 is the Middle Path:** `letterbox-1280` won. The margin over `letterbox-1024` was only +0.0003 mAP, which is statistically a tie at 200 frames (bootstrap SE ≈ 0.001). We chose 1280 because it preserves more small objects (24.9% are <32 px) and incurs no inference speed penalty (0.197 s/img vs 0.208 s/img).
4.  **Ignore-Awareness is Mandatory:** An early version of this ablation evaluated with `iscrowd=0` (no ignore regions). The ranking was skewed because strategies that generated many boxes in crowds and parking lots looked better than they actually were. The task metric exempts detections inside crowds, so evaluating without `iscrowd=1` rewards the wrong behavior.

---

## 3. Two-Stage Anchor-based: Faster R-CNN Baseline (nb03)

### 3.1. Setup & Fixed-Compute Protocol
We fine-tuned `fasterrcnn_resnet50_fpn_v2` (COCO-pretrained) with a 9-class head (8 VisDrone + background). The model has **43.3M trainable parameters**. Training used SGD (lr=0.005, momentum=0.9) with a StepLR schedule, batch size 2, and the locked `letterbox-1280` preprocessing.

**The Time Constraint:** A single epoch at 1280×1280 on a T4 GPU takes ~3.2 hours (11,809 seconds). A standard 10-epoch fine-tune would take ~32 hours, exceeding Kaggle's 12-hour session limit. Therefore, the model was trained for exactly **1 epoch**.

**Ignore-Region Handling:** The full val GT contains 114,132 normal boxes and 3,995 crowd regions. During training, we masked anchors overlapping ignore regions by assigning them label `-1` via a custom `IgnoreAwareAssigner` wrapping the RPN's Matcher. Without this, the model would learn to treat pedestrians in dense crowds as background.

**Training Sanity:** A single-batch smoke test before training produced `loss_classifier=2.32`, `loss_box_reg=0.67`, `loss_objectness=0.25`, `loss_rpn_box_reg=0.16` (total = 3.41). After 1 epoch, the mean training loss dropped to **0.6861**, confirming learning.

### 3.2. Results & Interpretation
Evaluated on the full validation set (2,846 frames):
*   **mAP:** 0.1373
*   **AP50:** 0.2809
*   **AP75:** 0.1209
*   **AP_small:** 0.0557
*   **AP_medium:** 0.1468
*   **AP_large:** 0.2515
*   **Inference Speed:** 4.67 FPS (214 ms/image on T4 GPU).

**Insight 1 (AP50 vs mAP gap):** The large gap between AP50 (0.28) and mAP (0.13) indicates that the RPN is successfully proposing regions containing objects, but the box regression head has not converged to tightly align the edges. This is expected after only 1 epoch.

**Insight 2 (The Small Object Problem):** The model performs ~4.5× better on large objects (AP_large = 0.251) than on small objects (AP_small = 0.055). Standard RoI pooling resizes features to a 7×7 grid. A 31px object becomes ~4px, making it unclassifiable. This confirms the EDA and justifies SAHI.

### 3.3. Failure Case Analysis
1.  **Missed Small Objects:** RPN retains only the top-1000 proposals during training. A 31px person competes poorly against large road/vegetation anchors. RoI pooling of a 31px object yields a ~4px feature map.
2.  **Duplicate Boxes in Dense Scenes:** In parking lots and crowds, standard NMS (IoU threshold 0.5) produces overlapping boxes. Two adjacent cars have IoU ≈ 0.5 with each other, but only ≈ 0.4 with the single ground-truth box. This justifies Soft-NMS.
3.  **Van ↔ Truck Misclassification:** Visually similar classes combined with class imbalance (trucks = 5.8%). A 40×30px van is pooled to ≈5×4px of features.

**Qualitative Check:** The 6-frame prediction grid confirms the metrics. On sparse highway scenes, the model confidently detects cars. In the dense basketball-court scene, the model over-predicts (~48 detections vs 52 GT), producing many overlapping duplicates.

---

## 4. What to do next (and why)

Based on the baseline insights, here is the roadmap for the remaining notebooks:

### 4.1. Two-Stage Tuning (nb04)
*   *Why:* The baseline missed small objects (AP_small=0.056) and confused vans/trucks.
*   *Experiment A:* Increase RPN proposals (`rpn_pre_nms_top_n_train=2000`). This directly targets the missed small objects.
*   *Experiment B:* Increase RoI sampling (`box_positive_fraction=0.5`). More positive examples per batch gives the classifier more gradient signal, targeting the van/truck confusion.

### 4.2. One-Stage Anchor-based (nb05, nb06)
*   *Why:* RetinaNet uses Focal Loss, which explicitly handles foreground-background imbalance. It also uses the K-Means anchors from EDA.
*   *What to test:* Does Focal Loss fix the class imbalance problem better than the two-stage RoI sampler? Does the lack of an RPN hurt or help dense scenes?

### 4.3. One-Stage Anchor-free (nb07, nb08)
*   *Why:* FCOS assigns boxes based on center sampling. In a dense basketball court, a person's center pixel might be surrounded by other people. Does this hurt?
*   *What to test:* Compare AP_small and AP_large directly with Faster R-CNN. Test `center_sampling_radius` (1.0 vs 1.5).

### 4.4. Query-based (nb09)
*   *Why:* DETR uses bipartite matching instead of NMS, which theoretically solves the duplicate box problem.
*   *What to test:* With 342 max objects per frame, we need >= 300 queries. Plot mAP over training budget against Faster R-CNN.

### 4.5. Your Pipeline (nb11)
*   *Why:* AP_small is the main bottleneck.
*   *Strategy 1:* **SAHI** at inference. Slicing 1280×1280 images into 640×640 patches increases the effective size of small objects.
*   *Strategy 2:* **Soft-NMS**. Targets the duplicate boxes in dense scenes.
*   *Strategy 3:* **Ensemble (WBF)**. Combining Faster R-CNN (strong on medium/large) with FCOS or RetinaNet (potentially better on small) could boost overall mAP.

### 4.6. Bonus (nb13)
*   *Temporal:* Use Seq-NMS or ByteTrack to link high-confidence detections across video frames, smoothing out missed detections.
*   *Visualization:* Extract RoI features and use UMAP to visualize van vs truck clusters, proving the misclassification cause.