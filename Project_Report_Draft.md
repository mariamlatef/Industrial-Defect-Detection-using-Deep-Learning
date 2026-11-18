# Project Report: Industrial Defect Detection using Deep Learning
### Dataset: DAGM 2007 Industrial Surface Defect Detection

---

## 1. Problem Formulation & EDA

### 1.1 Context and Industrial Motivation

Automated surface-defect inspection is a foundational computer vision
problem in manufacturing quality assurance. Production lines for textiles,
metal sheeting, and other textured materials must reliably flag defective
units — scratches, cracks, contamination, weave irregularities — before they
proceed downstream. Manual visual inspection does not scale to modern
production throughput and introduces variance due to inspector fatigue and
subjectivity.

The **DAGM 2007** (Deutsche Arbeitsgemeinschaft für Mustererkennung 2007)
dataset was created precisely to benchmark automated defect-detection
methods. It contains ten classes of artificially generated but statistically
realistic grayscale texture images, each with a binary defect flag. For this
project, we collapse the multi-class textures into a single **binary
classification problem**, which mirrors a realistic single-inspection-station
deployment scenario: *is this surface acceptable, or does it require manual
review?*

### 1.2 Input / Output Definition

| | Definition |
|---|---|
| **Input**  | A single grayscale (converted to RGB for model compatibility) surface image, resized to `224 x 224`. |
| **Output** | A binary label: `0` = Normal (defect-free), `1` = Defect (surface anomaly present). |
| **Task type** | Supervised binary image classification. |
| **Evaluation metrics** | Accuracy, Precision, Recall, F1-Score (Recall on the Defect class is weighted most heavily operationally, since missed defects are the costlier failure mode). |

### 1.3 Dataset Distribution Analysis

Exploratory analysis (see `eda.py` and the generated figures) covers three
angles:

1. **Class Distribution** (`class_distribution.png`): DAGM 2007 is
   moderately imbalanced toward Normal images within each of its ten texture
   classes (most classes contain substantially more defect-free samples than
   defective ones per class folder). This imbalance motivates:
   - Stratified splitting (70/15/15) so each split preserves the true class
     ratio.
   - `class_weight="balanced"` in the baseline Logistic Regression.
   - Monitoring **Recall** and **F1**, not just Accuracy, since a trivial
     always-predict-Normal classifier would already score deceptively high
     Accuracy on an imbalanced set.

2. **Sample Visualization** (`samples_grid.png`): Qualitatively, defective
   samples exhibit localized texture discontinuities (a scratch, blob, or
   band of contrast breaking the otherwise homogeneous background weave/
   pattern), while Normal samples exhibit consistent, near-stationary
   texture statistics across the whole image. This observation directly
   informs the modeling choice: local, translation-invariant filters (i.e.,
   convolutional feature extractors) should be well-suited to detecting
   spatially localized defects wherever they occur in the frame.

3. **Pixel Intensity Distribution** (`pixel_distribution.png`): The
   grayscale intensity histograms for Normal vs. Defect classes overlap
   substantially — global pixel statistics alone are **not** strongly
   discriminative. This is an important diagnostic: it tells us in advance
   that a baseline relying purely on global pixel/histogram summaries will
   have limited discriminative power, and that spatial/local pattern
   information (which deep CNNs are designed to exploit) will be necessary
   to substantially improve performance.

---

## 2. Methodology & Baseline

### 2.1 Non-Deep Baseline: Logistic Regression on Hand-Crafted Features

Per the EDA insight above, the baseline deliberately relies on
**global/coarse features** to establish a lower-bound reference:

- Downsampled (`32x32`) flattened grayscale pixel intensities.
- Global statistics: mean, std, min, max, median intensity.
- A 16-bin grayscale intensity histogram (density-normalized).

These features are standardized (`StandardScaler`) and fed into a
`scikit-learn` `LogisticRegression` classifier with `class_weight="balanced"`
to counter class imbalance. This baseline is intentionally simple and fast
to train (no GPU, no gradient descent over image tensors), serving as the
"is deep learning worth it?" sanity check required by rigorous applied ML
methodology.

### 2.2 Deep Learning Architecture: ResNet18 Transfer Learning

The primary model is a **ResNet18** convolutional backbone, pretrained on
ImageNet, with its original 1000-way classification head removed and
replaced by a custom head:

```
ResNet18 backbone (ImageNet-pretrained conv layers)
        │
Global Average Pooling (512-d feature vector)
        │
Linear(512 → 128) → ReLU → Dropout(p=0.3)
        │
Linear(128 → 2)   [Normal / Defect logits]
```

**Why ResNet18 / transfer learning?**
- Low-level convolutional filters (edges, textures, gradients) learned from
  ImageNet transfer well to industrial texture images, despite the domain
  gap, because these are largely domain-agnostic visual primitives.
- Residual connections mitigate vanishing gradients, allowing deeper
  effective feature extraction than a from-scratch shallow CNN could
  achieve with the relatively modest amount of labeled DAGM data.
- The full network is fine-tuned end-to-end (not frozen), since DAGM's
  texture domain differs enough from natural images that allowing the
  backbone to adapt improves performance over a frozen linear-probe setup.

### 2.3 Loss Function & Metrics

- **Loss:** `CrossEntropyLoss` over the 2-class logits — standard,
  well-calibrated choice for binary/multi-class classification with softmax
  outputs.
- **Metrics:** Accuracy, Precision, Recall, and F1-Score are computed on the
  held-out test set for both the baseline and the deep model, enabling a
  direct, apples-to-apples comparison (see Section 4 and the README results
  table).

---

## 3. Engineering Rigor & Optimization

### 3.1 Data Augmentation

Because industrial surface defects have **no canonical orientation** — a
scratch can occur at any angle, and the camera/part orientation is not
semantically meaningful — the training pipeline (`dataset.py`) applies:

- `RandomHorizontalFlip(p=0.5)`
- `RandomVerticalFlip(p=0.5)`
- `RandomRotation(degrees=15)`

This increases the effective diversity of the training set and discourages
the model from overfitting to incidental orientation artifacts, improving
generalization to the held-out test set.

### 3.2 Optimization Setup

- **Optimizer:** Adam (`lr=1e-4`, `weight_decay=1e-5`) — adaptive learning
  rates converge quickly on the relatively small DAGM training set while the
  small weight decay term acts as L2 regularization against overfitting.
- **LR Scheduler:** `ReduceLROnPlateau` (monitoring validation loss,
  `factor=0.5`, `patience=3`) — automatically halves the learning rate when
  validation loss plateaus, allowing finer convergence in later epochs
  without manual LR-schedule tuning.
- **Early Stopping:** Training halts if validation loss fails to improve for
  a configurable number of epochs (default `patience=7`), and the **best**
  checkpoint (lowest validation loss seen) is restored and saved to
  `best_model.pth` — this directly guards against overfitting to the
  training set in later epochs.
- **Dropout (p=0.3)** in the classification head provides additional
  regularization on the final decision boundary.

### 3.3 Regularization Summary

| Technique | Purpose |
|---|---|
| Random flips + rotation | Combat overfitting to orientation-specific artifacts |
| Weight decay (Adam) | L2 penalty discouraging large weights |
| Dropout (0.3) in head | Reduces co-adaptation of classifier-head neurons |
| Early stopping | Prevents over-training past the point of generalization |
| ReduceLROnPlateau | Fine-grained convergence, avoids overshooting minima |
| Stratified splitting | Ensures Val/Test distributions match true class balance |

---

## 4. Diagnostic Analysis & Insights

### 4.1 Confusion Matrix Evaluation

`evaluate.py` generates `confusion_matrix.png` on the held-out test set.
Diagnostically, the matrix should be inspected along two axes:

- **False Negatives (Defect → predicted Normal):** the operationally
  costliest error type — a defective part would pass inspection undetected.
  A high false-negative count would indicate the model needs either more
  defect-class training examples, stronger augmentation of rare defect
  sub-types, or a decision-threshold shift (trading some precision for
  recall) before industrial deployment.
- **False Positives (Normal → predicted Defect):** less costly but still
  operationally relevant, since excessive false alarms erode operator trust
  in the automated system and increase manual re-inspection workload.

### 4.2 Failure Analysis on Misclassified Images

`evaluate.py` additionally renders `misclassified_samples.png`: the top-K
test images the model got wrong, ranked by **prediction confidence**,
annotated with True vs. Predicted label. Reviewing the most-confidently-wrong
predictions is a standard diagnostic step because these cases most sharply
reveal a systematic model weakness rather than borderline/ambiguous noise.
Typical DAGM-specific failure modes to look for when inspecting this grid:

- **Subtle, low-contrast defects** that blend into background texture noise
  — these are the hardest true-Defect cases and the most likely source of
  false negatives.
- **Non-defect texture irregularities** (lighting artifacts, background
  vignetting) that superficially resemble a defect pattern — a likely
  source of false positives.

### 4.3 Industrial Impact

The comparison between the classical baseline and the ResNet18 transfer
learning model (see the metrics table in `README.md`) quantifies the value
of deep feature learning for this task: coarse global pixel/histogram
statistics are insufficient to reliably separate localized surface defects
from clean texture, whereas a CNN's hierarchical, spatially-aware feature
representations substantially close this gap. In a deployed setting, this
translates to:

- Fewer defective units reaching customers (higher Recall on Defect).
- Reduced manual re-inspection burden (higher Precision, fewer false
  alarms).
- A model architecture (ResNet18) that is lightweight enough for real-time
  inference on a production-line camera feed, unlike larger backbones that
  would introduce unacceptable latency.

### 4.4 Limitations & Future Work

- The current pipeline treats all ten DAGM texture classes as a single
  pooled binary problem; a natural extension is a per-texture-class model or
  a multi-task architecture conditioning on texture class, which could
  further improve Recall on visually subtle defect types.
- Future iterations could incorporate defect **localization** (bounding
  box/segmentation) using the DAGM ellipse/mask annotations available in the
  raw dataset, moving from "is there a defect?" to "where exactly is it?" —
  valuable for operator-facing inspection tools.
- A precision-recall threshold sweep (rather than the default 0.5 softmax
  cutoff) should be considered before deployment, to explicitly tune the
  operating point toward minimizing false negatives given the asymmetric
  cost of missed defects in a real production line.

---

*This report should be finalized with the actual numeric results and figures
generated by running `eda.py`, `baseline_model.py`, `train.py`, and
`evaluate.py` on your local copy of the DAGM 2007 dataset.*
