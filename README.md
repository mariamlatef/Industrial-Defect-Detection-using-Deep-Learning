# Industrial Defect Detection on DAGM 2007

Binary classification of industrial surface images (**Normal** vs. **Defect**)
using the [DAGM 2007 dataset](https://www.kaggle.com/datasets/bassam165/dagm-2007-industrial-defect-detection-dataset),
comparing a classical machine-learning baseline against a ResNet18
transfer-learning model.

---

## 1. Project Overview & Motivation

Automated visual inspection is a core problem in manufacturing quality
control. Manual inspection of surfaces (fabrics, metal sheets, etc.) is
slow, expensive, and prone to human fatigue-driven error. The **DAGM 2007**
dataset simulates this setting with artificially generated but statistically
realistic images of textured surfaces, each labeled as defect-free
(**Normal**, label `0`) or containing a surface flaw (**Defect**, label `1`).

The goal of this project is to build an end-to-end pipeline that:

1. Loads and explores the dataset (class balance, sample appearance, pixel
   statistics).
2. Establishes a simple, interpretable **non-deep baseline** (hand-crafted
   pixel/histogram features + Logistic Regression).
3. Trains a **deep transfer-learning model** (pre-trained ResNet18 + custom
   classification head) with proper regularization (augmentation, dropout,
   weight decay, early stopping, LR scheduling).
4. Performs **diagnostic analysis** — confusion matrix and failure-mode
   inspection — to understand *where* and *why* the model fails, which is
   critical before deploying any inspection system on a real production line.

A false negative (a real defect classified as Normal) is the costlier error
in this industrial setting, since it means a defective part reaches the
customer — this motivates paying close attention to **Recall** on the
Defect class, not just overall Accuracy.

---

## 2. Repository Structure

```
dagm_defect_detection/
├── dataset.py              # Dataset/DataLoader, transforms, train/val/test split
├── eda.py                  # Exploratory Data Analysis plots
├── baseline_model.py       # Classical ML baseline (Logistic Regression)
├── model.py                # ResNet18 transfer-learning model definition
├── train.py                # Training loop (Adam, early stopping, LR scheduler)
├── evaluate.py              # Test-set evaluation, confusion matrix, error analysis
├── requirements.txt
├── README.md
└── Project_Report_Draft.md
```

## 3. Dataset Layout

`dataset.py` auto-detects **either** of the following folder layouts under
`--data_root`:

**A. Official raw DAGM 2007 layout:**
```
data_root/
    Class1/
        Train/
            0001.PNG ...
            Label/Labels.txt
        Test/
            0001.PNG ...
            Label/Labels.txt
    Class2/ ... Class10/
```

**B. Flattened binary layout** (common on Kaggle re-uploads):
```
data_root/
    Normal/   (or OK / NoDefect)
        img_0001.png ...
    Defect/   (or NG / Defective)
        img_0001.png ...
```

Download the dataset from Kaggle and point `--data_root` at whichever
top-level folder matches one of the layouts above:
<https://www.kaggle.com/datasets/bassam165/dagm-2007-industrial-defect-detection-dataset>

---

## 4. Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 5. Step-by-Step Execution

### Step 1 — Exploratory Data Analysis
```bash
python eda.py --data_root /path/to/DAGM2007 --output_dir outputs/eda
```
Produces `class_distribution.png`, `samples_grid.png`, `pixel_distribution.png`.

### Step 2 — Train the Classical Baseline
```bash
python baseline_model.py --data_root /path/to/DAGM2007
```
Prints Accuracy / Precision / Recall / F1 for the Logistic Regression baseline.

### Step 3 — Train the Deep Learning Model (ResNet18)
```bash
python train.py --data_root /path/to/DAGM2007 --epochs 30 --batch_size 32 --lr 1e-4
```
Saves:
- `best_model.pth` — best checkpoint (lowest validation loss)
- `outputs/train/training_curves.png` — loss & accuracy curves

### Step 4 — Evaluate on the Test Set
```bash
python evaluate.py --data_root /path/to/DAGM2007 --checkpoint_path best_model.pth
```
Saves:
- `outputs/evaluate/confusion_matrix.png`
- `outputs/evaluate/misclassified_samples.png`
- `outputs/evaluate/test_metrics.txt`

---

## 6. Metrics Comparison: Baseline vs. ResNet18

> **Note:** The exact numbers below are placeholders illustrating the
> expected format — the real values populate automatically after running
> `baseline_model.py` and `evaluate.py` on your local copy of the dataset,
> and should be copied into this table (and into `Project_Report_Draft.md`)
> for your final submission.

| Metric      | Baseline (Logistic Regression) | ResNet18 (Transfer Learning) |
|-------------|:-------------------------------:|:-----------------------------:|
| Accuracy    | ~0.78                           | ~0.97                         |
| Precision   | ~0.74                           | ~0.96                         |
| Recall      | ~0.70                           | ~0.95                         |
| F1-Score    | ~0.72                           | ~0.95                         |

**Interpretation:** The hand-crafted-feature baseline struggles to separate
subtle, spatially-localized texture anomalies from background noise since it
only relies on coarse pixel/histogram statistics. The ResNet18 transfer
learning model, thanks to hierarchical convolutional features learned from
ImageNet and fine-tuned on DAGM's textures, captures the local defect
patterns far more reliably — the expected practical justification for
choosing a deep model in a real industrial deployment.

---

## 7. Reproducibility Notes

- Global seed fixed at `42` across dataset split, model init, and training.
- Train/Val/Test split: 70% / 15% / 15%, stratified by label.
- All figures and checkpoints are written to disk under `outputs/` and the
  repository root, so a full run is fully auditable.

## 8. License & Attribution

Dataset: DAGM 2007 (Deutsche Arbeitsgemeinschaft für Mustererkennung),
re-hosted on Kaggle:
<https://www.kaggle.com/datasets/bassam165/dagm-2007-industrial-defect-detection-dataset>.
Please refer to the dataset's original license terms before commercial use.
