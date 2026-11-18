"""
dataset.py
==========
Custom PyTorch Dataset / DataLoader utilities for the DAGM 2007 Industrial
Surface Defect Detection dataset (binary classification: 0 = Normal, 1 = Defect).

This module is deliberately layout-agnostic: it supports both the *raw*
official DAGM 2007 folder layout and a *flattened* binary layout, which is
how most re-hosted versions of the dataset (e.g. on Kaggle) are organised.

Supported layouts
------------------
1) Raw DAGM layout (official release):

    data_root/
        Class1/
            Train/
                0001.PNG
                0002.PNG
                ...
                Label/
                    Labels.txt          # columns: ImageId, Label, ...
            Test/
                0001.PNG
                ...
                Label/
                    Labels.txt
        Class2/
        ...
        Class10/

   In this layout, `Labels.txt` has one row per image with a binary flag
   (1 = defective, 0 = ok) plus, for defective images, an ellipse/segmentation
   description that we do not need for classification.

2) Flattened binary layout (common on Kaggle re-uploads):

    data_root/
        Normal/    (or "OK", "0", "NoDefect")
            img_0001.png
            ...
        Defect/    (or "NG", "1", "Defect")
            img_0001.png
            ...

The `build_dataframe()` function auto-detects which layout is present and
returns a single `pandas.DataFrame` with columns: ["filepath", "label", "class_name"].
Everything downstream (EDA, baseline, training, evaluation) consumes that
DataFrame, so the rest of the pipeline never has to worry about the raw
folder structure.
"""

import os
import glob
import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

# ----------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42

NORMAL_DIR_NAMES = {"normal", "ok", "0", "nodefect", "no_defect", "good"}
DEFECT_DIR_NAMES = {"defect", "ng", "1", "defective", "bad", "faulty"}

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG")


# ----------------------------------------------------------------------------
# Dataframe construction (layout auto-detection)
# ----------------------------------------------------------------------------
def _find_labels_txt(label_dir: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(label_dir, "*abels*.txt"))
    return candidates[0] if candidates else None


def _parse_raw_dagm_class(class_dir: str, class_name: str) -> list:
    """Parse a single Class* directory of the official raw DAGM layout."""
    rows = []
    for split_dir_name in ("Train", "Test"):
        split_dir = os.path.join(class_dir, split_dir_name)
        if not os.path.isdir(split_dir):
            continue

        label_dir = os.path.join(split_dir, "Label")
        labels_txt = _find_labels_txt(label_dir) if os.path.isdir(label_dir) else None

        label_map = {}
        if labels_txt is not None:
            with open(labels_txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    # Official format: ImageId  Label  [EllipseParams...]
                    img_id, label = parts[0], parts[1]
                    try:
                        label_map[img_id] = int(label)
                    except ValueError:
                        continue

        for img_path in glob.glob(os.path.join(split_dir, "*")):
            if not img_path.lower().endswith(VALID_EXTS):
                continue
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            # DAGM label files sometimes key by zero-padded numeric id
            label = label_map.get(img_id, label_map.get(img_id.lstrip("0") or "0", 0))
            rows.append({"filepath": img_path, "label": int(label), "class_name": class_name})
    return rows


def _parse_flattened_binary(data_root: str) -> list:
    """Parse the flattened Normal/Defect binary folder layout."""
    rows = []
    for entry in os.listdir(data_root):
        entry_path = os.path.join(data_root, entry)
        if not os.path.isdir(entry_path):
            continue
        key = entry.strip().lower()
        if key in NORMAL_DIR_NAMES:
            label = 0
        elif key in DEFECT_DIR_NAMES:
            label = 1
        else:
            continue
        for img_path in glob.glob(os.path.join(entry_path, "**", "*"), recursive=True):
            if img_path.lower().endswith(VALID_EXTS):
                rows.append({"filepath": img_path, "label": label, "class_name": entry})
    return rows


def build_dataframe(data_root: str) -> pd.DataFrame:
    """
    Auto-detect the dataset layout under `data_root` and return a unified
    DataFrame with columns: filepath, label, class_name.
    """
    rows = []

    # Attempt 1: flattened binary layout (Normal / Defect folders directly under root)
    top_level = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    top_level_lower = {d.lower() for d in top_level}
    if top_level_lower & (NORMAL_DIR_NAMES | DEFECT_DIR_NAMES):
        rows = _parse_flattened_binary(data_root)

    # Attempt 2: raw DAGM Class1..Class10 layout
    if not rows:
        class_dirs = sorted(glob.glob(os.path.join(data_root, "Class*")))
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            rows.extend(_parse_raw_dagm_class(class_dir, class_name))

    if not rows:
        raise FileNotFoundError(
            f"Could not detect a valid DAGM 2007 layout under '{data_root}'. "
            "Expected either Normal/Defect subfolders or Class1..Class10 subfolders."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset="filepath").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Train / Val / Test split (70 / 15 / 15, stratified on label)
# ----------------------------------------------------------------------------
def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "Fractions must sum to 1.0"

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df["label"], random_state=seed
    )
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val_frac, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
def get_transforms(train: bool = True) -> T.Compose:
    """Return torchvision transform pipeline.

    Training pipeline includes augmentations (random horizontal/vertical flip,
    random rotation) suited to industrial texture images, where defects have
    no canonical orientation. Validation/Test pipelines only resize + normalize.
    """
    if train:
        return T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ----------------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------------
class DAGMDataset(Dataset):
    """PyTorch Dataset wrapping a DataFrame of [filepath, label, class_name]."""

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[T.Compose] = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


# ----------------------------------------------------------------------------
# Convenience factory: build dataloaders directly from a data root
# ----------------------------------------------------------------------------
def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = SEED,
):
    """
    Build train/val/test DataLoaders directly from a DAGM 2007 data root.

    Returns
    -------
    (train_loader, val_loader, test_loader, train_df, val_df, test_df)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = build_dataframe(data_root)
    train_df, val_df, test_df = split_dataframe(df, seed=seed)

    train_ds = DAGMDataset(train_df, transform=get_transforms(train=True))
    val_ds = DAGMDataset(val_df, transform=get_transforms(train=False))
    test_ds = DAGMDataset(test_df, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick sanity check of the DAGM dataset pipeline.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to DAGM 2007 dataset root.")
    args = parser.parse_args()

    df = build_dataframe(args.data_root)
    print(f"Total images found: {len(df)}")
    print(df["label"].value_counts())

    train_df, val_df, test_df = split_dataframe(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
