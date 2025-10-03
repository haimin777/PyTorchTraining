# train_efficientnet_b0.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import math
import os
from dataclasses import dataclass
from typing import Literal, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class ImageDFDataset(Dataset):
    """
    Dataset for (path, label) rows in a pandas DataFrame.

    Required columns:
      - 'path': str, absolute or relative paths to image files
      - 'label': int or str (class labels)

    Args:
        df: DataFrame with columns ['path', 'label']
        transform: torchvision transforms
        label_to_index: mapping from label string -> index (shared across splits)
        verify_exists: filter out rows with missing files if True
        strict_open: if True, raise on image open errors; otherwise use a black fallback image
        fallback_size: (H, W) for fallback image if strict_open=False
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        label_to_index: Optional[dict] = None,
        verify_exists: bool = True,
        strict_open: bool = False,
        fallback_size: Tuple[int, int] = (224, 224),
    ):
        assert {'paths', 'label'}.issubset(df.columns), "DataFrame must have 'path' and 'label' columns"
        self.transform = transform
        self.strict_open = strict_open
        self.fallback_size = fallback_size

        df = df.copy()
        df['paths'] = df['paths'].astype(str).map(lambda p: str(Path(p)))

        if verify_exists:
            exists_mask = df['paths'].map(lambda p: Path(p).is_file())
            missing = int((~exists_mask).sum())
            if missing:
                print(f"[ImageDFDataset] Skipping {missing} rows with missing files.")
            df = df.loc[exists_mask].reset_index(drop=True)

        # If mapping not provided, infer from current df
        if label_to_index is None:
            classes = sorted(df['label'].astype(str).unique().tolist())
            self.label_to_index = {c: i for i, c in enumerate(classes)}
        else:
            self.label_to_index = label_to_index

        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.paths: List[str] = df['paths'].tolist()
        self.labels: List[int] = [self.label_to_index[str(l)] for l in df['label']]

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y = self.labels[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            if self.strict_open:
                raise
            # Fallback black image so rare corrupt files don't crash training
            h, w = self.fallback_size
            img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        x = self.transform(img)
        return x, y


def make_transforms(img_size: int = 224, aug: bool = True) -> Tuple[Callable, Callable]:
    train_tfms = [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip() if aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ]
    val_tfms = [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ]
    train_tfms = [t for t in train_tfms if not isinstance(t, transforms.Lambda)]
    return transforms.Compose(train_tfms), transforms.Compose(val_tfms)


def build_dataloaders_from_two_dfs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32,
    img_size: int = 224,
    aug: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    balance_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Build DataLoaders when train/val are already split into two DataFrames.

    Returns:
        train_loader, val_loader, label_to_index mapping
    """
    # Build a shared label mapping from union of labels across train+val
    all_labels = pd.concat([train_df['label'].astype(str), val_df['label'].astype(str)], ignore_index=True)
    classes = sorted(all_labels.unique().tolist())
    label_to_index = {c: i for i, c in enumerate(classes)}

    train_tfms, val_tfms = make_transforms(img_size=img_size, aug=aug)

    train_ds = ImageDFDataset(
        train_df,
        transform=train_tfms,
        label_to_index=label_to_index,
        verify_exists=True,
        strict_open=False,
        fallback_size=(img_size, img_size),
    )
    val_ds = ImageDFDataset(
        val_df,
        transform=val_tfms,
        label_to_index=label_to_index,
        verify_exists=True,
        strict_open=False,
        fallback_size=(img_size, img_size),
    )

    # Optional: class-balanced sampler on train set
    sampler = None
    if balance_sampler and len(train_ds) > 0:
        labels = np.array(train_ds.labels)
        class_sample_counts = np.bincount(labels, minlength=len(label_to_index))
        class_weights = 1.0 / np.maximum(class_sample_counts, 1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.float),
            num_samples=len(train_ds),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, label_to_index


@dataclass
class TrainCfg:
    num_classes: int
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: Literal["cosine", "cosine_warm_restarts", "none"] = "cosine"
    # Cosine (decay)
    cosine_eta_min: float = 1e-6
    cosine_step_per_batch: bool = True  # if False, step once per epoch
    # Cosine Warm Restarts (cyclic)
    cwr_T0: int = 5          # first cycle length (in epochs if step_per_batch=False; otherwise in "epoch-equivalents")
    cwr_Tmult: int = 2       # cycle length multiplier
    cwr_eta_min: float = 1e-6
    # Model / training niceties
    pretrained: bool = True
    freeze_backbone: bool = False
    label_smoothing: float = 0.0
    amp: bool = True                     # automatic mixed precision
    grad_clip_norm: float | None = 1.0   # set None to disable
    compile_model: bool = False          # torch.compile (PyTorch 2+)
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "effb0_best.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_tmpl: str = "effb0_e{epoch:03d}_valLoss={val_loss:.4f}_valAcc={val_acc:.4f}.pt"


def build_model(cfg: TrainCfg) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
    model = efficientnet_b0(weights=weights)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, cfg.num_classes)

    if cfg.freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad = False

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # requires PyTorch 2.x
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

