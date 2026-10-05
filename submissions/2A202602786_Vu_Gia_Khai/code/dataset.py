"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Cung cấp đầy đủ hàm đọc dữ liệu, kiểm tra tính toàn vẹn (S1-S6), augmentation và DataLoader.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_dir(path_str: str | Path, fallbacks: list[str]) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    for fb in fallbacks:
        p_fb = Path(fb)
        if p_fb.exists():
            return p_fb
    return p


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    """
    labels_path = _resolve_dir(
        labels_dir,
        [
            str(Path("..") / labels_dir),
            str(Path(__file__).resolve().parent.parent / labels_dir),
            str(Path(__file__).resolve().parent.parent / "data" / "labels"),
        ]
    )
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy đủ file split fold {fold} tại {labels_path}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu."""
    img_path = _resolve_dir(
        images_dir,
        [
            str(Path("..") / images_dir),
            str(Path(__file__).resolve().parent.parent / images_dir),
            str(Path(__file__).resolve().parent.parent / "images"),
            str(Path(__file__).resolve().parent.parent / "data" / "images"),
        ]
    )
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    # 1. Kiểm tra giao rỗng
    inter_tv = train_files & val_files
    inter_tt = train_files & test_files
    inter_vt = val_files & test_files
    assert len(inter_tv) == 0, f"Lỗi rò rỉ: train và val có {len(inter_tv)} ảnh trùng nhau!"
    assert len(inter_tt) == 0, f"Lỗi rò rỉ: train và test có {len(inter_tt)} ảnh trùng nhau!"
    assert len(inter_vt) == 0, f"Lỗi rò rỉ: val và test có {len(inter_vt)} ảnh trùng nhau!"

    # 2. Kiểm tra tổng số ảnh hợp đủ 17509
    union_all = train_files | val_files | test_files
    assert len(union_all) == 17509, f"Lỗi: Tổng số ảnh không phải 17.509 mà là {len(union_all)}"

    # 3. Kiểm tra mọi file tồn tại
    missing_files = []
    for fn in union_all:
        if not (img_path / fn).exists():
            missing_files.append(fn)
            if len(missing_files) >= 5:
                break
    assert len(missing_files) == 0, f"Lỗi: Không tìm thấy các file ảnh trong {img_path}, ví dụ: {missing_files}"

    # 4. Thống kê
    stats = {
        "n": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(union_all),
        },
        "ratio": {
            "train": len(train_df) / len(union_all),
            "val": len(val_df) / len(union_all),
            "test": len(test_df) / len(union_all),
        },
        "per_class": {
            "train": train_df["Label"].value_counts().to_dict(),
            "val": val_df["Label"].value_counts().to_dict(),
            "test": test_df["Label"].value_counts().to_dict(),
        },
        "overlap": {
            "train_val": len(inter_tv),
            "train_test": len(inter_tt),
            "val_test": len(inter_vt),
        }
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> T.Compose:
    """Tạo transform theo `train` và `aug`."""
    if train:
        if aug == "basic":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "color":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "trivial":
            return T.Compose([
                T.TrivialAugmentWide(),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "randaug":
            return T.Compose([
                T.RandAugment(num_ops=2, magnitude=9),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            raise ValueError(f"Augmentation không hợp lệ: {aug}")
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = _resolve_dir(
            images_dir,
            [
                str(Path("..") / images_dir),
                str(Path(__file__).resolve().parent.parent / images_dir),
                str(Path(__file__).resolve().parent.parent / "images"),
                str(Path(__file__).resolve().parent.parent / "data" / "images"),
            ]
        )
        self.transform = transform
        self.filenames = self.df["Filename"].values
        self.labels = self.df["Label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        fn = self.filenames[i]
        img_path = self.images_dir / fn
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), str(fn)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform)
    if train and sampler == "balanced":
        class_counts = df["Label"].value_counts().to_dict()
        weights = [1.0 / class_counts[y] for y in df["Label"]]
        sampler_obj = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
