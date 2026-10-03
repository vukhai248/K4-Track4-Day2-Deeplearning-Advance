"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Đảm bảo import được code/ và eval.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from eval import compute_metrics, save_predictions
import dataset as ds
import model as md
import losses as ls


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet34"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (tối ưu cho 4GB VRAM) ---
    epochs: int = 12
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = md.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi Cosine Annealing."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average của trọng số mô hình."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def plot_curves(history: list[dict], save_path: Path, title: str):
    """Vẽ đường cong loss và macro-F1 theo epoch."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    epochs = df["epoch"]
    l1 = ax1.plot(epochs, df["train_loss"], "r-", label="Train Loss")
    l2 = ax1.plot(epochs, df["val_loss"], "r--", label="Val Loss")
    l3 = ax2.plot(epochs, df["val_macro_f1"], "b-", label="Val Macro-F1")
    if "val_acc" in df.columns:
        l4 = ax2.plot(epochs, df["val_acc"], "g--", label="Val Top-1 Acc")
    else:
        l4 = []

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="r")
    ax2.set_ylabel("Metric", color="b")
    ax1.tick_params(axis="y", labelcolor="r")
    ax2.tick_params(axis="y", labelcolor="b")

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run(cfg: Config) -> dict:
    """Hàm chạy toàn bộ pipeline huấn luyện cho một Config."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 1. Dữ liệu
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    train_tf = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = ds.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds.make_loader(
        train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = ds.make_loader(
        val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 2. Mô hình
    model = md.build_model(cfg.backbone, pretrained=True, num_classes=ds.NUM_CLASSES,
                           drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)

    # 3. Loss
    weights = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().to_dict()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        weights = ls.class_weights(counts, beta=beta).to(device)

    smoothing = cfg.label_smoothing if cfg.loss == "ls" else 0.0
    criterion = ls.build_criterion(cfg.loss, smoothing=smoothing, gamma=cfg.focal_gamma, weight=weights)

    # 4. Tối ưu hoá
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.amp)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    start_time = time.time()

    print(f"\n[{cfg.exp_id}] Bắt đầu train {cfg.backbone} | seed={cfg.seed} | batch={cfg.batch_size} | AMP={cfg.amp} | epochs={cfg.epochs}")

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        if cfg.init == "frozen":
            md.freeze_backbone(model)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0.0
        train_samples = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if cfg.mix in ("mixup", "cutmix"):
                x_mixed, targets = ls.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
                with autocast(enabled=cfg.amp):
                    logits = model(x_mixed)
                    loss = ls.mixed_loss(criterion, logits, targets)
            else:
                with autocast(enabled=cfg.amp):
                    logits = model(x)
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)

        train_loss /= max(1, train_samples)

        # Đánh giá trên VAL
        if ema is not None:
            ema.apply_shadow(model)

        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_preds = []
        val_targets = []
        val_probs = []
        val_filenames = []

        with torch.no_grad():
            for x, y, fns in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=cfg.amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                val_samples += x.size(0)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

                val_probs.append(probs)
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())
                val_filenames.extend(fns)

        if ema is not None:
            ema.restore(model)

        val_loss /= max(1, val_samples)
        metrics = compute_metrics(np.array(val_targets), np.array(val_preds), np.vstack(val_probs))
        val_macro_f1 = metrics["macro_f1"]
        val_acc = metrics["top1"]
        epoch_sec = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_acc": val_acc,
            "epoch_sec": epoch_sec,
        })

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            if ema is not None:
                ema.apply_shadow(model)
                best_state = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
                best_state = copy.deepcopy(model.state_dict())

            # Lưu dự đoán val của checkpoint tốt nhất
            save_predictions(
                pred_path(cfg, "val"),
                val_filenames,
                np.array(val_targets),
                np.vstack(val_probs)
            )

        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f} | {epoch_sec:.1f}s")

    # Lưu checkpoint tốt nhất
    torch.save(best_state, r_dir / "best_model.pth")
    pd.DataFrame(history).to_csv(r_dir / "history.csv", index=False)
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png", f"{cfg.exp_id} ({cfg.backbone}) - Best Val F1: {best_val_f1:.4f}")

    total_time = time.time() - start_time
    print(f"[{cfg.exp_id}] Hoàn tất! Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_f1:.4f} | Tổng thời gian: {total_time/60:.2f}m")

    # Đánh giá trên TEST (chỉ ở Bước 4 khi save_test_predictions = True)
    test_metrics = None
    if cfg.save_test_predictions:
        print(f"[{cfg.exp_id}] Chạy đánh giá trên tập TEST...")
        model.load_state_dict(best_state)
        model.eval()

        test_loader = ds.make_loader(
            test_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_probs = []
        test_preds = []
        test_targets = []
        test_filenames = []

        with torch.no_grad():
            for x, y, fns in test_loader:
                x = x.to(device)
                with autocast(enabled=cfg.amp):
                    logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                test_probs.append(probs)
                test_preds.extend(probs.argmax(axis=1))
                test_targets.extend(y.numpy())
                test_filenames.extend(fns)

        test_probs = np.vstack(test_probs)
        test_targets = np.array(test_targets)
        test_preds = np.array(test_preds)

        save_predictions(pred_path(cfg, "test"), test_filenames, test_targets, test_probs)
        test_metrics = compute_metrics(test_targets, test_preds, test_probs)
        print(f"[{cfg.exp_id}] TEST Macro-F1: {test_metrics['macro_f1']:.4f} | TEST Acc: {test_metrics['top1']:.4f}")

    return {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
        "total_sec": total_time,
    }
