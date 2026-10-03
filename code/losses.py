"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: 'ce', 'ls' (label smoothing), 'focal', 'ce_weighted'."""
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    elif kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1), weight=kw.get("weight"))
    elif kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    elif kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    else:
        raise ValueError(f"Loại loss không hỗ trợ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).
    Khi gamma = 0, hàm tương đương CrossEntropy hoàn toàn.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[target]
            loss = alpha_t * loss
        return loss.mean()


def class_weights(counts: dict | list | np.ndarray, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.
    - beta = 0: trọng số nghịch đảo (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al.)
    """
    if isinstance(counts, dict):
        ordered_counts = [counts[k] for k in sorted(counts.keys())]
        counts_arr = np.array(ordered_counts, dtype=np.float32)
    else:
        counts_arr = np.array(counts, dtype=np.float32)

    if beta <= 0.0:
        w = 1.0 / np.maximum(counts_arr, 1.0)
        w = w / np.mean(w)
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        w = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        w = w / np.sum(w) * len(counts_arr)

    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn theo Mixup hoặc CutMix."""
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[index]
    elif mode == "cutmix":
        _, _, h, w = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    else:
        raise ValueError(f"Chế độ mix không hỗ trợ: {mode}")

    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * loss(y_a) + (1 - lam) * loss(y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
