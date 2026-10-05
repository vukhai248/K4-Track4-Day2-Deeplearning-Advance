"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: torch.device | str, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    device = torch.device(device)
    filenames = []
    y_true = []
    logits_list = []

    with torch.inference_mode():
        for x, y, fns in loader:
            x = x.to(device)
            if view is not None:
                x = view(x)
            logits = model(x)
            logits_list.append(logits.cpu().numpy())
            y_true.extend(y.numpy())
            filenames.extend(fns)

    return filenames, np.array(y_true), np.vstack(logits_list)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> list[torch.Tensor]:
    """5 crops: 4 góc và trung tâm."""
    _, _, h, w = x.shape
    crops = []
    # 4 góc
    crops.append(x[:, :, :crop, :crop])
    crops.append(x[:, :, :crop, w - crop:])
    crops.append(x[:, :, h - crop:, :crop])
    crops.append(x[:, :, h - crop:, w - crop:])
    # Giữa
    cy, cx = (h - crop) // 2, (w - crop) // 2
    crops.append(x[:, :, cy:cy + crop, cx:cx + crop])
    return crops


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    views = []
    for s in sizes:
        views.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False))
    return views


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một phân bố xác suất."""
    if space == "prob":
        probs = [np.exp(z - np.max(z, axis=1, keepdims=True)) for z in logits_per_view]
        probs = [p / np.sum(p, axis=1, keepdims=True) for p in probs]
        return np.mean(probs, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        e = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL bằng L-BFGS-B."""
    def nll_fn(t_arr):
        T = t_arr[0]
        scaled = val_logits / T
        log_sum_exp = np.log(np.sum(np.exp(scaled - np.max(scaled, axis=1, keepdims=True)), axis=1, keepdims=True)) + np.max(scaled, axis=1, keepdims=True)
        log_probs = scaled - log_sum_exp
        nll = -np.mean(log_probs[np.arange(len(val_labels)), val_labels])
        return nll

    res = minimize(nll_fn, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0])


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    scaled = logits / max(1e-4, T)
    e = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (áp dụng cho CNN như ResNet)."""
    model_copy = copy.deepcopy(model).eval()
    for name, module in model_copy.named_children():
        if len(list(module.children())) > 0:
            fuse_conv_bn(module)
        # Kiểm tra nếu có thể gộp bằng tiện ích PyTorch
        if hasattr(torch.nn.utils, "fuse_conv_bn_eval"):
            try:
                torch.nn.utils.fuse_conv_bn_eval(module)
            except Exception:
                pass
    return model_copy
