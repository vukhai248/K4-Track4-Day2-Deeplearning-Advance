"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 50, sync=None) -> dict:
    """Đo thời gian hàm `fn()` (mili-giây) với warmup và đồng bộ GPU."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ forward của `model` với đầu vào chuẩn (batch_size, 3, img_size, img_size)."""
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_obj).eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)

    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None

    if dtype == "fp16":
        model = model.half()
        dummy = dummy.half()
        fn = lambda: model(dummy)
    elif dtype == "amp":
        fn = lambda: torch.cuda.amp.autocast(True)(lambda: model(dummy))()
    else:  # fp32
        fn = lambda: model(dummy)

    with torch.inference_mode():
        res = bench(fn, warmup=warmup, iters=iters, sync=sync)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    p50 = res["p50"]
    images_per_s = (batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(res["p50"], 2),
        "p95": round(res["p95"], 2),
        "p99": round(res["p99"], 2),
        "mean": round(res["mean"], 2),
        "images_per_s": round(images_per_s, 1),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1,
                img_size: int = 224, device: str = "cuda", **kw) -> dict:
    """Đo độ trễ cho K views TTA."""
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_obj).eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)
    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None

    def fn():
        for _ in range(k_views):
            model(dummy)

    with torch.inference_mode():
        res = bench(fn, warmup=kw.get("warmup", 10), iters=kw.get("iters", 50), sync=sync)

    return res
