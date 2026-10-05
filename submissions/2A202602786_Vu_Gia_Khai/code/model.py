"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "mobilenetv3": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp qua timm.
    `init`:
      - 'scratch': pretrained=False, train toàn bộ
      - 'frozen': pretrained=True, đóng băng backbone, chỉ train head
      - 'finetune': pretrained=True, train toàn bộ
    """
    is_pretrained = pretrained if init != "scratch" else False
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head phân loại."""
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if hasattr(classifier, "parameters") else set()
    
    for param in model.parameters():
        if param in head_params:
            param.requires_grad = True
        else:
            param.requires_grad = False


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:
    - Backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - Norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - Head mới: lr = lr_head, weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if hasattr(classifier, "parameters") else set()

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in head_params:
            head_decay.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Ước lượng GMAC cho một ảnh 3 x img_size x img_size bằng hooks trên Conv2d và Linear."""
    macs = 0

    def conv_hook(self, input, output):
        nonlocal macs
        batch_size, _, out_h, out_w = output.shape
        in_c = self.in_channels
        out_c = self.out_channels
        k_h, k_w = self.kernel_size
        groups = self.groups
        macs += int(batch_size * out_h * out_w * (in_c // groups) * k_h * k_w * out_c)

    def linear_hook(self, input, output):
        nonlocal macs
        batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
        macs += int(batch_size * self.in_features * self.out_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_x = torch.zeros(1, 3, img_size, img_size, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(dummy_x)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return macs / 1e9
