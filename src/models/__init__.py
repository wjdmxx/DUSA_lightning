# src/models/__init__.py
from __future__ import annotations

from typing import Any, Dict

from .base import IDiscModel, IGenModel
from .disc.timm_classifier import TimmClassifier
from .gen.sit_wrapper import SiTWrapper


def build_disc_model(cfg: Dict[str, Any]) -> IDiscModel:
    """
    统一的判别模型构建函数。
    约定 cfg 里至少有一个 'name' 字段来区分不同的实现。
    例如：
      model:
        name: timm_classifier
        backbone: convnext_large
        num_classes: 1000
        pretrained: true
    """
    name = cfg.get("name", "timm_classifier")

    if name == "timm_classifier":
        return TimmClassifier.from_config(cfg)
    else:
        raise ValueError(f"Unknown disc model name: {name}")


def build_gen_model(cfg: Dict[str, Any]) -> IGenModel:
    """
    统一的生成模型构建函数。
    目前先只接 SiT，具体实现你自己在 SiTWrapper 里填。
    cfg 示例：
      model:
        name: sit
        ckpt: ...
        image_size: 256
        num_classes: 1000
    """
    name = cfg.get("name", "sit")

    if name == "sit":
        return SiTWrapper.from_config(cfg)
    else:
        raise ValueError(f"Unknown gen model name: {name}")


__all__ = [
    "IDiscModel",
    "IGenModel",
    "build_disc_model",
    "build_gen_model",
    "TimmClassifier",
    "SiTWrapper",
]
