# src/models/disc/timm_classifier.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for TimmClassifier. Please install timm: pip install timm"
    ) from e

from ..base import IDiscModel


class TimmClassifier(nn.Module, IDiscModel):
    """
    一个基于 timm 的分类模型封装，符合 IDiscModel 接口。
    默认使用 convnext_large，支持从 config 中覆盖。
    """

    def __init__(
        self,
        backbone: str = "convnext_large",
        num_classes: int = 1000,
        pretrained: bool = True,
        in_chans: int = 3,
        global_pool: str = "avg",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._num_classes = num_classes

        # 创建 timm 模型
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            global_pool=global_pool,
            **kwargs,
        )

    # -------- IDiscModel required --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 logits: (B, num_classes)
        """
        return self.model(x)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    # -------- factory style --------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TimmClassifier":
        """
        支持从 hydra / yaml 里直接构建。
        推荐的 yaml:
          model:
            name: timm_classifier
            backbone: convnext_large
            num_classes: 1000
            pretrained: true
        """
        # 给一些安全的默认值
        backbone = cfg.get("backbone", "convnext_large")
        num_classes = cfg.get("num_classes", 1000)
        pretrained = cfg.get("pretrained", True)
        in_chans = cfg.get("in_chans", 3)
        global_pool = cfg.get("global_pool", "avg")

        # 允许把多余参数原样透传给 timm
        extra_kwargs = {
            k: v
            for k, v in cfg.items()
            if k
            not in {
                "name",
                "backbone",
                "num_classes",
                "pretrained",
                "in_chans",
                "global_pool",
            }
        }

        return cls(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool=global_pool,
            **extra_kwargs,
        )
