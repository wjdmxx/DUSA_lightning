# src/models/gen/sit_wrapper.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base import IGenModel


class SiTWrapper(nn.Module, IGenModel):
    """
    SiT / flow-matching 生成模型的外层包装。
    真正的 SiT 模型你可以在这里面加载：
      - 从你的仓库 import
      - 从 ckpt 里 load_state_dict
    这里先给一个壳子，保证工程跑得起来。
    """

    def __init__(
        self,
        model_name: str = "SiT-XL/2",
        ckpt: Optional[str] = None,
        image_size: int = 256,
        num_classes: int = 1000,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.ckpt = ckpt
        self.image_size = image_size
        self.num_classes = num_classes
        self.extra_cfg = kwargs

        # 这里留给你：你可以在这里真正构建你的 SiT
        # self.model = build_sit_model(...)
        self.model = None  # placeholder

        if device is not None:
            self.to(device)

    def forward(self, *args, **kwargs) -> Any:
        if self.model is None:
            raise NotImplementedError(
                "SiTWrapper.forward is called but inner SiT model is not initialized. "
                "Please load your SiT model in SiTWrapper.__init__."
            )
        return self.model(*args, **kwargs)

    # ------- factory style -------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SiTWrapper":
        """
        cfg 示例：
          model:
            name: sit
            model_name: SiT-XL/2
            ckpt: classification/pretrained_models/sit_xl_2.pt
            image_size: 256
            num_classes: 1000
        """
        return cls(**cfg)
