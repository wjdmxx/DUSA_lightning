# src/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IDiscModel(ABC):
    """
    判别式模型接口：分类 / 分割都应该实现这个接口里至少两个方法：
    - forward: 标准前向
    - num_classes: 便于外面构建 head / metric
    分类任务默认 forward 返回 logits: (B, C)
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError



class IGenModel(ABC):
    """
    生成式模型 / flow / SiT 的抽象接口。
    这里不强行规定输入输出形式，因为不同flow的签名差别挺大。
    我们只要求有 forward 和一个 from_config 的构造器，便于统一构建。
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "IGenModel":
        """
        生成模型的统一构建入口，方便在 builder 里一行调用。
        你如果用 hydra，可以直接把 DictConfig 传进来。
        """
        return cls(**cfg)
