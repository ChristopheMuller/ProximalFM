# models/base.py
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseCATEModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, a, w, z, y):
        pass

    @abstractmethod
    def estimate_CATE(self, x, a_train, w, z, y_train, **kwargs):
        pass
