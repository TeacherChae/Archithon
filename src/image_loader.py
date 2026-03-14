from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class ImageData:
    rgb: np.ndarray       # (H, W, 3) uint8
    tensor: torch.Tensor  # (3, H, W) float32 [0, 1]
    path: str
    height: int
    width: int


class ImageLoader:
    def load(self, path: str) -> ImageData:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._make(rgb, path)

    def load_from_array(self, rgb: np.ndarray, path: str = "") -> ImageData:
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb 배열은 uint8이어야 합니다. 현재: {rgb.dtype}")
        return self._make(rgb, path)

    def _make(self, rgb: np.ndarray, path: str) -> ImageData:
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return ImageData(rgb=rgb, tensor=tensor, path=path, height=h, width=w)
