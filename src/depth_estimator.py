from dataclasses import dataclass

import numpy as np
import torch

from .image_loader import ImageData


@dataclass
class DepthResult:
    points: np.ndarray              # (H, W, 3) float32 — 픽셀별 3D 좌표 (미터)
    depth: np.ndarray               # (H, W) float32
    normal: np.ndarray | None       # (H, W, 3) float32 — 없으면 None
    mask: np.ndarray                # (H, W) bool — 유효 픽셀
    intrinsics: np.ndarray          # (3, 3) float32


class DepthEstimator:
    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vitl-normal",
        device: str = "cuda",
        fp16: bool = False,
        resolution_level: int = 9,
    ):
        self.device = device
        self.fp16 = fp16
        self.resolution_level = resolution_level

        from moge.model import import_model_class_by_version
        cls = import_model_class_by_version("v2")
        self.model = cls.from_pretrained(model_name).to(device)
        self.model.eval()

    def infer(self, image: ImageData) -> DepthResult:
        tensor = image.tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.inference_mode():
            output = self.model.infer(
                tensor,
                resolution_level=self.resolution_level,
                use_fp16=self.fp16,
            )

        points = output["points"].squeeze(0).cpu().numpy()          # (H, W, 3)
        depth = output["depth"].squeeze(0).cpu().numpy()            # (H, W)
        mask = output["mask"].squeeze(0).cpu().numpy().astype(bool) # (H, W)
        intrinsics = output["intrinsics"].squeeze(0).cpu().numpy()  # (3, 3)

        normal = None
        if "normal" in output:
            normal = output["normal"].squeeze(0).cpu().numpy()      # (H, W, 3)

        return DepthResult(
            points=points,
            depth=depth,
            normal=normal,
            mask=mask,
            intrinsics=intrinsics,
        )
