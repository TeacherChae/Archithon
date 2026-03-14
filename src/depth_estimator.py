from __future__ import annotations

import json
import os
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

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        # ── .npz: 빠른 로딩용 바이너리 ──────────────────────
        arrays = dict(points=self.points, depth=self.depth,
                      mask=self.mask, intrinsics=self.intrinsics)
        if self.normal is not None:
            arrays["normal"] = self.normal
        np.savez_compressed(os.path.join(directory, "depth_result.npz"), **arrays)

        # ── GeoJSON ───────────────────────────────────────────
        valid = self.mask.astype(bool)
        d = self.depth[valid]
        p = self.points[valid]   # (N, 3)  [x, y, z]

        geojson = {
            "type": "FeatureCollection",
            "properties": {                      # 씬 전체 메타데이터
                "source": "MoGe-2",
                "model": "Ruicheng/moge-2-vitl-normal",
                "image_shape": {
                    "H": int(self.mask.shape[0]),
                    "W": int(self.mask.shape[1]),
                },
                "has_normal": self.normal is not None,
                "intrinsics": self.intrinsics.tolist(),   # 3×3 카메라 내부 파라미터
                "mask": {
                    "valid_pixels": int(valid.sum()),
                    "total_pixels": int(valid.size),
                    "coverage_pct": round(float(valid.mean()) * 100, 2),
                },
                "depth_stats": {
                    "min_m": round(float(d.min()), 4),
                    "max_m": round(float(d.max()), 4),
                    "mean_m": round(float(d.mean()), 4),
                    "std_m": round(float(d.std()), 4),
                },
            },
            "features": [
                {
                    "type": "Feature",
                    "id": "scene_point_cloud",
                    "geometry": {
                        "type": "MultiPoint",
                        # GeoJSON Position: [x, y, z] (미터 단위 카메라 좌표계)
                        "coordinates": p[:10].round(4).tolist(),  # 샘플 10개 (전체는 .npz)
                    },
                    "properties": {
                        "description": "MoGe-2 출력 포인트맵의 유효 픽셀 3D 좌표 (샘플 10개, 전체는 depth_result.npz)",
                        "total_points": len(p),
                        "bbox": {
                            "x_range": [round(float(p[:, 0].min()), 4), round(float(p[:, 0].max()), 4)],
                            "y_range": [round(float(p[:, 1].min()), 4), round(float(p[:, 1].max()), 4)],
                            "z_range": [round(float(p[:, 2].min()), 4), round(float(p[:, 2].max()), 4)],
                        },
                        **({"normal_sample_10": self.normal[valid][:10].round(4).tolist()}
                           if self.normal is not None else {}),
                    },
                }
            ],
        }
        with open(os.path.join(directory, "depth_result.json"), "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"[DepthResult] 저장 → {directory}")

    @classmethod
    def load(cls, directory: str) -> DepthResult:
        data = np.load(os.path.join(directory, "depth_result.npz"))
        with open(os.path.join(directory, "depth_result.json")) as f:
            geo = json.load(f)
        has_normal = geo["properties"]["has_normal"]
        return cls(
            points=data["points"],
            depth=data["depth"],
            normal=data["normal"] if has_normal else None,
            mask=data["mask"].astype(bool),
            intrinsics=data["intrinsics"],
        )


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
