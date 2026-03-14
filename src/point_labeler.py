from dataclasses import dataclass, field

import numpy as np

from .image_loader import ImageData
from .depth_estimator import DepthResult
from .segmentor import SegResult


@dataclass
class LabeledCloud:
    # 레이블별 3D 데이터
    points: dict[str, np.ndarray]           # label → (N, 3) float32
    colors: dict[str, np.ndarray]           # label → (N, 3) uint8
    normals: dict[str, np.ndarray | None]   # label → (N, 3) float32 | None

    # 전체 (레이블 무관, MoGe 유효 마스크 기준)
    all_points: np.ndarray                  # (M, 3)
    all_colors: np.ndarray                  # (M, 3)
    label_map: np.ndarray                   # (H, W) 문자열 — 픽셀별 레이블 ("" = 미지정)


class PointLabeler:
    def label(
        self,
        image: ImageData,
        depth: DepthResult,
        seg: SegResult,
    ) -> LabeledCloud:
        H, W = image.height, image.width

        points_per_label: dict[str, np.ndarray] = {}
        colors_per_label: dict[str, np.ndarray] = {}
        normals_per_label: dict[str, np.ndarray | None] = {}
        label_map = np.full((H, W), "", dtype=object)

        for mask, label in zip(seg.masks, seg.labels):
            # SAM 마스크 ∩ MoGe 유효 마스크
            combined = mask & depth.mask                        # (H, W) bool

            points_per_label[label] = depth.points[combined]   # (N, 3)
            colors_per_label[label] = image.rgb[combined]      # (N, 3)

            if depth.normal is not None:
                normals_per_label[label] = depth.normal[combined]  # (N, 3)
            else:
                normals_per_label[label] = None

            label_map[combined] = label

        # 전체 유효 점군
        all_pts = depth.points[depth.mask]   # (M, 3)
        all_col = image.rgb[depth.mask]      # (M, 3)

        return LabeledCloud(
            points=points_per_label,
            colors=colors_per_label,
            normals=normals_per_label,
            all_points=all_pts,
            all_colors=all_col,
            label_map=label_map,
        )
