from __future__ import annotations

import json
import os
from dataclasses import dataclass

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

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        labels = list(self.points.keys())

        # ── .npz ─────────────────────────────────────────────
        arrays: dict[str, np.ndarray] = {
            "all_points": self.all_points,
            "all_colors": self.all_colors,
            "label_map": self.label_map.astype(str),
        }
        for lbl in labels:
            arrays[f"pts_{lbl}"] = self.points[lbl]
            arrays[f"col_{lbl}"] = self.colors[lbl]
            if self.normals[lbl] is not None:
                arrays[f"nrm_{lbl}"] = self.normals[lbl]
        np.savez_compressed(os.path.join(directory, "cloud.npz"), **arrays)

        # ── GeoJSON ───────────────────────────────────────────
        features = []
        for lbl in labels:
            pts = self.points[lbl]   # (N, 3) float32
            col = self.colors[lbl]   # (N, 3) uint8
            nrm = self.normals[lbl]  # (N, 3) | None
            bbox = None
            if len(pts) > 0:
                bbox = {
                    "x_range": [round(float(pts[:, 0].min()), 4), round(float(pts[:, 0].max()), 4)],
                    "y_range": [round(float(pts[:, 1].min()), 4), round(float(pts[:, 1].max()), 4)],
                    "z_range": [round(float(pts[:, 2].min()), 4), round(float(pts[:, 2].max()), 4)],
                }
            props = {
                "label": lbl,
                "point_count": len(pts),
                "bbox": bbox,
                # RGB 색상: coordinates와 동일 인덱스 대응
                "colors_rgb": col.tolist(),
            }
            if nrm is not None:
                props["normals_xyz"] = nrm.round(4).tolist()

            features.append({
                "type": "Feature",
                "id": lbl,
                "geometry": {
                    "type": "MultiPoint",
                    # GeoJSON Position: [x, y, z] (미터 단위 카메라 좌표계)
                    "coordinates": pts.round(4).tolist(),
                },
                "properties": props,
            })

        geojson = {
            "type": "FeatureCollection",
            "properties": {
                "source": "PointLabeler",
                "description": "SAM3 마스크 ∩ MoGe-2 유효 마스크 → 레이블별 3D 점군",
                "total_valid_points": len(self.all_points),
                "labels": labels,
                "point_counts": {lbl: len(self.points[lbl]) for lbl in labels},
            },
            "features": features,
        }
        with open(os.path.join(directory, "cloud.json"), "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"[LabeledCloud] 저장 → {directory}")

    @classmethod
    def load(cls, directory: str) -> LabeledCloud:
        data = np.load(os.path.join(directory, "cloud.npz"), allow_pickle=True)
        with open(os.path.join(directory, "cloud.json")) as f:
            geo = json.load(f)
        labels = geo["properties"]["labels"]
        points  = {lbl: data[f"pts_{lbl}"] for lbl in labels}
        colors  = {lbl: data[f"col_{lbl}"] for lbl in labels}
        normals = {
            lbl: (data[f"nrm_{lbl}"] if f"nrm_{lbl}" in data else None)
            for lbl in labels
        }
        return cls(
            points=points, colors=colors, normals=normals,
            all_points=data["all_points"], all_colors=data["all_colors"],
            label_map=data["label_map"],
        )


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
