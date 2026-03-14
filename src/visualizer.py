from __future__ import annotations

import os

import cv2
import numpy as np

from .depth_estimator import DepthResult
from .image_loader import ImageData
from .point_labeler import LabeledCloud
from .segmentor import SegResult


class Visualizer:
    LABEL_PALETTE = [
        (255,  80,  80),
        ( 80, 200,  80),
        ( 80, 120, 255),
        (255, 200,  50),
        (200,  80, 255),
    ]

    @staticmethod
    def save_depth(depth: DepthResult, path: str) -> None:
        """깊이맵 → JET 컬러맵 PNG"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = depth.depth.copy()
        valid = depth.mask
        vis = np.zeros((*d.shape, 3), dtype=np.uint8)
        if valid.any():
            d_valid = d[valid]
            d_norm = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min() + 1e-8)
            colored = cv2.applyColorMap(
                (d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
            )                                         # (N, 1, 3)
            vis[valid] = colored.squeeze(1)
        cv2.imwrite(path, vis)

    @staticmethod
    def save_normal(depth: DepthResult, path: str) -> None:
        """법선맵 → RGB PNG ([-1,1] → [0,255])"""
        if depth.normal is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        normal_vis = ((depth.normal * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

    @staticmethod
    def save_segmentation_overlay(
        image: ImageData, seg: SegResult, path: str
    ) -> None:
        """원본 이미지 위에 레이블별 컬러 마스크 오버레이"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        overlay = image.rgb.copy()
        for i, (mask, lbl) in enumerate(zip(seg.masks, seg.labels)):
            color = np.array(Visualizer.LABEL_PALETTE[i % len(Visualizer.LABEL_PALETTE)])
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.45
                + color.astype(np.float32) * 0.55
            ).astype(np.uint8)
            # 레이블 텍스트
            ys, xs = np.where(mask)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(overlay, lbl, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    @staticmethod
    def save_labeled_cloud_image(
        image: ImageData, cloud: LabeledCloud, path: str
    ) -> None:
        """픽셀별 레이블을 고유 색으로 표현한 이미지"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        labels = list(cloud.points.keys())
        color_map = {
            lbl: Visualizer.LABEL_PALETTE[i % len(Visualizer.LABEL_PALETTE)]
            for i, lbl in enumerate(labels)
        }
        vis = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        for lbl, color in color_map.items():
            mask = cloud.label_map == lbl
            vis[mask] = color
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
