from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .image_loader import ImageData


@dataclass
class Prompt:
    label: str
    points: list[tuple[int, int]] | None = None      # [(x, y), ...]
    point_labels: list[int] | None = None             # 1=전경, 0=배경
    box: tuple[int, int, int, int] | None = None      # (x1, y1, x2, y2)


@dataclass
class SegResult:
    masks: list[np.ndarray]    # List of (H, W) bool
    labels: list[str]          # 각 마스크의 레이블 이름
    scores: list[float]        # 예측 신뢰도

    def save(self, directory: str, image_rgb: np.ndarray | None = None) -> None:
        """
        directory  : 출력 디렉토리
        image_rgb  : (H, W, 3) uint8 원본 이미지 — 제공 시 레이블별 RGBA PNG도 저장
        """
        os.makedirs(directory, exist_ok=True)
        # ── .npz ─────────────────────────────────────────────
        np.savez_compressed(
            os.path.join(directory, "seg_result.npz"),
            **{f"mask_{i}": m for i, m in enumerate(self.masks)},
        )
        # ── RGBA 마스크 이미지 ────────────────────────────────
        if image_rgb is not None:
            H_img, W_img = image_rgb.shape[:2]
            # 컬러 오버레이 (모든 레이블을 하나의 이미지에)
            overlay = image_rgb.copy()
            palette = [
                (255, 80,  80),   # red
                (80,  200, 80),   # green
                (80,  120, 255),  # blue
                (255, 200, 50),   # yellow
                (200, 80,  255),  # purple
            ]
            for i, (mask, lbl) in enumerate(zip(self.masks, self.labels)):
                color = palette[i % len(palette)]
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * 0.45
                    + np.array(color, dtype=np.float32) * 0.55
                ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(directory, "overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            # 레이블별 RGBA (배경 투명)
            for mask, lbl in zip(self.masks, self.labels):
                alpha = mask.astype(np.uint8) * 255           # (H, W)
                rgba = np.dstack([image_rgb, alpha])           # (H, W, 4)
                cv2.imwrite(
                    os.path.join(directory, f"{lbl}_masked.png"),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                )
        # ── GeoJSON ───────────────────────────────────────────
        H, W = (self.masks[0].shape if self.masks else (0, 0))
        features = []
        for i, (lbl, mask, score) in enumerate(zip(self.labels, self.masks, self.scores)):
            ys, xs = np.where(mask)   # True 픽셀의 행(y)·열(x)
            bbox = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    if len(xs) else None)
            features.append({
                "type": "Feature",
                "id": lbl,
                "geometry": {
                    "type": "MultiPoint",
                    # GeoJSON Position: [x, y] = [열(col), 행(row)] — 이미지 픽셀 좌표
                    "coordinates": np.stack([xs, ys], axis=1).tolist(),
                },
                "properties": {
                    "label": lbl,
                    "sam3_score": round(score, 6),
                    "pixel_count": int(mask.sum()),
                    "coverage_pct": round(float(mask.mean()) * 100, 2),
                    # [x_min, y_min, x_max, y_max] 픽셀 좌표
                    "bbox_xyxy": bbox,
                },
            })

        geojson = {
            "type": "FeatureCollection",
            "properties": {
                "source": "SAM3",
                "model": "facebook/sam3",
                "image_shape": {"H": int(H), "W": int(W)},
                "num_labels": len(self.labels),
                "labels": self.labels,
            },
            "features": features,
        }
        with open(os.path.join(directory, "seg_result.json"), "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"[SegResult] 저장 → {directory}")

    @classmethod
    def load(cls, directory: str) -> SegResult:
        data = np.load(os.path.join(directory, "seg_result.npz"))
        with open(os.path.join(directory, "seg_result.json")) as f:
            geo = json.load(f)
        features = geo["features"]
        labels = [f["properties"]["label"] for f in features]
        scores = [f["properties"]["sam3_score"] for f in features]
        masks  = [data[f"mask_{i}"].astype(bool) for i in range(len(features))]
        return cls(masks=masks, labels=labels, scores=scores)


class BaseSegmentor(ABC):
    @abstractmethod
    def segment(self, image: ImageData, prompts: list[Prompt]) -> SegResult: ...


class SAMSegmentor(BaseSegmentor):
    """
    SAM3 기반 세그멘테이션.

    추론 경로:
      1. SAM2Transforms(1008) 로 이미지 전처리
      2. model.backbone.forward_image() 로 피처 추출
      3. model.predict_inst(inference_state, ...) 로 point/box 프롬프트 마스크 예측
    """

    IMAGE_SIZE = 1008

    def __init__(self, device: str = "cuda"):
        from sam3 import build_sam3_image_model
        from sam3.model.utils.sam1_utils import SAM2Transforms

        self.device = device
        self.model = build_sam3_image_model(
            device=device,
            load_from_HF=True,
            enable_inst_interactivity=True,
        )
        self.model.eval()

        # 이미지 전처리 & 마스크 후처리용 transforms
        self.transforms = SAM2Transforms(
            resolution=self.IMAGE_SIZE,
            mask_threshold=0.0,
        )

    @torch.inference_mode()
    def segment(self, image: ImageData, prompts: list[Prompt]) -> SegResult:
        H, W = image.height, image.width

        # 1. 이미지 전처리: (H,W,3) uint8 → (1,3,1008,1008) float32 [-1,1]
        preprocessed = self.transforms(image.rgb).unsqueeze(0).to(self.device)

        # 2. 백본 피처 추출 (SAM3 + SAM2 dual features)
        full_backbone_out = self.model.backbone.forward_image(preprocessed)

        # 3. sam2_backbone_out에 SAM tracker의 conv_s0/conv_s1 프로젝션 적용
        #    Sam3TrackerBase.forward_image가 하는 것과 동일
        sam2_out = full_backbone_out["sam2_backbone_out"]
        tracker = self.model.inst_interactive_predictor.model
        sam2_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(sam2_out["backbone_fpn"][0])
        sam2_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(sam2_out["backbone_fpn"][1])
        for i in range(len(sam2_out["backbone_fpn"])):
            sam2_out["backbone_fpn"][i] = sam2_out["backbone_fpn"][i].clone()
            sam2_out["vision_pos_enc"][i] = sam2_out["vision_pos_enc"][i].clone()

        # inference_state 구성
        inference_state = {
            "backbone_out": {"sam2_backbone_out": sam2_out},
            "original_height": H,
            "original_width": W,
        }

        masks_out: list[np.ndarray] = []
        labels_out: list[str] = []
        scores_out: list[float] = []

        for prompt in prompts:
            point_coords = (
                np.array(prompt.points, dtype=np.float32) if prompt.points else None
            )
            point_labels = (
                np.array(prompt.point_labels, dtype=np.int32) if prompt.point_labels else None
            )
            box = (
                np.array(prompt.box, dtype=np.float32) if prompt.box else None
            )

            # predict_inst가 내부적으로 inst_interactive_predictor에
            # backbone 피처를 주입하고 SAM-style predict를 수행
            masks, scores, _ = self.model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            masks_out.append(masks[best_idx].astype(bool))
            labels_out.append(prompt.label)
            scores_out.append(float(scores[best_idx]))

        return SegResult(masks=masks_out, labels=labels_out, scores=scores_out)
