from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .image_loader import ImageData


@dataclass
class Prompt:
    """
    SAM3 grounding 프롬프트.

    label            : 자연어 텍스트 ("roof", "brick wall", "window" 등)
    box              : 선택적 기하학적 힌트 박스
                       (cx, cy, w, h) 형식, 이미지 크기 기준 정규화 [0, 1]
                       예) 이미지 중앙 절반 → (0.5, 0.5, 0.5, 0.5)
    box_positive     : True = 전경(포함) 박스, False = 배경(제외) 박스
    merge_instances  : True  → 탐지된 여러 인스턴스를 하나의 마스크로 합침 (OR)
                       False → score 가장 높은 단일 인스턴스만 사용
    """
    label: str
    box: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    box_positive: bool = True
    merge_instances: bool = True


@dataclass
class SegResult:
    masks: list[np.ndarray]    # List of (H, W) bool
    labels: list[str]          # 각 마스크의 레이블 이름
    scores: list[float]        # 예측 신뢰도 (인스턴스가 여럿이면 최고 score)

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
            overlay = image_rgb.copy()
            palette = [
                (255, 80,  80),
                (80,  200, 80),
                (80,  120, 255),
                (255, 200, 50),
                (200, 80,  255),
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
            for mask, lbl in zip(self.masks, self.labels):
                alpha = mask.astype(np.uint8) * 255
                rgba = np.dstack([image_rgb, alpha])
                cv2.imwrite(
                    os.path.join(directory, f"{lbl}_masked.png"),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                )
        # ── GeoJSON ───────────────────────────────────────────
        H, W = (self.masks[0].shape if self.masks else (0, 0))
        features = []
        for i, (lbl, mask, score) in enumerate(zip(self.labels, self.masks, self.scores)):
            ys, xs = np.where(mask)
            bbox = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    if len(xs) else None)
            features.append({
                "type": "Feature",
                "id": lbl,
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": np.stack([xs, ys], axis=1).tolist(),
                },
                "properties": {
                    "label": lbl,
                    "sam3_score": round(score, 6),
                    "pixel_count": int(mask.sum()),
                    "coverage_pct": round(float(mask.mean()) * 100, 2),
                    "bbox_xyxy": bbox,
                },
            })

        geojson = {
            "type": "FeatureCollection",
            "properties": {
                "source": "SAM3",
                "model": "facebook/sam3",
                "inference": "grounding",
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
    SAM3 grounding 기반 세그멘테이션.

    추론 경로 (Sam3Processor):
      1. set_image()       : 이미지 → 비전 피처 추출 (1회)
      2. set_text_prompt() : 텍스트 → 언어 피처 추출 → forward_grounding
      3. add_geometric_prompt() : (선택) 박스 힌트 → forward_grounding 재실행

    반환된 state["masks"] : (K, 1, H, W) bool  — K개 인스턴스
    Prompt.merge_instances=True  → K개 마스크를 OR 합산
    Prompt.merge_instances=False → score 최고 인스턴스만 선택
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = device
        model = build_sam3_image_model(
            device=device,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        model.eval()
        self.processor = Sam3Processor(
            model,
            resolution=1008,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    @torch.inference_mode()
    def segment(self, image: ImageData, prompts: list[Prompt]) -> SegResult:
        from PIL import Image as PILImage

        H, W = image.height, image.width
        pil_image = PILImage.fromarray(image.rgb)

        # 이미지 피처는 한 번만 추출
        state = self.processor.set_image(pil_image)

        masks_out:  list[np.ndarray] = []
        labels_out: list[str]        = []
        scores_out: list[float]      = []

        for prompt in prompts:
            self.processor.reset_all_prompts(state)

            # 텍스트 프롬프트 → grounding 실행
            state = self.processor.set_text_prompt(prompt.label, state)

            # 선택적 박스 힌트 → grounding 재실행
            if prompt.box is not None:
                state = self.processor.add_geometric_prompt(
                    box=list(prompt.box),
                    label=prompt.box_positive,
                    state=state,
                )

            # 결과 수집
            if "masks" in state and len(state["masks"]) > 0:
                # state["masks"]: (K, 1, H, W) bool tensor
                # state["scores"]: (K,) float tensor
                k_masks  = state["masks"].squeeze(1).cpu().numpy()  # (K, H, W)
                k_scores = state["scores"].cpu().numpy()            # (K,)

                if prompt.merge_instances:
                    # 모든 인스턴스 OR 합산 → 창문처럼 여러 개인 경우 유리
                    mask  = k_masks.any(axis=0)                     # (H, W)
                    score = float(k_scores.max())
                else:
                    best  = int(k_scores.argmax())
                    mask  = k_masks[best]
                    score = float(k_scores[best])
            else:
                print(f"[{prompt.label}] confidence threshold 이하 — 빈 마스크 반환")
                mask  = np.zeros((H, W), dtype=bool)
                score = 0.0

            masks_out.append(mask)
            labels_out.append(prompt.label)
            scores_out.append(score)

        return SegResult(masks=masks_out, labels=labels_out, scores=scores_out)
