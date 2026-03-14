"""
Step 3: SAM3 세그멘테이션 재실행 (새 프롬프트 적용)

실행:
    cd /root/workspace/Archithon
    source .venv/bin/activate
    python3 samples/run_step3_seg.py
"""

import sys
sys.path.insert(0, ".")

import cv2
import numpy as np

from src.image_loader import ImageLoader
from src.segmentor import SAMSegmentor, SegResult
from src.prompts_config import ARCHITECTURAL_PROMPTS

OUT_DIR = "samples/output/step3_seg"

image = ImageLoader().load("samples/input/sample_image.png")
print(f"image: {image.width}x{image.height}")
print(f"prompts: {[p.label for p in ARCHITECTURAL_PROMPTS]}")

segmentor = SAMSegmentor(device="cuda")
seg = segmentor.segment(image, ARCHITECTURAL_PROMPTS)

for lbl, mask, score in zip(seg.labels, seg.masks, seg.scores):
    print(f"  {lbl}: score={score:.4f}  pixels={mask.sum():,}")

seg.save(OUT_DIR, image_rgb=image.rgb)
print(f"\n저장 완료 → {OUT_DIR}")
print("확인: samples/output/step3_seg/overlay.png")
