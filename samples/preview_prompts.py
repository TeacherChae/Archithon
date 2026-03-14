"""
프롬프트 좌표를 이미지 위에 시각화 — SAM3 실행 없이 박스·점 위치만 확인.

실행:
    cd /root/workspace/Archithon
    source .venv/bin/activate
    python3 samples/preview_prompts.py
"""

import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from src.prompts_config import ARCHITECTURAL_PROMPTS

img = cv2.imread("samples/input/sample_image.png")

COLORS = {
    "roof":   (255, 200,   0),   # 노랑
    "wall":   (0,   200, 100),   # 초록
    "window": ( 80, 120, 255),   # 파랑
}

for prompt in ARCHITECTURAL_PROMPTS:
    color = COLORS.get(prompt.label, (200, 200, 200))

    # 박스
    if prompt.box is not None:
        x1, y1, x2, y2 = prompt.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, prompt.label, (x1 + 5, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 전경/배경 점
    if prompt.points is not None:
        for (x, y), lbl in zip(prompt.points, prompt.point_labels):
            if lbl == 1:   # 전경 — 초록 원
                cv2.circle(img, (x, y), 14, (0, 255, 0), -1)
                cv2.circle(img, (x, y), 14, (0, 0, 0),   2)
            else:           # 배경 — 빨간 X
                cv2.drawMarker(img, (x, y), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 20, 3)

out = "samples/output/prompt_preview.png"
cv2.imwrite(out, img)
print(f"saved → {out}")
print("  green circle = foreground(1),  red X = background(0)")
