#!/bin/bash
# 서버용 — RTX Pro 6000 (Blackwell) CUDA 12.8+
# RunPod PyTorch 이미지 사용 시 torch는 이미 설치됨 → 이 스크립트 불필요
# 직접 설치가 필요한 경우에만 실행
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
