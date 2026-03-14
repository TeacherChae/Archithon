# 로컬 개발 환경 설정

> CPU only. 파이프라인 구조 개발 및 단위 테스트 용도.

---

## 환경

| 항목 | 값 |
|------|----|
| OS | WSL2 (Linux 6.6, Ubuntu) |
| Python | 3.10.12 |
| 가상환경 | `/home/keonheechae/Archithon/.venv` |
| GPU | 없음 (CPU only) |
| MoGe 소스 | `/home/keonheechae/MoGe` |

---

## 초기 설치

```bash
cd /home/keonheechae/Archithon

# 1. CPU PyTorch
bash scripts/install_torch_cpu.sh

# 2. 나머지 의존성
.venv/bin/pip install -r requirements.txt
```

---

## SAM3 세그멘테이션 전략

로컬에서 SAM3 모델 추론은 느리거나 불안정할 수 있음.
**대안: SAM3 Playground에서 마스크 이미지 직접 다운로드 후 주입**

파이프라인에서 `SAMSegmentor` 대신 `PlaygroundMaskSegmentor` 사용:

```python
from src import PlaygroundMaskSegmentor, Prompt

segmentor = PlaygroundMaskSegmentor()
# SAM3 Playground에서 다운받은 마스크 PNG 경로를 Prompt에 전달
```

---

## device 자동 감지

코드 내 device는 자동 감지로 동작:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"  # 로컬 → "cpu"
```

`PipelineConfig` 기본값이 이 방식을 따르므로 별도 설정 불필요.

---

## 주의사항

- MoGe 추론: CPU 기준 이미지 1장당 1~3분 소요
- open3d: Python 3.10까지만 공식 지원 (현재 버전 적합)
- SAM3 체크포인트 다운로드는 `huggingface_hub` 통해 자동 처리
