# 서버 환경 설정 (RunPod)

> RTX Pro 6000 (Blackwell) + CUDA 12.8+. 실제 품질 테스트 및 최종 실행 용도.

---

## 환경

| 항목 | 값 |
|------|----|
| 플랫폼 | RunPod |
| GPU | NVIDIA RTX Pro 6000 (Blackwell, GB202) |
| VRAM | 96 GB |
| CUDA | 12.8+ 필수 (Blackwell 아키텍처 요구사항) |
| Python | 3.11 (RunPod PyTorch 이미지 기본) |

---

## RunPod 이미지 선택

Pod 생성 시 **RunPod 공식 PyTorch 템플릿** 사용.
torch + CUDA가 이미지에 포함되어 있으므로 별도 torch 설치 불필요.

> 선택 기준: `CUDA 12.8` 이상 버전의 PyTorch 이미지 선택
> (Blackwell GPU는 CUDA 12.8 미만에서 동작하지 않음)

---

## 서버 초기 설정

```bash
# 1. 레포 clone
git clone <Archithon repo URL>
cd Archithon

# 2. MoGe clone (별도 필요)
git clone https://github.com/microsoft/MoGe.git ~/MoGe

# 3. 의존성 설치 (torch는 이미지에 포함 → 자동 스킵됨)
pip install -r requirements.txt

# 4. SAM3 체크포인트 (자동 다운로드 또는 수동)
# huggingface_hub 통해 첫 실행 시 자동 다운로드됨
```

---

## device 자동 감지

로컬과 동일한 코드, 서버에서는 자동으로 CUDA 선택:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"  # 서버 → "cuda"
```

코드 수정 없이 전환됨.

---

## 주의사항

- RunPod PyTorch 이미지의 Python 버전이 3.11일 경우 open3d 호환성 확인 필요
  (open3d 공식 지원: Python ≤ 3.11 → 정상)
- MoGe는 PyPI 미등록이므로 서버에서도 별도 `git clone` 필요
- `scripts/install_torch_cuda.sh`는 RunPod 이미지 사용 시 실행 불필요
  (직접 환경 구성 시에만 사용)
- Pod 종료 시 `/root` 이하 데이터 삭제됨 → Network Volume 마운트 권장 (체크포인트, 출력물 보존)
