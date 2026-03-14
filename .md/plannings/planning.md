# Archithon — 기획 문서

> 단일 2D 이미지에서 객체별로 분리된 3D 메시를 추출하는 파이프라인

---

## 1. 프로젝트 개요

### 목적

하나의 RGB 이미지로부터:
1. 전체 씬의 3D 점군(Point Cloud)을 복원하고
2. 원하는 객체를 2D에서 세그멘테이션한 후
3. 해당 객체의 3D 점들만 분리해
4. 객체별 3D 메시(Mesh)를 재구성한다

### 핵심 전제

- **학습 없음**: 사전학습된 모델(MoGe-2, SAM3)을 추론 전용으로 사용
- **단일 이미지**: 멀티뷰, 스테레오, LiDAR 불필요
- **2D↔3D 픽셀 대응 보장**: MoGe-2의 point map `(H, W, 3)` 구조 덕분에
  픽셀 `(h, w)` = 3D 좌표 `points[h, w]`가 인덱스로 바로 대응됨

---

## 2. 전체 파이프라인

```
[INPUT]  2D RGB 이미지
    │
    ├─────────────────────────────────────────────────────────────┐
    │                                                             │
    ▼                                                             ▼
[STEP 2] MoGe-2 추론                                   [STEP 3] SAM 세그멘테이션
  점지도 (H, W, 3)  ← 픽셀별 3D 좌표                    세그 마스크 (H, W)  ← 객체 레이블
  깊이맵 (H, W)                                          프롬프트: 텍스트 / 좌표 / 바운딩박스
  유효마스크 (H, W)
  법선맵 (H, W, 3)
    │                                                             │
    └──────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                  [STEP 4] PointLabeler
                    SAM 마스크 → MoGe 점지도에 적용
                    combined_mask = sam_mask & moge_mask
                    labeled_points[label] = points[combined_mask]
                           │
                           ▼
                  [STEP 5] MeshReconstructor
                    레이블별 점군 → 메시 재구성 (Poisson / BPA)
                    labeled_mesh[label] = trimesh / open3d mesh
                           │
                           ▼
                  [OUTPUT] 객체별 .glb / .ply / .obj 메시 파일
```

---

## 3. 핵심 기술 분석

### 3-1. MoGe-2 (Step 2)

| 항목 | 내용 |
|------|------|
| 모델 | `Ruicheng/moge-2-vitl-normal` |
| 입력 | RGB 이미지 `(3, H, W)` float32 [0,1] |
| 출력 | `points (H,W,3)`, `depth (H,W)`, `normal (H,W,3)`, `mask (H,W)`, `intrinsics (3,3)` |
| 핵심 | **point map 구조** — 인덱스 `(h,w)` = 픽셀의 3D 좌표. 역투영 불필요 |
| 스케일 | 미터 단위 절대값 (MoGe-2) |

### 3-2. SAM3 (Step 3)

| 항목 | 내용 |
|------|------|
| 모델 | SAM3 (Meta, `sam3` 패키지) |
| 프롬프트 종류 | 좌표 클릭, 바운딩박스, 텍스트(Grounded SAM) |
| 출력 | 이진 마스크 `(H, W)` bool |
| 다중 객체 | 여러 프롬프트 → 여러 마스크 → 각각 레이블 |
| 로컬 실행 불가 시 | SAM3 Playground에서 마스킹된 이미지 직접 다운로드 후 파이프라인에 주입 가능 |

### 3-3. PointLabeler (Step 4) — 핵심 로직

```
pixel (h, w)이 SAM 마스크에 포함  →  points[h, w]에 동일 레이블 부여

combined_mask = sam_mask & moge_valid_mask
labeled_points = points[combined_mask]   # (N, 3)
labeled_colors = image[combined_mask]    # (N, 3) RGB
```

역투영(back-projection) 없이 인덱스 대응만으로 해결.

### 3-4. MeshReconstructor (Step 5)

| 방법 | 특징 | 추천 상황 |
|------|------|---------|
| **Poisson Surface Reconstruction** | 매끄러운 폐쇄 메시 | 큰 면적 객체, 내부가 있는 구조물 |
| **Ball-Pivoting Algorithm (BPA)** | 점군 밀도 기반, 빠름 | 얇은 표면, 외곽선이 중요한 객체 |
| **Alpha Shape** | 오목한 형상 처리 | 불규칙한 형상 |

---

## 4. 설계 원칙

### OOP 구조

각 스텝은 독립적인 클래스로 분리. 의존성은 파이프라인 오케스트레이터(`Pipeline`)가 조립.

```
ImageLoader          → ImageData(rgb, tensor, path)
DepthEstimator       → DepthResult(points, depth, normal, mask, intrinsics)
BaseSegmentor        → SegResult(masks, labels, scores)
  └─ SAMSegmentor       (SAM3 구현체 / Playground 마스크 주입 모드 지원)
PointLabeler         → LabeledCloud(points_per_label, colors_per_label, normals_per_label)
MeshReconstructor    → Dict[label → trimesh.Trimesh]
Visualizer           → 시각화 유틸리티
ArchithonPipeline    → 전체 오케스트레이션
```

### 데이터 흐름 원칙

- 각 클래스는 **dataclass**로 정의된 결과 객체를 반환 (dict 반환 금지)
- 클래스 간 의존성은 생성자 주입(DI)
- GPU 메모리 관리는 각 클래스가 책임

### 확장성

- `BaseSegmentor` 추상 클래스로 SAM3 로컬 추론, Playground 마스크 주입, Grounded SAM, 커스텀 모델 교체 가능
- `MeshReconstructor`의 알고리즘 전략 패턴으로 교체 가능
- `Pipeline`에 새로운 스텝 삽입 가능 (텍스처 베이킹, 포인트 필터링 등)

---

## 5. 입출력 명세

### 입력

| 항목 | 타입 | 설명 |
|------|------|------|
| 이미지 경로 | `str` | jpg / png |
| 세그멘테이션 프롬프트 | `List[Prompt]` | 좌표, 박스, 텍스트 중 하나 이상 |
| 레이블 이름 | `str` | 각 프롬프트에 대응하는 이름 |

### 출력

| 항목 | 형식 | 설명 |
|------|------|------|
| `{label}.glb` | GLTF Binary | 텍스처 3D 메시 (Blender, Unity 호환) |
| `{label}.ply` | PLY | 레이블된 포인트 클라우드 |
| `depth_vis.png` | PNG | 깊이 시각화 |
| `segmentation_vis.png` | PNG | 2D 세그멘테이션 오버레이 |
| `labeled_cloud_vis.png` | PNG | 레이블별 색상 포인트 클라우드 |

---

## 6. 환경 의존성

| 패키지 | 용도 | 비고 |
|--------|------|------|
| `torch >= 2.0` | 딥러닝 추론 | CUDA 권장 |
| `moge` | MoGe-2 추론 | `/home/keonheechae/MoGe` |
| `sam3` | SAM 세그멘테이션 | Meta SAM3 패키지 (로컬 실행 불가 시 Playground에서 마스크 이미지 다운로드 후 주입) |
| `open3d` | 메시 재구성 | Poisson, BPA |
| `trimesh` | 메시 I/O | GLB 저장 |
| `opencv-python` | 이미지 처리 | |
| `numpy` | 배열 연산 | |

---

## 7. 미해결 질문

| # | 질문 | 현재 답 |
|---|------|--------|
| 1 | SAM3 로컬 실행 환경 구성 | 로컬 실행 우선 시도, 불가 시 SAM3 Playground에서 마스크 이미지 다운로드 후 `BaseSegmentor`에 주입 |
| 2 | 단일 객체 메시의 내부 채움 방법 | Poisson이 기본, 추후 mesh closing 추가 가능 |
| 3 | 점군 밀도가 낮을 때 메시 품질 | 법선(normal) 활용으로 보완 |
