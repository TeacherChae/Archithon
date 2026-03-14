# Archithon — 실행 계획

> 구현 순서, 파일 구조, 각 파일의 책임 및 인터페이스를 정의한다

---

## 1. 파일 구조

```
~/Archithon/
├── planning.md
├── specification.md
├── README.md
└── src/
    ├── __init__.py            # 패키지 공개 API
    ├── image_loader.py        # [Step 1] 이미지 로딩
    ├── depth_estimator.py     # [Step 2] MoGe-2 추론
    ├── segmentor.py           # [Step 3] SAM 세그멘테이션
    ├── point_labeler.py       # [Step 4] 2D 마스크 → 3D 레이블링
    ├── mesh_reconstructor.py  # [Step 5] 메시 재구성
    ├── pipeline.py            # 전체 오케스트레이션
    └── visualizer.py          # 시각화 유틸리티
```

---

## 2. 구현 순서 및 각 파일 명세

### Phase 1 — 데이터 구조 & 로딩

#### `image_loader.py`

**책임**: 경로에서 이미지를 읽고 MoGe 추론에 쓸 수 있는 형태로 변환

**클래스**: `ImageLoader`
**데이터**: `ImageData` (dataclass)

```python
@dataclass
class ImageData:
    rgb: np.ndarray        # (H, W, 3) uint8  — 원본 픽셀값
    tensor: torch.Tensor   # (3, H, W) float32 [0,1]  — MoGe 입력
    path: str
    height: int
    width: int

class ImageLoader:
    def load(self, path: str) -> ImageData
    def load_from_array(self, rgb: np.ndarray) -> ImageData
```

---

### Phase 2 — MoGe-2 추론

#### `depth_estimator.py`

**책임**: MoGe-2를 로드하고 ImageData를 받아 point map을 반환

**클래스**: `DepthEstimator`
**데이터**: `DepthResult` (dataclass)

```python
@dataclass
class DepthResult:
    points: np.ndarray        # (H, W, 3) float32  — 픽셀별 3D 좌표 (미터)
    depth: np.ndarray         # (H, W) float32
    normal: np.ndarray | None # (H, W, 3) float32  — 모델에 따라 None 가능
    mask: np.ndarray          # (H, W) bool  — 유효 픽셀
    intrinsics: np.ndarray    # (3, 3) float32

class DepthEstimator:
    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vitl-normal",
        device: str = "cuda",
        fp16: bool = False,
        resolution_level: int = 9,
    )
    def infer(self, image: ImageData) -> DepthResult
```

**핵심 구현 메모**
- `model.infer(image.tensor)` 호출
- 결과 텐서를 `.cpu().numpy()`로 변환 후 `DepthResult`에 담아 반환
- `torch.inference_mode()` 컨텍스트 안에서 실행

---

### Phase 3 — SAM 세그멘테이션

#### `segmentor.py`

**책임**: 이미지와 프롬프트를 받아 2D 이진 마스크를 반환
SAM3 구현체 기반

**클래스**: `BaseSegmentor` (ABC), `SAMSegmentor`, `Prompt` (dataclass)
**데이터**: `SegResult` (dataclass)

```python
@dataclass
class Prompt:
    label: str                       # 이 프롬프트가 대응하는 레이블 이름
    points: list[tuple[int,int]] | None   # [(x, y), ...]  클릭 좌표
    point_labels: list[int] | None        # 1=전경, 0=배경
    box: tuple[int,int,int,int] | None    # (x1, y1, x2, y2)

@dataclass
class SegResult:
    masks: list[np.ndarray]    # List of (H, W) bool
    labels: list[str]          # 각 마스크의 레이블 이름
    scores: list[float]        # 예측 신뢰도

class BaseSegmentor(ABC):
    @abstractmethod
    def segment(self, image: ImageData, prompts: list[Prompt]) -> SegResult: ...

class SAMSegmentor(BaseSegmentor):
    def __init__(
        self,
        checkpoint: str,        # SAM3 체크포인트 경로
        model_cfg: str,         # SAM3 모델 설정 파일
        device: str = "cuda",
    )
    def segment(self, image: ImageData, prompts: list[Prompt]) -> SegResult
```

**핵심 구현 메모**
- 프롬프트마다 `predictor.predict()`를 호출해 마스크 한 장씩 획득
- `multimask_output=True`로 받은 후 `scores` 기준으로 best mask 선택
- 결과를 `SegResult`에 묶어 반환

---

### Phase 4 — 2D 마스크 → 3D 레이블링 (핵심)

#### `point_labeler.py`

**책임**: `DepthResult`와 `SegResult`를 받아 레이블별 3D 점/색/법선을 분리

**클래스**: `PointLabeler`
**데이터**: `LabeledCloud` (dataclass)

```python
@dataclass
class LabeledCloud:
    # 레이블별 3D 데이터
    points: dict[str, np.ndarray]   # label → (N, 3) float32
    colors: dict[str, np.ndarray]   # label → (N, 3) uint8
    normals: dict[str, np.ndarray]  # label → (N, 3) float32 (없으면 None)

    # 전체 (레이블 무관)
    all_points: np.ndarray          # (M, 3)
    all_colors: np.ndarray          # (M, 3)
    label_map: np.ndarray           # (H, W) str  — 픽셀별 레이블

class PointLabeler:
    def label(
        self,
        image: ImageData,
        depth: DepthResult,
        seg: SegResult,
    ) -> LabeledCloud
```

**핵심 구현 메모**

```python
# 핵심 로직 (3줄)
for mask, label in zip(seg.masks, seg.labels):
    combined = mask & depth.mask                 # SAM 마스크 ∩ MoGe 유효 마스크
    points_per_label[label] = depth.points[combined]  # (N, 3) — 역투영 불필요
    colors_per_label[label] = image.rgb[combined]     # (N, 3)
```

---

### Phase 5 — 메시 재구성

#### `mesh_reconstructor.py`

**책임**: 레이블별 점군을 받아 각각 메시로 변환하고 파일로 저장

**클래스**: `MeshReconstructor`

```python
class MeshReconstructor:
    def __init__(
        self,
        method: Literal["poisson", "bpa", "alpha"] = "poisson",
        depth: int = 9,          # Poisson depth (높을수록 세밀)
        radii: list[float] | None = None,  # BPA radii
    )

    def reconstruct(
        self,
        cloud: LabeledCloud,
    ) -> dict[str, o3d.geometry.TriangleMesh]

    def save(
        self,
        meshes: dict[str, o3d.geometry.TriangleMesh],
        output_dir: str,
        formats: list[str] = ["glb", "ply"],
    ) -> dict[str, list[str]]   # label → [저장된 파일 경로들]
```

**핵심 구현 메모**
- Open3D `PointCloud` 생성 → 법선 추정(없을 경우) → Poisson 재구성
- 법선이 이미 있으면 (`depth.normal`) 그대로 사용해 품질 향상
- `trimesh`로 GLB 변환 및 저장

---

### Phase 6 — 오케스트레이터

#### `pipeline.py`

**책임**: 모든 클래스를 조립해서 이미지 경로 + 프롬프트 → 메시 파일까지 원스톱 실행

```python
@dataclass
class PipelineConfig:
    # 모델
    moge_model: str = "Ruicheng/moge-2-vitl-normal"
    sam_checkpoint: str = "checkpoints/sam3_hiera_large.pt"
    sam_cfg: str = "sam3_hiera_l.yaml"
    device: str = "cuda"
    fp16: bool = False
    resolution_level: int = 9

    # 메시
    mesh_method: str = "poisson"
    mesh_depth: int = 9
    output_formats: list[str] = field(default_factory=lambda: ["glb", "ply"])

    # 출력
    output_dir: str = "output"
    save_intermediates: bool = True   # 깊이, 세그멘테이션 시각화 저장 여부

class ArchithonPipeline:
    def __init__(self, config: PipelineConfig)

    def run(
        self,
        image_path: str,
        prompts: list[Prompt],
    ) -> dict[str, list[str]]   # label → [저장된 파일 경로들]
```

---

### Phase 7 — 시각화

#### `visualizer.py`

**책임**: 각 단계의 결과를 이미지로 저장하는 유틸리티

```python
class Visualizer:
    @staticmethod
    def save_depth(depth: DepthResult, path: str) -> None
        # 컬러맵 적용 깊이 이미지 저장

    @staticmethod
    def save_segmentation_overlay(image: ImageData, seg: SegResult, path: str) -> None
        # 원본 이미지 위에 마스크 오버레이 (레이블별 다른 색상)

    @staticmethod
    def save_labeled_cloud_image(image: ImageData, cloud: LabeledCloud, path: str) -> None
        # 픽셀별 레이블을 색상으로 표현한 이미지 저장

    @staticmethod
    def save_normal(depth: DepthResult, path: str) -> None
        # 법선맵 이미지 저장
```

---

## 3. 구현 순서 (의존성 순)

```
1. image_loader.py      — 의존성 없음
2. depth_estimator.py   — ImageData 사용
3. segmentor.py         — ImageData 사용
4. point_labeler.py     — ImageData + DepthResult + SegResult 사용
5. mesh_reconstructor.py — LabeledCloud 사용
6. visualizer.py        — 모든 결과 타입 사용
7. pipeline.py          — 모든 클래스 조립
8. __init__.py          — 공개 API 정리
```

---

## 4. 사용 예시 (완성 후)

### CLI 스타일 실행

```python
from src import ArchithonPipeline, PipelineConfig, Prompt

config = PipelineConfig(
    moge_model="Ruicheng/moge-2-vitl-normal",
    sam_checkpoint="checkpoints/sam3_hiera_large.pt",
    sam_cfg="sam3_hiera_l.yaml",
    output_dir="output/my_scene",
)

pipeline = ArchithonPipeline(config)

prompts = [
    Prompt(label="chair",   points=[(320, 480)], point_labels=[1]),
    Prompt(label="table",   box=(100, 200, 600, 700)),
    Prompt(label="window",  points=[(800, 150)], point_labels=[1]),
]

result = pipeline.run("scene.jpg", prompts)
# output/my_scene/
# ├── chair.glb
# ├── chair.ply
# ├── table.glb
# ├── table.ply
# ├── window.glb
# ├── window.ply
# ├── depth_vis.png
# ├── segmentation_vis.png
# └── labeled_cloud_vis.png
```

### 단계별 직접 제어

```python
from src import ImageLoader, DepthEstimator, SAMSegmentor, PointLabeler, MeshReconstructor

loader    = ImageLoader()
estimator = DepthEstimator(fp16=True)
segmentor = SAMSegmentor("checkpoints/sam3_hiera_large.pt", "sam3_hiera_l.yaml")
labeler   = PointLabeler()
recon     = MeshReconstructor(method="poisson", depth=9)

image  = loader.load("scene.jpg")
depth  = estimator.infer(image)
seg    = segmentor.segment(image, prompts)
cloud  = labeler.label(image, depth, seg)
meshes = recon.reconstruct(cloud)
recon.save(meshes, "output/")
```

---

## 5. 체크리스트

### 구현 전 준비
- [ ] SAM3 패키지 설치: `pip install git+https://github.com/facebookresearch/sam3.git`
- [ ] SAM3 체크포인트 다운로드: `sam3_hiera_large.pt`
- [ ] Open3D 설치: `pip install open3d`
- [ ] MoGe 설치 확인: `/home/keonheechae/MoGe` 패키지 경로 설정

### 구현 완료 기준
- [ ] `image_loader.py` — `ImageData` 반환, BGR→RGB 변환 포함
- [ ] `depth_estimator.py` — `DepthResult` 반환, GPU/CPU 선택 가능
- [ ] `segmentor.py` — `SAMSegmentor` 동작, 다중 프롬프트 지원
- [ ] `point_labeler.py` — `combined_mask` 인덱싱으로 3D 레이블링
- [ ] `mesh_reconstructor.py` — Poisson 기본, GLB/PLY 저장
- [ ] `visualizer.py` — 깊이/세그/레이블 이미지 저장
- [ ] `pipeline.py` — 단일 `run()` 호출로 전체 파이프라인 실행
- [ ] `__init__.py` — 공개 클래스 모두 노출

---

## 6. 알려진 제약사항

| 제약 | 내용 | 대응 |
|------|------|------|
| 단일 이미지 깊이 품질 | 반사/투명 재질 예측 어려움 | `moge_mask`로 신뢰도 낮은 픽셀 제거 |
| 메시 내부 채움 | Poisson은 폐쇄 메시를 가정 | `trimesh.repair.fill_holes()` 후처리 |
| SAM 다중 객체 겹침 | 마스크 간 픽셀 중복 가능 | 레이블 우선순위 또는 교집합 제거 |
| 점군 밀도 불균일 | 원거리 영역은 밀도 낮음 | Voxel downsampling 전처리 가능 |
