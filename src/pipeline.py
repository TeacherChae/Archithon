from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    # ── Device ────────────────────────────────────────────────
    device: str = os.getenv("DEVICE", "cuda")

    # ── MoGe-2 ────────────────────────────────────────────────
    moge_model:           str  = os.getenv("MOGE_MODEL", "Ruicheng/moge-2-vitl-normal")
    moge_resolution_level: int = int(os.getenv("MOGE_RESOLUTION_LEVEL", "9"))
    moge_fp16:            bool = os.getenv("MOGE_FP16", "false").lower() == "true"

    # ── SAM3 ──────────────────────────────────────────────────
    sam_confidence_threshold: float = float(os.getenv("SAM_CONFIDENCE_THRESHOLD", "0.3"))
    # 쉼표 구분 레이블 목록  예) "rooftop,brick wall,window"
    sam_labels: list[str] = field(
        default_factory=lambda: os.getenv("SAM_LABELS", "rooftop,brick wall,window").split(",")
    )
    # 여러 인스턴스를 하나로 합칠 레이블 (OR)  예) "window"
    sam_merge_labels: list[str] = field(
        default_factory=lambda: [
            l for l in os.getenv("SAM_MERGE_LABELS", "window").split(",") if l
        ]
    )

    def build_prompts(self) -> list:
        """SAM_LABELS / SAM_MERGE_LABELS 설정으로 Prompt 리스트 생성"""
        from .segmentor import Prompt
        return [
            Prompt(
                label=lbl,
                merge_instances=(lbl in self.sam_merge_labels),
            )
            for lbl in self.sam_labels
        ]

    # ── Mesh ──────────────────────────────────────────────────
    mesh_method:           str = os.getenv("MESH_METHOD", "voxel_mc")
    mesh_depth:            int = int(os.getenv("MESH_DEPTH", "8"))
    mesh_max_points:       int = int(os.getenv("MESH_MAX_POINTS", "150000"))
    mesh_voxel_resolution: int = int(os.getenv("MESH_VOXEL_RESOLUTION", "128"))
    mesh_output_formats: list[str] = field(
        default_factory=lambda: os.getenv("MESH_OUTPUT_FORMATS", "glb,ply").split(",")
    )

    # ── Pipeline ──────────────────────────────────────────────
    output_dir:         str  = os.getenv("OUTPUT_DIR", "output")
    save_intermediates: bool = os.getenv("SAVE_INTERMEDIATES", "true").lower() == "true"


class ArchithonPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        from .depth_estimator import DepthEstimator
        from .image_loader import ImageLoader
        from .mesh_reconstructor import MeshReconstructor
        from .point_labeler import PointLabeler
        from .segmentor import SAMSegmentor

        self.config = config or PipelineConfig()
        cfg = self.config

        self.loader       = ImageLoader()
        self.depth_est    = DepthEstimator(
            model_name=cfg.moge_model,
            device=cfg.device,
            fp16=cfg.moge_fp16,
            resolution_level=cfg.moge_resolution_level,
        )
        self.segmentor    = SAMSegmentor(
            device=cfg.device,
            confidence_threshold=cfg.sam_confidence_threshold,
        )
        self.labeler      = PointLabeler()
        self.reconstructor = MeshReconstructor(
            method=cfg.mesh_method,
            depth=cfg.mesh_depth,
            max_points=cfg.mesh_max_points,
            voxel_resolution=cfg.mesh_voxel_resolution,
            device=cfg.device,
        )

    def run(
        self,
        image_path: str,
        prompts: list | None = None,    # None이면 .env의 SAM_LABELS로 자동 생성
    ) -> dict[str, list[str]]:
        """
        image_path 하나를 받아 레이블별 메시 파일까지 생성.

        Returns:
            { label: ["/path/to/label.glb", "/path/to/label.ply", ...] }
        """
        if prompts is None:
            prompts = self.config.build_prompts()
        from .visualizer import Visualizer

        cfg = self.config
        out = cfg.output_dir
        intermediates = cfg.save_intermediates

        # ── Step 1: 이미지 로딩 ───────────────────────────────
        print(f"\n[1/5] 이미지 로딩: {image_path}")
        image = self.loader.load(image_path)
        print(f"      {image.width}x{image.height}")

        # ── Step 2: 깊이 추정 ─────────────────────────────────
        print("\n[2/5] MoGe-2 깊이 추정...")
        depth = self.depth_est.infer(image)
        print(f"      유효 픽셀: {depth.mask.sum():,} / {depth.mask.size:,}")
        if intermediates:
            depth.save(os.path.join(out, "step2_depth"))
            Visualizer.save_depth(depth, os.path.join(out, "step2_depth", "depth_vis.png"))
            Visualizer.save_normal(depth, os.path.join(out, "step2_depth", "normal_vis.png"))

        # ── Step 3: 세그멘테이션 ──────────────────────────────
        print("\n[3/5] SAM3 세그멘테이션...")
        seg = self.segmentor.segment(image, prompts)
        for lbl, mask, score in zip(seg.labels, seg.masks, seg.scores):
            print(f"      {lbl}: score={score:.3f}  pixels={mask.sum():,}")
        if intermediates:
            seg.save(os.path.join(out, "step3_seg"), image_rgb=image.rgb)
            Visualizer.save_segmentation_overlay(
                image, seg, os.path.join(out, "step3_seg", "seg_vis.png")
            )

        # ── Step 4: 포인트 레이블링 ───────────────────────────
        print("\n[4/5] 포인트 레이블링...")
        cloud = self.labeler.label(image, depth, seg)
        for lbl, pts in cloud.points.items():
            print(f"      {lbl}: {len(pts):,} pts")
        if intermediates:
            cloud.save(os.path.join(out, "step4_cloud"))
            Visualizer.save_labeled_cloud_image(
                image, cloud, os.path.join(out, "step4_cloud", "cloud_vis.png")
            )

        # ── Step 5: 메시 재구성 ───────────────────────────────
        print("\n[5/5] 메시 재구성...")
        meshes = self.reconstructor.reconstruct(cloud)
        mesh_dir = os.path.join(out, "step5_mesh")
        saved = self.reconstructor.save(
            meshes, mesh_dir, formats=cfg.mesh_output_formats
        )

        print(f"\n완료. 출력 디렉토리: {out}")
        return saved


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Archithon pipeline")
    parser.add_argument(
        "image",
        nargs="?",
        default="samples/input/sample_image.png",
        help="입력 이미지 경로 (기본값: samples/input/sample_image.png)",
    )
    args = parser.parse_args()

    # src/ 패키지를 직접 실행할 때 상위 디렉토리를 경로에 추가
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    pipeline = ArchithonPipeline()
    cfg = pipeline.config
    print(f"labels : {cfg.sam_labels}")
    print(f"merge  : {cfg.sam_merge_labels}")
    print(f"device : {cfg.device}")
    print(f"output : {cfg.output_dir}")

    result = pipeline.run(args.image)
    print("\n생성된 파일:")
    for label, paths in result.items():
        for p in paths:
            print(f"  {p}")
