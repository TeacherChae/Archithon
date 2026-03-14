import os
from typing import Literal

import numpy as np
import open3d as o3d
import trimesh

from .point_labeler import LabeledCloud


class MeshReconstructor:
    def __init__(
        self,
        method: Literal["voxel_mc", "poisson", "bpa", "alpha"] = "voxel_mc",
        depth: int = 9,                   # poisson 전용
        radii: list[float] | None = None, # bpa 전용
        voxel_size: float | None = None,  # None → max_points random sampling
        max_points: int = 150_000,        # 라벨당 최대 포인트 수 (poisson/bpa/alpha)
        voxel_resolution: int = 128,      # voxel_mc 전용: 격자 해상도
        device: str = "cuda",
    ):
        self.method = method
        self.poisson_depth = depth
        self.radii = radii or [0.005, 0.01, 0.02, 0.04]
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.voxel_resolution = voxel_resolution
        self.device = device

    # ──────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────
    def reconstruct(
        self,
        cloud: LabeledCloud,
    ) -> dict[str, o3d.geometry.TriangleMesh]:
        meshes: dict[str, o3d.geometry.TriangleMesh] = {}

        for label, pts in cloud.points.items():
            if len(pts) < 10:
                print(f"[{label}] 점군이 너무 작음 ({len(pts)}개), 스킵")
                continue

            cols = cloud.colors.get(label)
            nrm = cloud.normals.get(label)

            if self.method == "voxel_mc":
                mesh = self._voxel_mc(pts, cols)
            else:
                mesh = self._open3d_reconstruct(pts, cols, nrm)

            if mesh is not None:
                meshes[label] = mesh
                print(f"[{label}] 메시: {len(mesh.vertices)}v / {len(mesh.triangles)}t")

        return meshes

    def save(
        self,
        meshes: dict[str, o3d.geometry.TriangleMesh],
        output_dir: str,
        formats: list[str] | None = None,
    ) -> dict[str, list[str]]:
        if formats is None:
            formats = ["glb", "ply"]

        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, list[str]] = {}

        for label, mesh in meshes.items():
            paths = []
            for fmt in formats:
                path = os.path.join(output_dir, f"{label}.{fmt}")

                if fmt == "ply":
                    o3d.io.write_triangle_mesh(path, mesh)
                    paths.append(path)

                elif fmt in ("glb", "obj"):
                    verts = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.triangles)
                    colors = np.asarray(mesh.vertex_colors)

                    tm = trimesh.Trimesh(vertices=verts, faces=faces)
                    if len(colors) == len(verts):
                        tm.visual.vertex_colors = (colors * 255).astype(np.uint8)

                    trimesh.repair.fill_holes(tm)
                    tm.export(path)
                    paths.append(path)

                else:
                    print(f"지원하지 않는 포맷: {fmt}")

            saved[label] = paths

        return saved

    # ──────────────────────────────────────────────────────────────
    # voxel_mc  (GPU 복셀화 + marching cubes)
    # ──────────────────────────────────────────────────────────────
    def _voxel_mc(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> o3d.geometry.TriangleMesh | None:
        try:
            import torch
            import torch.nn.functional as F
            from skimage.measure import marching_cubes
        except ImportError as e:
            print(f"voxel_mc 의존성 없음: {e}. poisson으로 fallback.")
            return self._open3d_reconstruct(pts, colors, None)

        res = self.voxel_resolution
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")

        pts_t = torch.tensor(pts, dtype=torch.float32, device=dev)

        # ── 정규화 ─────────────────────────────────────────────
        min_pt = pts_t.min(0).values
        max_pt = pts_t.max(0).values
        scale = (max_pt - min_pt).max()

        pts_norm = (pts_t - min_pt) / (scale + 1e-8)   # [0, ~1]

        # ── 복셀 인덱스 ────────────────────────────────────────
        idx = (pts_norm * (res - 1)).long().clamp(0, res - 1)  # (N, 3)
        flat = idx[:, 0] * res * res + idx[:, 1] * res + idx[:, 2]  # (N,)

        # ── 점유 격자 ──────────────────────────────────────────
        grid = torch.zeros(res ** 3, device=dev)
        grid.scatter_(0, flat, 1.0)
        grid = grid.reshape(1, 1, res, res, res)

        # GPU 3D Gaussian smoothing (avg_pool3d 3회 ≈ Gaussian)
        for _ in range(3):
            grid = F.avg_pool3d(grid, kernel_size=3, stride=1, padding=1)

        grid_np = grid.squeeze().cpu().numpy()

        # ── 색상 격자 ──────────────────────────────────────────
        color_grid: np.ndarray | None = None
        if colors is not None and len(colors) == len(pts):
            cols_t = torch.tensor(colors, dtype=torch.float32, device=dev) / 255.0
            counts = torch.zeros(res ** 3, device=dev)
            color_acc = torch.zeros((res ** 3, 3), device=dev)
            counts.scatter_add_(0, flat, torch.ones(len(flat), device=dev))
            color_acc.scatter_add_(
                0, flat.unsqueeze(1).expand(-1, 3), cols_t
            )
            nonzero = counts > 0
            color_acc[nonzero] /= counts[nonzero].unsqueeze(1)
            color_grid = color_acc.reshape(res, res, res, 3).cpu().numpy()

        # ── Marching cubes ─────────────────────────────────────
        level = float(grid_np.max() * 0.1)   # 점유도 10% 이상을 표면으로
        if grid_np.max() < 1e-6:
            print("복셀 격자가 비어 있음, 스킵")
            return None

        verts, faces, normals, _ = marching_cubes(grid_np, level=level)

        # ── 역정규화 ───────────────────────────────────────────
        scale_np = scale.cpu().item()
        min_np = min_pt.cpu().numpy()
        verts_world = verts / (res - 1) * scale_np + min_np

        # ── Open3D 메시 조립 ───────────────────────────────────
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

        if color_grid is not None:
            vert_idx = np.clip(verts.astype(int), 0, res - 1)
            vert_colors = color_grid[
                vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]
            ]
            mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)

        mesh.compute_vertex_normals()
        return mesh

    # ──────────────────────────────────────────────────────────────
    # Open3D 기반 (poisson / bpa / alpha) — 기존 로직 유지
    # ──────────────────────────────────────────────────────────────
    def _open3d_reconstruct(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None,
        normals: np.ndarray | None,
    ) -> o3d.geometry.TriangleMesh | None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        if colors is not None and len(colors) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(
                colors.astype(np.float64) / 255.0
            )
        if normals is not None and len(normals) == len(pts):
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        # 다운샘플
        n_before = len(pcd.points)
        if self.voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        elif len(pcd.points) > self.max_points:
            idx = np.random.choice(len(pcd.points), self.max_points, replace=False)
            pcd = pcd.select_by_index(idx.tolist())
        n_after = len(pcd.points)
        if n_after < n_before:
            print(f"  다운샘플: {n_before:,} → {n_after:,} pts")

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        try:
            if self.method == "poisson":
                mesh, densities = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=self.poisson_depth
                    )
                )
                thresh = np.quantile(np.asarray(densities), 0.05)
                mesh.remove_vertices_by_mask(np.asarray(densities) < thresh)

            elif self.method == "bpa":
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd,
                    o3d.utility.DoubleVector(self.radii),
                )

            elif self.method == "alpha":
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )

            else:
                raise ValueError(f"알 수 없는 method: {self.method}")

            mesh.compute_vertex_normals()
            return mesh

        except Exception as e:
            print(f"메시 재구성 실패: {e}")
            return None
