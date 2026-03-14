import os
from typing import Literal

import numpy as np
import open3d as o3d
import trimesh

from .point_labeler import LabeledCloud


class MeshReconstructor:
    def __init__(
        self,
        method: Literal["poisson", "bpa", "alpha"] = "poisson",
        depth: int = 9,
        radii: list[float] | None = None,
        voxel_size: float | None = None,    # None이면 max_points 기반 random sampling
        max_points: int = 150_000,          # 라벨당 최대 포인트 수
    ):
        self.method = method
        self.poisson_depth = depth
        self.radii = radii or [0.005, 0.01, 0.02, 0.04]
        self.voxel_size = voxel_size
        self.max_points = max_points

    def reconstruct(
        self,
        cloud: LabeledCloud,
    ) -> dict[str, o3d.geometry.TriangleMesh]:
        meshes: dict[str, o3d.geometry.TriangleMesh] = {}

        for label, pts in cloud.points.items():
            if len(pts) < 10:
                print(f"[{label}] 점군이 너무 작음 ({len(pts)}개), 스킵")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

            cols = cloud.colors[label]
            if cols is not None and len(cols) == len(pts):
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

            # 법선 설정
            nrm = cloud.normals.get(label)
            if nrm is not None and len(nrm) == len(pts):
                pcd.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))

            # Voxel downsampling — 고밀도 점군의 Poisson 재구성 속도 개선
            n_before = len(pcd.points)
            if self.voxel_size is not None:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            elif len(pcd.points) > self.max_points:
                # voxel_size 미지정 시 uniform random sampling으로 cap
                idx = np.random.choice(len(pcd.points), self.max_points, replace=False)
                pcd = pcd.select_by_index(idx.tolist())
            n_after = len(pcd.points)
            if n_after < n_before:
                print(f"[{label}] 다운샘플: {n_before:,} → {n_after:,} pts")

            # 법선이 없으면 (다운샘플 후) 재추정
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(100)

            mesh = self._reconstruct_one(pcd)
            if mesh is not None:
                meshes[label] = mesh
                print(f"[{label}] 메시: {len(mesh.vertices)}v / {len(mesh.triangles)}t")

        return meshes

    def _reconstruct_one(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh | None:
        try:
            if self.method == "poisson":
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=self.poisson_depth
                )
                # 밀도 낮은 삼각형 제거
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
                    # trimesh 경유 GLB/OBJ 저장
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
