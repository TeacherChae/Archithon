import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .point_labeler import LabeledCloud


@dataclass
class MeshData:
    vertices: np.ndarray   # (V, 3) float32
    faces: np.ndarray      # (F, 3) int32
    normals: np.ndarray    # (V, 3) float32 — empty array if unavailable
    colors: np.ndarray     # (V, 3) uint8   — empty array if unavailable

    @property
    def has_colors(self) -> bool:
        return len(self.colors) == len(self.vertices) > 0

    @property
    def has_normals(self) -> bool:
        return len(self.normals) == len(self.vertices) > 0


class MeshReconstructor:
    def __init__(
        self,
        method: Literal["voxel_mc", "poisson", "bpa", "alpha"] = "voxel_mc",
        depth: int = 9,                    # poisson 전용
        radii: list[float] | None = None,  # bpa 전용
        voxel_size: float | None = None,
        max_points: int = 150_000,
        voxel_resolution: int = 128,       # voxel_mc 전용
        device: str = "cuda",
    ):
        self.method = method
        self.poisson_depth = depth
        self.radii = radii or [0.005, 0.01, 0.02, 0.04]
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.voxel_resolution = voxel_resolution
        self.device = device

    # ── public ─────────────────────────────────────────────────────

    def reconstruct(self, cloud: LabeledCloud) -> dict[str, MeshData]:
        meshes: dict[str, MeshData] = {}

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
                print(f"[{label}] 메시: {len(mesh.vertices)}v / {len(mesh.faces)}t")

        return meshes

    def save(
        self,
        meshes: dict[str, MeshData],
        output_dir: str,
        formats: list[str] | None = None,
    ) -> dict[str, list[str]]:
        if formats is None:
            formats = ["glb", "ply"]

        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, list[str]] = {}

        # 3dm은 레이블 전체를 하나의 파일로 — 별도 처리
        non_3dm = [f for f in formats if f != "3dm"]
        has_3dm = "3dm" in formats

        for label, mesh in meshes.items():
            paths = []
            for fmt in non_3dm:
                path = os.path.join(output_dir, f"{label}.{fmt}")
                self._write_mesh(mesh, path, fmt)
                paths.append(path)
            saved[label] = paths

        # 전체 합친 메시
        if len(meshes) > 1 and non_3dm:
            combined = self._merge_meshes(meshes)
            paths = []
            for fmt in non_3dm:
                path = os.path.join(output_dir, f"combined.{fmt}")
                self._write_mesh(combined, path, fmt)
                paths.append(path)
            saved["combined"] = paths
            print(f"[combined] {len(combined.vertices)}v / {len(combined.faces)}t")

        # 레이어 분리된 Rhino 파일
        if has_3dm:
            path_3dm = os.path.join(output_dir, "scene.3dm")
            self._write_3dm(meshes, path_3dm)
            saved["scene_3dm"] = [path_3dm]

        return saved

    # ── merge ───────────────────────────────────────────────────────

    def _merge_meshes(self, meshes: dict[str, MeshData]) -> MeshData:
        all_verts, all_faces, all_normals, all_colors = [], [], [], []
        v_offset = 0

        for mesh in meshes.values():
            all_verts.append(mesh.vertices)
            all_faces.append(mesh.faces + v_offset)
            all_normals.append(mesh.normals if mesh.has_normals else None)
            all_colors.append(mesh.colors if mesh.has_colors else None)
            v_offset += len(mesh.vertices)

        merged_verts = np.concatenate(all_verts, axis=0)
        merged_faces = np.concatenate(all_faces, axis=0)

        total_v = len(merged_verts)
        valid_normals = [n for n in all_normals if n is not None]
        merged_normals = (
            np.concatenate(valid_normals, axis=0)
            if valid_normals and sum(len(n) for n in valid_normals) == total_v
            else np.zeros((0, 3), dtype=np.float32)
        )
        valid_colors = [c for c in all_colors if c is not None]
        merged_colors = (
            np.concatenate(valid_colors, axis=0)
            if valid_colors and sum(len(c) for c in valid_colors) == total_v
            else np.zeros((0, 3), dtype=np.uint8)
        )

        return MeshData(
            vertices=merged_verts,
            faces=merged_faces,
            normals=merged_normals,
            colors=merged_colors,
        )

    # ── writers ─────────────────────────────────────────────────────

    def _write_mesh(self, mesh: MeshData, path: str, fmt: str) -> None:
        if fmt == "ply":
            self._write_ply(mesh, path)
        elif fmt in ("glb", "obj"):
            self._write_trimesh(mesh, path)
        else:
            print(f"지원하지 않는 포맷: {fmt}")

    def _write_ply(self, mesh: MeshData, path: str) -> None:
        """float32 binary PLY — Open3D double 문제 없이 Rhino 호환"""
        V = len(mesh.vertices)
        F = len(mesh.faces)

        # vertex 구조체 dtype
        fields: list[tuple] = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        if mesh.has_normals:
            fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
        if mesh.has_colors:
            fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        vdata = np.zeros(V, dtype=np.dtype(fields))
        v32 = mesh.vertices.astype(np.float32)
        vdata['x'], vdata['y'], vdata['z'] = v32[:, 0], v32[:, 1], v32[:, 2]
        if mesh.has_normals:
            n32 = mesh.normals.astype(np.float32)
            vdata['nx'], vdata['ny'], vdata['nz'] = n32[:, 0], n32[:, 1], n32[:, 2]
        if mesh.has_colors:
            c8 = mesh.colors.astype(np.uint8)
            vdata['red'], vdata['green'], vdata['blue'] = c8[:, 0], c8[:, 1], c8[:, 2]

        # face 구조체: count(u8) + 3×i32
        fdata = np.zeros(
            F,
            dtype=np.dtype([('n', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')]),
        )
        fdata['n'] = 3
        f32 = mesh.faces.astype(np.int32)
        fdata['v0'], fdata['v1'], fdata['v2'] = f32[:, 0], f32[:, 1], f32[:, 2]

        # header — normals 제외 (Rhino PLY importer 호환성)
        hlines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment Archithon",
            f"element vertex {V}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if mesh.has_colors:
            hlines += ["property uchar red", "property uchar green", "property uchar blue"]
        hlines += [f"element face {F}", "property list uchar int vertex_indices", "end_header", ""]

        with open(path, 'wb') as f:
            f.write("\n".join(hlines).encode('ascii'))
            f.write(vdata.tobytes())
            f.write(fdata.tobytes())

    def _write_trimesh(self, mesh: MeshData, path: str) -> None:
        import trimesh
        vc = mesh.colors if mesh.has_colors else None
        tm = trimesh.Trimesh(
            vertices=mesh.vertices.astype(np.float64),
            faces=mesh.faces,
            vertex_colors=vc,
            process=False,
        )
        trimesh.repair.fill_holes(tm)
        tm.export(path)

    def _write_3dm(self, meshes: dict[str, MeshData], path: str) -> None:
        """rhino3dm을 별도 서브프로세스에서 실행 — CUDA allocator 충돌 방지"""
        import subprocess
        import sys
        import tempfile
        import pickle

        data = {
            label: {
                "vertices": mesh.vertices,
                "faces": mesh.faces,
                "colors": mesh.colors if mesh.has_colors else np.zeros((0, 3), dtype=np.uint8),
            }
            for label, mesh in meshes.items()
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"meshes": data, "path": path}, f)
            tmp_path = f.name

        script = """
import pickle, sys, numpy as np
try:
    import rhino3dm
except ImportError:
    print("rhino3dm 없음")
    sys.exit(1)

with open(sys.argv[1], "rb") as f:
    payload = pickle.load(f)

meshes = payload["meshes"]
out_path = payload["path"]

model = rhino3dm.File3dm()
for label, md in meshes.items():
    layer = rhino3dm.Layer()
    layer.Name = label
    layer_idx = model.Layers.Add(layer)

    r_mesh = rhino3dm.Mesh()
    for v in md["vertices"]:
        r_mesh.Vertices.Add(float(v[0]), float(v[1]), float(v[2]))
    for face in md["faces"]:
        r_mesh.Faces.AddFace(int(face[0]), int(face[1]), int(face[2]))
    if len(md["colors"]) > 0:
        for c in md["colors"]:
            r_mesh.VertexColors.Add(int(c[0]), int(c[1]), int(c[2]))
    r_mesh.Normals.ComputeNormals()

    attr = rhino3dm.ObjectAttributes()
    attr.LayerIndex = layer_idx
    attr.Name = label
    model.Objects.AddMesh(r_mesh, attr)

model.Write(out_path, 7)
print(f"[3dm] 저장 완료: {out_path}  ({len(meshes)} 레이어)")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, tmp_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"[3dm] 실패:\n{result.stderr.strip()}")
            else:
                print(result.stdout.strip())
        finally:
            os.unlink(tmp_path)

    # ── voxel_mc: GPU 복셀화 + CPU marching cubes ───────────────────

    def _voxel_mc(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> MeshData | None:
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
        min_pt = pts_t.min(0).values
        max_pt = pts_t.max(0).values
        scale = (max_pt - min_pt).max()

        pts_norm = (pts_t - min_pt) / (scale + 1e-8)
        idx = (pts_norm * (res - 1)).long().clamp(0, res - 1)
        flat = idx[:, 0] * res * res + idx[:, 1] * res + idx[:, 2]

        # GPU 점유 격자
        grid = torch.zeros(res ** 3, device=dev)
        grid.scatter_(0, flat, 1.0)
        grid = grid.reshape(1, 1, res, res, res)

        # GPU 3D Gaussian smoothing
        for _ in range(3):
            grid = F.avg_pool3d(grid, kernel_size=3, stride=1, padding=1)

        grid_np = grid.squeeze().cpu().numpy()

        # GPU 색상 격자
        color_grid: np.ndarray | None = None
        if colors is not None and len(colors) == len(pts):
            cols_t = torch.tensor(colors, dtype=torch.float32, device=dev) / 255.0
            counts = torch.zeros(res ** 3, device=dev)
            color_acc = torch.zeros((res ** 3, 3), device=dev)
            counts.scatter_add_(0, flat, torch.ones(len(flat), device=dev))
            color_acc.scatter_add_(0, flat.unsqueeze(1).expand(-1, 3), cols_t)
            nz = counts > 0
            color_acc[nz] /= counts[nz].unsqueeze(1)
            color_grid = color_acc.reshape(res, res, res, 3).cpu().numpy()

        if grid_np.max() < 1e-6:
            print("복셀 격자가 비어 있음, 스킵")
            return None

        level = float(grid_np.max() * 0.1)
        verts, faces, normals, _ = marching_cubes(grid_np, level=level)

        scale_val = scale.cpu().item()
        min_np = min_pt.cpu().numpy()
        verts_world = (verts / (res - 1) * scale_val + min_np).astype(np.float32)

        vert_colors = np.zeros((0, 3), dtype=np.uint8)
        if color_grid is not None:
            vi = np.clip(verts.astype(int), 0, res - 1)
            vert_colors = (color_grid[vi[:, 0], vi[:, 1], vi[:, 2]] * 255).astype(np.uint8)

        return MeshData(
            vertices=verts_world,
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32),
            colors=vert_colors,
        )

    # ── open3d fallback (poisson / bpa / alpha) ─────────────────────

    def _open3d_reconstruct(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None,
        normals: np.ndarray | None,
    ) -> MeshData | None:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if colors is not None and len(colors) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        if normals is not None and len(normals) == len(pts):
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        n_before = len(pcd.points)
        if self.voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        elif len(pcd.points) > self.max_points:
            idx = np.random.choice(len(pcd.points), self.max_points, replace=False)
            pcd = pcd.select_by_index(idx.tolist())
        if len(pcd.points) < n_before:
            print(f"  다운샘플: {n_before:,} → {len(pcd.points):,} pts")

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        try:
            if self.method == "poisson":
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=self.poisson_depth
                )
                thresh = np.quantile(np.asarray(densities), 0.05)
                mesh.remove_vertices_by_mask(np.asarray(densities) < thresh)
            elif self.method == "bpa":
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(self.radii)
                )
            elif self.method == "alpha":
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )
            else:
                raise ValueError(f"알 수 없는 method: {self.method}")
            mesh.compute_vertex_normals()
        except Exception as e:
            print(f"메시 재구성 실패: {e}")
            return None

        v = np.asarray(mesh.vertices).astype(np.float32)
        f = np.asarray(mesh.triangles).astype(np.int32)
        n = np.asarray(mesh.vertex_normals).astype(np.float32)
        c_raw = np.asarray(mesh.vertex_colors)
        c = (c_raw * 255).astype(np.uint8) if len(c_raw) == len(v) else np.zeros((0, 3), dtype=np.uint8)

        return MeshData(vertices=v, faces=f, normals=n, colors=c)
