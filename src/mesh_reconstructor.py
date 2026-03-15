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
        method: Literal["range_image", "gpu_poisson", "voxel_mc", "poisson", "bpa", "alpha"] = "range_image",
        depth: int = 9,                        # poisson / gpu_poisson 전용
        radii: list[float] | None = None,      # bpa 전용
        voxel_size: float | None = None,
        max_points: int = 150_000,
        voxel_resolution: int = 128,           # voxel_mc 전용
        depth_discontinuity: float = 0.05,     # range_image 전용: 깊이 불연속 임계 비율
        device: str = "cuda",
    ):
        self.method = method
        self.poisson_depth = depth
        self.radii = radii or [0.005, 0.01, 0.02, 0.04]
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.voxel_resolution = voxel_resolution
        self.depth_discontinuity = depth_discontinuity
        self.device = device

    # ── public ─────────────────────────────────────────────────────

    def reconstruct_structured(
        self,
        points: np.ndarray,          # (H, W, 3) depth.points — 카메라 좌표계
        valid_mask: np.ndarray,      # (H, W) bool depth.mask — MoGe 유효 픽셀
        masks: list,                 # list of (H, W) bool — SAM 레이블별 마스크
        labels: list,                # list of str
        colors: np.ndarray,          # (H, W, 3) uint8 image.rgb
        normals: np.ndarray | None,  # (H, W, 3) depth.normal
    ) -> dict[str, MeshData]:
        """
        구조화 포인트맵(H,W,3)에서 직접 메시 생성 — Poisson/Voxel 없음.

        MoGe 출력이 이미 픽셀 격자 구조이므로 인접 픽셀을 삼각형으로 연결.
        100% GPU (torch), O(H×W).
        """
        meshes: dict[str, MeshData] = {}

        for label, mask in zip(labels, masks):
            combined = mask & valid_mask
            if combined.sum() < 10:
                print(f"[{label}] 유효 픽셀 부족 ({combined.sum()}px), 스킵")
                continue

            mesh = self._range_image_mesh(
                points=points,
                mask=combined,
                colors=colors,
                normals=normals,
            )

            if mesh is not None:
                meshes[label] = mesh
                print(f"[{label}] 메시: {len(mesh.vertices):,}v / {len(mesh.faces):,}t")

        return meshes

    def reconstruct(self, cloud: LabeledCloud) -> dict[str, MeshData]:
        meshes: dict[str, MeshData] = {}

        for label, pts in cloud.points.items():
            if len(pts) < 10:
                print(f"[{label}] 점군이 너무 작음 ({len(pts)}개), 스킵")
                continue

            cols = cloud.colors.get(label)
            nrm = cloud.normals.get(label)

            if self.method == "gpu_poisson":
                mesh = self._gpu_poisson(pts, cols, nrm)
            elif self.method == "voxel_mc":
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

    # ── range_image: 픽셀 격자 → 삼각형 (100% GPU) ─────────────────

    def _range_image_mesh(
        self,
        points: np.ndarray,          # (H, W, 3) 카메라 좌표계
        mask: np.ndarray,            # (H, W) bool — 이 레이블의 유효 픽셀
        colors: np.ndarray,          # (H, W, 3) uint8
        normals: np.ndarray | None,  # (H, W, 3)
    ) -> "MeshData | None":
        """
        인접 픽셀 4개를 쿼드(삼각형 2개)로 연결.

        CUDA 연산:
          - 쿼드 유효성 검사 (마스크 AND, 깊이 불연속 필터)
          - 정점 인덱스 재매핑 (unique + inverse)
          - 색상·법선 인덱싱
        CPU:
          - 없음 (최종 numpy 변환만)

        깊이 불연속 필터:
          쿼드 내 최대·최소 Z 차이가
          mean(Z) * depth_discontinuity_ratio 초과 시 삼각형 생성 스킵.
          → 객체 경계·오클루전 부분에서 늘어난 삼각형 방지.
        """
        import torch

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        H, W = mask.shape

        pts_t   = torch.tensor(points,  dtype=torch.float32, device=device)  # (H, W, 3)
        mask_t  = torch.tensor(mask,    dtype=torch.bool,    device=device)  # (H, W)
        depth_t = pts_t[..., 2]                                               # (H, W) Z값

        # ── 쿼드 4 꼭짓점 마스크 ─────────────────────────────────
        # 쿼드 (i,j): 좌상=(i,j), 좌하=(i+1,j), 우상=(i,j+1), 우하=(i+1,j+1)
        m00 = mask_t[:-1, :-1]   # (H-1, W-1)
        m10 = mask_t[1:,  :-1]
        m01 = mask_t[:-1, 1:]
        m11 = mask_t[1:,  1:]
        quad_valid = m00 & m10 & m01 & m11

        # ── 깊이 불연속 필터 ─────────────────────────────────────
        d00 = depth_t[:-1, :-1]
        d10 = depth_t[1:,  :-1]
        d01 = depth_t[:-1, 1:]
        d11 = depth_t[1:,  1:]

        stacked   = torch.stack([d00, d10, d01, d11], dim=0)
        depth_rng = stacked.max(0).values - stacked.min(0).values
        mean_d    = stacked.mean(0).abs().clamp(min=1e-6)
        quad_valid = quad_valid & (depth_rng / mean_d < self.depth_discontinuity)

        # ── 유효 쿼드 인덱스 → 평탄화 픽셀 인덱스 ───────────────
        qi = quad_valid.nonzero(as_tuple=False)   # (Q, 2): (row, col)
        if len(qi) == 0:
            return None

        rows, cols = qi[:, 0], qi[:, 1]
        idx00 = rows * W + cols            # (Q,)
        idx10 = (rows + 1) * W + cols
        idx01 = rows * W + (cols + 1)
        idx11 = (rows + 1) * W + (cols + 1)

        # 삼각형 와인딩 (카메라 좌표계 Z-forward, 법선이 카메라를 향하도록)
        # tri A: (i,j) → (i+1,j) → (i,j+1)
        # tri B: (i+1,j+1) → (i,j+1) → (i+1,j)
        tri_a = torch.stack([idx00, idx10, idx01], dim=1)   # (Q, 3)
        tri_b = torch.stack([idx11, idx01, idx10], dim=1)   # (Q, 3)
        faces_flat = torch.cat([tri_a, tri_b], dim=0)       # (2Q, 3)

        # ── 사용된 픽셀만 정점으로 추출 ──────────────────────────
        unique_px, inverse = torch.unique(faces_flat.reshape(-1), return_inverse=True)
        # unique_px: (V,) 픽셀 인덱스 / inverse: (2Q*3,) 새 인덱스

        pts_flat  = pts_t.reshape(-1, 3)
        verts     = pts_flat[unique_px]                        # (V, 3)
        faces_new = inverse.reshape(-1, 3).to(torch.int32)    # (2Q, 3)

        # 색상
        cols_t   = torch.tensor(colors, dtype=torch.uint8, device=device).reshape(-1, 3)
        v_colors = cols_t[unique_px]                           # (V, 3) uint8

        # 법선
        if normals is not None:
            nrm_t   = torch.tensor(normals, dtype=torch.float32, device=device).reshape(-1, 3)
            v_norms = nrm_t[unique_px]                         # (V, 3)
        else:
            v_norms = torch.zeros(0, 3, device=device)

        return MeshData(
            vertices=verts.cpu().numpy().astype(np.float32),
            faces=faces_new.cpu().numpy(),
            normals=v_norms.cpu().numpy().astype(np.float32),
            colors=v_colors.cpu().numpy(),
        )

    # ── gpu_poisson: CUDA 전처리 + CPU Screened Poisson ────────────

    def _gpu_poisson(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None,
        normals: np.ndarray | None,
    ) -> "MeshData | None":
        """
        CUDA 전처리 + CPU Screened Poisson 재구성.

        CUDA 사용:
          - 포인트 서브샘플링 (torch.randperm on GPU)
          - 법선 카메라 방향 정렬 (torch 벡터 연산 on GPU)
        CPU:
          - Open3D Screened Poisson solver
          MoGe-2-vitl-normal이 법선을 이미 출력하므로
          estimate_normals() 생략 → Poisson 에서 가장 느린 단계 제거.
        """
        import torch
        import open3d as o3d

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
        N = len(pts_t)

        # ── 1. GPU 서브샘플링 ──────────────────────────────────────
        if N > self.max_points:
            idx = torch.randperm(N, device=device)[: self.max_points]
            idx_np = idx.cpu().numpy()
            pts_t = pts_t[idx]
            colors = colors[idx_np] if colors is not None else None
            normals = normals[idx_np] if normals is not None else None
            print(f"  GPU 서브샘플: {N:,} → {len(pts_t):,} pts")

        # ── 2. GPU 법선 카메라 방향 정렬 ──────────────────────────
        #    MoGe는 카메라 좌표계 출력: 카메라 = 원점
        #    dot(normal, -point) < 0  →  법선이 카메라 반대 방향 → flip
        if normals is not None:
            nrm_t = torch.tensor(normals, dtype=torch.float32, device=device)
            flip_mask = (nrm_t * pts_t).sum(dim=1) > 0  # dot(n, pt) > 0 → 뒤집기
            nrm_t[flip_mask] = -nrm_t[flip_mask]
            normals = nrm_t.cpu().numpy()

        pts_cpu = pts_t.cpu().numpy().astype(np.float64)

        # ── 3. CPU Screened Poisson (법선 외부 제공 → estimate 생략) ──
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cpu)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        else:
            # MoGe normal 없는 경우 fallback (느림)
            print("  MoGe 법선 없음 — CPU estimate_normals() 실행 (느릴 수 있음)")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        if colors is not None and len(colors) == len(pts_cpu):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.poisson_depth, linear_fit=False
            )
            # 밀도 낮은 부동 삼각형 제거
            dens = np.asarray(densities)
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
        except Exception as e:
            print(f"  Poisson 실패: {e}")
            return None

        mesh.compute_vertex_normals()
        v = np.asarray(mesh.vertices).astype(np.float32)
        f = np.asarray(mesh.triangles).astype(np.int32)
        n = np.asarray(mesh.vertex_normals).astype(np.float32)
        c_raw = np.asarray(mesh.vertex_colors)
        c = (c_raw * 255).astype(np.uint8) if len(c_raw) == len(v) else np.zeros((0, 3), dtype=np.uint8)

        return MeshData(vertices=v, faces=f, normals=n, colors=c)

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
