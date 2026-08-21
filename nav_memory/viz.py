"""Memory visualization helpers: sphere markers and PLY overlay.

Used by ROS wrapper / offline tools for debug point clouds.
Not part of the SLAM core path.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .types import SID_DUST_BIN, SID_WALKED

# sid -> RGB (uint8), high-saturation for visibility on dense maps
MEMORY_COLORS = {
    SID_WALKED: np.array([0, 255, 255], dtype=np.uint8),  # cyan = walked
    SID_DUST_BIN: np.array([255, 0, 255], dtype=np.uint8),  # magenta = object
}
MEMORY_COLOR_DEFAULT = np.array([255, 255, 0], dtype=np.uint8)  # yellow = other
MEMORY_CENTER_COLOR = np.array([255, 40, 40], dtype=np.uint8)  # red = center


def fibonacci_sphere(n: int, radius: float, center: np.ndarray) -> np.ndarray:
    """Approximately uniform points on a sphere. Returns (n, 3) float32."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (i / max(n - 1, 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    pts = np.stack([x, y, z], axis=1) * float(radius) + center
    return pts.astype(np.float32)


def memory_marker_cloud(
    memory,
    shell_samples: int = 180,
    ring_samples: int = 48,
    center_samples: int = 24,
    kappa: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Materialize each circle as shell + equator ring + red core.

    Returns:
        points (N, 3) float32, colors (N, 3) uint8
    """
    rows = memory.export_centers_W() if memory is not None else []
    if not rows:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    all_pts = []
    all_cols = []
    for row in rows:
        mu = np.asarray(row["mu_W"], dtype=np.float32).reshape(3)
        radius_W = max(float(row.get("radius_W", 0.15)), 0.05)
        radius = kappa * radius_W
        sid = int(row.get("sid", 0))
        base = MEMORY_COLORS.get(sid, MEMORY_COLOR_DEFAULT)

        shell = fibonacci_sphere(shell_samples, radius, mu)
        all_pts.append(shell)
        all_cols.append(np.tile(base, (shell.shape[0], 1)))

        th = np.linspace(0, 2 * np.pi, ring_samples, endpoint=False)
        ring = np.stack(
            [
                mu[0] + radius * np.cos(th),
                np.full(ring_samples, mu[1], dtype=np.float64),
                mu[2] + radius * np.sin(th),
            ],
            axis=1,
        ).astype(np.float32)
        all_pts.append(ring)
        ring_col = np.clip(base.astype(np.int32) + 40, 0, 255).astype(np.uint8)
        all_cols.append(np.tile(ring_col, (ring.shape[0], 1)))

        core = fibonacci_sphere(center_samples, max(0.03, 0.25 * radius), mu)
        all_pts.append(core)
        all_cols.append(np.tile(MEMORY_CENTER_COLOR, (core.shape[0], 1)))

    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0)


def append_memory_to_ply(
    map_ply_path: Union[str, Path],
    out_ply_path: Union[str, Path],
    memory,
    voxel_size: Optional[float] = None,
    save_ply_fn=None,
) -> int:
    """
    Read map.ply, optionally voxel-downsample map only, append memory markers,
    write out_ply_path.

    Returns number of memory marker points written.
    """
    map_ply_path = Path(map_ply_path)
    out_ply_path = Path(out_ply_path)
    mem_pts, mem_cols = memory_marker_cloud(memory)

    if mem_pts.shape[0] == 0:
        if voxel_size is None:
            if map_ply_path.resolve() != out_ply_path.resolve():
                shutil.copy2(map_ply_path, out_ply_path)
            return 0
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(map_ply_path))
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
            o3d.io.write_point_cloud(str(out_ply_path), pcd)
        except Exception:
            shutil.copy2(map_ply_path, out_ply_path)
        return 0

    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(map_ply_path))
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        map_pts = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            map_cols = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            map_cols = np.full((map_pts.shape[0], 3), 180, dtype=np.uint8)

        pts = np.vstack([map_pts, mem_pts.astype(np.float64)])
        cols = np.vstack([map_cols, mem_cols])
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts)
        out.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(out_ply_path), out)
        return int(mem_pts.shape[0])
    except Exception:
        if save_ply_fn is not None:
            save_ply_fn(out_ply_path, mem_pts, mem_cols)
        else:
            # minimal ASCII PLY fallback
            _write_ply_ascii(out_ply_path, mem_pts, mem_cols)
        return int(mem_pts.shape[0])


def _write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    n = int(points.shape[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
