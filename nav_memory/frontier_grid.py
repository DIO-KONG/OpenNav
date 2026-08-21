"""Grid-based nearest frontier: local FREE \\ WALKED, geodesic BFS.

Query-time only (no persistent grid). Walked circles are rasterized as
disks (radius kappa*radius_W) or squares on the X-Z navigation plane.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .types import SID_WALKED

# 几何/阈值统一来源: nav_constants.py (ROBOT_RADIUS 派生, 与 nav 联动)
from nav_constants import (
    MEM_FRONTIER_ROBOT_RADIUS,
    MEM_FRONTIER_RESOLUTION,
    MEM_FRONTIER_WIN_RADIUS,
    MEM_FRONTIER_OCC_MIN_POINTS,
    MEM_FRONTIER_KAPPA,
    MEM_FRONTIER_SHAPE,
    MEM_FRONTIER_INFLATE_TYPE,
    MEM_FRONTIER_SIGMA_WALKED,
    MEM_FRONTIER_GLOBAL_RESOLUTION,
    PRETURN_WALL_RAY_MIN_DEG,
    PRETURN_WALL_RAY_MAX_DEG,
    PRETURN_WALL_RAY_STEP_DEG,
    PRETURN_WALL_RAY_MAX_DIST,
    PRETURN_WALL_MIN_HIT_RATIO,
    PRETURN_WALL_MAX_MEDIAN_DIST,
    PRETURN_WALL_MAX_LINE_RESIDUAL,
)

if TYPE_CHECKING:
    from .system import MemorySystem


@dataclass
class GridFrontierResult:
    found: bool
    xz: Optional[Tuple[float, float]] = None  # world (x, z)
    dist_m: float = 0.0
    message: str = ""
    search_type: str = "local"  # "local" or "global"
    elapsed_ms: float = 0.0
    n_cells: int = 0
    n_walked: int = 0
    # Per-step timing (ms), for perf debug
    t_occ_ms: float = 0.0      # obstacle project + inflate
    t_walked_ms: float = 0.0   # cand + paint (sum)
    t_cand_ms: float = 0.0     # candidates_in_ball / world_params (T_WC)
    t_paint_ms: float = 0.0    # disk/square raster + UNEXP mask
    t_bfs_ms: float = 0.0      # start + BFS
    # Grid debug: boolean arrays and geo params for overlay rendering
    occ_inf: Optional[object] = None   # (n, n) bool — inflated OCC
    walked_grid: Optional[object] = None  # (n, n) bool — WALKED cells
    grid_n: int = 0
    grid_origin_x: float = 0.0
    grid_origin_z: float = 0.0
    grid_res: float = MEM_FRONTIER_RESOLUTION


@dataclass(frozen=True)
class RawOccGrid:
    """未膨胀的局部 OCC 栅格及其世界坐标参数。"""

    occ: np.ndarray
    n: int
    origin_x: float
    origin_z: float
    resolution: float


@dataclass(frozen=True)
class FrontWallResult:
    """前方墙状 OCC 检测的结构化结果。"""

    detected: bool
    n_hits: int
    n_rays: int
    hit_ratio: float
    median_distance: float
    line_residual: float


def _world_to_cell(
    x: float, z: float, origin_x: float, origin_z: float, resolution: float, n: int
) -> Tuple[int, int]:
    """Map world (x,z) to grid indices (ix, iz). May be out of [0, n)."""
    ix = int(np.floor((x - origin_x) / resolution))
    iz = int(np.floor((z - origin_z) / resolution))
    return ix, iz


def _cell_center(
    ix: int, iz: int, origin_x: float, origin_z: float, resolution: float
) -> Tuple[float, float]:
    return (
        origin_x + (ix + 0.5) * resolution,
        origin_z + (iz + 0.5) * resolution,
    )


def build_raw_occ_grid(
    pose_xz,
    obstacle_points: Optional[np.ndarray],
    *,
    resolution: float = MEM_FRONTIER_RESOLUTION,
    win_radius: float = MEM_FRONTIER_WIN_RADIUS,
    min_points: int = MEM_FRONTIER_OCC_MIN_POINTS,
) -> Optional[RawOccGrid]:
    """把障碍点投影为机器人周围的未膨胀 OCC 栅格。

    Frontier 和近墙检测共用此入口，保证两者对原始 OCC 的定义一致。
    """
    pose = np.asarray(pose_xz, dtype=np.float64).reshape(-1)
    if pose.size < 2 or not np.all(np.isfinite(pose[:2])):
        return None
    px, pz = float(pose[0]), float(pose[1])
    res = float(resolution)
    wr = float(win_radius)
    min_pts = int(min_points)
    if (not np.isfinite(res) or not np.isfinite(wr) or
            res <= 0.0 or wr <= 0.0 or min_pts <= 0):
        return None

    n = max(int(np.floor(2.0 * wr / res)) + 1, 3)
    origin_x = px - wr
    origin_z = pz - wr
    occ = np.zeros((n, n), dtype=bool)

    if obstacle_points is not None:
        pts = np.asarray(obstacle_points, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 3:
            xs = pts[:, 0]
            zs = pts[:, 2]
            finite = np.isfinite(xs) & np.isfinite(zs)
            xs = xs[finite]
            zs = zs[finite]
            if xs.size > 0:
                ixs = np.floor((xs - origin_x) / res).astype(np.int32)
                izs = np.floor((zs - origin_z) / res).astype(np.int32)
                inside = (ixs >= 0) & (ixs < n) & (izs >= 0) & (izs < n)
                ixs = ixs[inside]
                izs = izs[inside]
                if ixs.size > 0:
                    flat = ixs * n + izs
                    counts = np.bincount(flat, minlength=n * n)
                    occ = counts.reshape(n, n) >= min_pts

    return RawOccGrid(
        occ=occ,
        n=n,
        origin_x=origin_x,
        origin_z=origin_z,
        resolution=res,
    )


def _first_ray_occ_hit(
    grid: RawOccGrid,
    px: float,
    pz: float,
    angle: float,
    max_distance: float,
) -> Optional[Tuple[float, float, float]]:
    """以 DDA 遍历一条射线，返回首个 OCC 格中心的 (距离, x, z)。"""
    dx = float(np.sin(angle))
    dz = float(np.cos(angle))
    ix, iz = _world_to_cell(
        px, pz, grid.origin_x, grid.origin_z, grid.resolution, grid.n)
    if ix < 0 or ix >= grid.n or iz < 0 or iz >= grid.n:
        return None

    step_x = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    step_z = 1 if dz > 0.0 else (-1 if dz < 0.0 else 0)
    inf = float("inf")
    if step_x:
        boundary_x = grid.origin_x + (ix + (1 if step_x > 0 else 0)) * grid.resolution
        t_max_x = (boundary_x - px) / dx
        t_delta_x = grid.resolution / abs(dx)
    else:
        t_max_x = inf
        t_delta_x = inf
    if step_z:
        boundary_z = grid.origin_z + (iz + (1 if step_z > 0 else 0)) * grid.resolution
        t_max_z = (boundary_z - pz) / dz
        t_delta_z = grid.resolution / abs(dz)
    else:
        t_max_z = inf
        t_delta_z = inf

    while True:
        t_enter = min(t_max_x, t_max_z)
        if not np.isfinite(t_enter) or t_enter > max_distance:
            return None
        if abs(t_max_x - t_max_z) <= 1e-12:
            ix += step_x
            iz += step_z
            t_max_x += t_delta_x
            t_max_z += t_delta_z
        elif t_max_x < t_max_z:
            ix += step_x
            t_max_x += t_delta_x
        else:
            iz += step_z
            t_max_z += t_delta_z
        if ix < 0 or ix >= grid.n or iz < 0 or iz >= grid.n:
            return None
        if grid.occ[ix, iz]:
            hx, hz = _cell_center(
                ix, iz, grid.origin_x, grid.origin_z, grid.resolution)
            dist = float(np.hypot(hx - px, hz - pz))
            if dist <= max_distance + grid.resolution:
                return dist, hx, hz
            return None


def detect_front_wall(
    pose_xz,
    yaw: float,
    obstacle_points: Optional[np.ndarray],
    *,
    resolution: float = MEM_FRONTIER_RESOLUTION,
    ray_min_deg: int = PRETURN_WALL_RAY_MIN_DEG,
    ray_max_deg: int = PRETURN_WALL_RAY_MAX_DEG,
    ray_step_deg: int = PRETURN_WALL_RAY_STEP_DEG,
    ray_max_distance: float = PRETURN_WALL_RAY_MAX_DIST,
    min_hit_ratio: float = PRETURN_WALL_MIN_HIT_RATIO,
    max_median_distance: float = PRETURN_WALL_MAX_MEDIAN_DIST,
    max_line_residual: float = PRETURN_WALL_MAX_LINE_RESIDUAL,
) -> FrontWallResult:
    """用首命中比例、距离中位数和 PCA 直线残差判断前方墙面。"""
    invalid = FrontWallResult(False, 0, 0, 0.0, float("inf"), float("inf"))
    try:
        px, pz = float(pose_xz[0]), float(pose_xz[1])
        yaw_f = float(yaw)
        step = int(ray_step_deg)
        max_dist = float(ray_max_distance)
    except (TypeError, ValueError, IndexError):
        return invalid
    if (not np.all(np.isfinite([px, pz, yaw_f, max_dist])) or
            step <= 0 or int(ray_min_deg) > int(ray_max_deg) or max_dist <= 0.0):
        return invalid

    ray_degrees = range(int(ray_min_deg), int(ray_max_deg) + 1, step)
    n_rays = len(ray_degrees)
    if n_rays <= 0:
        return invalid
    grid = build_raw_occ_grid(
        (px, pz), obstacle_points,
        resolution=resolution,
        win_radius=max_dist + float(resolution),
    )
    if grid is None:
        return FrontWallResult(False, 0, n_rays, 0.0, float("inf"), float("inf"))

    distances = []
    hit_points = []
    for rel_deg in ray_degrees:
        hit = _first_ray_occ_hit(
            grid, px, pz, yaw_f + np.deg2rad(rel_deg), max_dist)
        if hit is not None:
            distances.append(hit[0])
            hit_points.append((hit[1], hit[2]))

    n_hits = len(distances)
    hit_ratio = float(n_hits) / float(n_rays)
    median_distance = (float(np.median(distances))
                       if distances else float("inf"))
    line_residual = float("inf")
    if n_hits >= 2:
        pts = np.asarray(hit_points, dtype=np.float64)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        covariance = centered.T @ centered / float(n_hits)
        try:
            _, eigenvectors = np.linalg.eigh(covariance)
            normal = eigenvectors[:, 0]
            line_residual = float(np.median(np.abs(centered @ normal)))
        except np.linalg.LinAlgError:
            line_residual = float("inf")

    metrics_finite = np.all(np.isfinite(
        [hit_ratio, median_distance, line_residual]))
    detected = bool(
        n_hits >= 2 and metrics_finite and
        hit_ratio >= float(min_hit_ratio) and
        median_distance <= float(max_median_distance) and
        line_residual <= float(max_line_residual)
    )
    return FrontWallResult(
        detected=detected,
        n_hits=n_hits,
        n_rays=n_rays,
        hit_ratio=hit_ratio,
        median_distance=median_distance,
        line_residual=line_residual,
    )



def _inflate_square(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    """Square structural-element dilation of boolean OCC."""
    if radius_cells <= 0:
        return occ
    h, w = occ.shape
    # Integral image / max-pool via cumulative OR on expanded pad
    pad = radius_cells
    padded = np.pad(occ, pad, mode="constant", constant_values=False)
    out = np.zeros_like(occ)
    # For each offset in square, OR — vectorized over offsets with striding
    # Use cumulative sum trick for speed on small grids
    integ = padded.astype(np.int32)
    integ = np.cumsum(np.cumsum(integ, axis=0), axis=1)
    # pad so integ has shape (h+2*pad+1, w+2*pad+1) after zero-row/col
    integ = np.pad(integ, ((1, 0), (1, 0)), mode="constant")
    # For each cell (i,j) in original, sum in square [i-r, i+r] x [j-r, j+r]
    # in padded coords: base = (i+pad, j+pad)
    r = radius_cells
    # indices in padded (before integ pad): i' = i + pad
    # integ is 1-based cumsum: integ[a,b] = sum of padded[0:a, 0:b]
    i0 = np.arange(h) + pad - r
    i1 = np.arange(h) + pad + r + 1  # exclusive in integ sense
    j0 = np.arange(w) + pad - r
    j1 = np.arange(w) + pad + r + 1
    # Broadcast: out[i,j] = sum over square
    # integ[i1, j1] - integ[i0, j1] - integ[i1, j0] + integ[i0, j0]
    ii0 = i0[:, None]
    ii1 = i1[:, None]
    jj0 = j0[None, :]
    jj1 = j1[None, :]
    s = integ[ii1, jj1] - integ[ii0, jj1] - integ[ii1, jj0] + integ[ii0, jj0]
    out = s > 0
    return out


def _inflate_circle(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    """Circular Euclidean dilation of boolean OCC grid.

    Each True cell in ``occ`` is expanded to a disk of radius ``radius_cells``
    cells (world robot_radius / resolution).  The disk includes the boundary:
    every cell whose centre lies within (radius_cells + 0.5) of the source cell.

    Complexity: O(N_occ · R²) per-OCC-cell loop with precomputed offset set.
    For typical grids (n≤121, occ≤20000 unique cells, R≤4 → ~80 offsets) this
    is dominated by the numpy boolean indexing and runs well under 1 ms.
    """
    if radius_cells <= 0:
        return occ
    h, w = occ.shape
    r = radius_cells
    r2_boundary = (r + 0.5) ** 2

    # Precompute disk offsets (dix, diz) within the inflated radius
    offsets: list = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj <= r2_boundary:
                offsets.append((di, dj))

    out = occ.copy()
    occ_ix, occ_iz = np.where(occ)
    for ix, iz in zip(occ_ix, occ_iz):
        for dix, diz in offsets:
            ni = ix + dix
            nj = iz + diz
            if 0 <= ni < h and 0 <= nj < w:
                out[ni, nj] = True
    return out


def _paint_disk(
    walked: np.ndarray,
    cx: float,
    cz: float,
    radius: float,
    origin_x: float,
    origin_z: float,
    resolution: float,
) -> None:
    """Fill disk of world radius into walked boolean grid (in-place)."""
    n = walked.shape[0]
    if radius <= 0:
        return
    r_cells = int(np.ceil(radius / resolution)) + 1
    ix0, iz0 = _world_to_cell(cx, cz, origin_x, origin_z, resolution, n)
    r2 = radius * radius
    for diz in range(-r_cells, r_cells + 1):
        iz = iz0 + diz
        if iz < 0 or iz >= n:
            continue
        for dix in range(-r_cells, r_cells + 1):
            ix = ix0 + dix
            if ix < 0 or ix >= n:
                continue
            wx, wz = _cell_center(ix, iz, origin_x, origin_z, resolution)
            dx = wx - cx
            dz = wz - cz
            if dx * dx + dz * dz <= r2:
                walked[ix, iz] = True


def _paint_square(
    walked: np.ndarray,
    cx: float,
    cz: float,
    half: float,
    origin_x: float,
    origin_z: float,
    resolution: float,
) -> None:
    """Fill axis-aligned square of half-side half into walked (in-place)."""
    n = walked.shape[0]
    if half <= 0:
        return
    r_cells = int(np.ceil(half / resolution)) + 1
    ix0, iz0 = _world_to_cell(cx, cz, origin_x, origin_z, resolution, n)
    for diz in range(-r_cells, r_cells + 1):
        iz = iz0 + diz
        if iz < 0 or iz >= n:
            continue
        for dix in range(-r_cells, r_cells + 1):
            ix = ix0 + dix
            if ix < 0 or ix >= n:
                continue
            wx, wz = _cell_center(ix, iz, origin_x, origin_z, resolution)
            if abs(wx - cx) <= half and abs(wz - cz) <= half:
                walked[ix, iz] = True


# 4-connected steps on (ix, iz) grid; world Δ matches (+x, +z).
_NEIGH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# 8-connected steps: fwd-biased for Dijkstra tie-break via _forward_neighbor_order.
_NEIGH8 = (
    (0, 1),   # forward  (world +Z)
    (-1, 1),  # left-fwd
    (1, 1),   # right-fwd
    (1, 0),   # right
    (-1, 0),  # left
    (-1, -1), # left-back
    (1, -1),  # right-back
    (0, -1),  # back
)

_SQRT2 = 1.4142135623730951


def _resolve_start_cell(
    free: np.ndarray,
    px: float,
    pz: float,
    origin_x: float,
    origin_z: float,
    res: float,
    n: int,
    stack_free_xz: list,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Resolve Dijkstra start without wall-jumping through OCC.

    - Robot on FREE: start = robot cell, snap=0.
    - Robot on OCC / OOB: inspect stack_free_xz from newest to oldest. Pop
      entries that are still OCC on this grid, then reuse the first FREE entry.
      An out-of-bounds entry is preserved for a possible larger/global grid.
    - Otherwise: no valid start (start=None).

    Returns (start_cell | None, snap_dist_m).
    """
    six, siz = _world_to_cell(px, pz, origin_x, origin_z, res, n)
    if 0 <= six < n and 0 <= siz < n and free[six, siz]:
        return (six, siz), 0.0

    while stack_free_xz:
        top = stack_free_xz[-1]
        if top is None or len(top) < 2:
            stack_free_xz.pop()
            continue
        lx, lz = float(top[0]), float(top[1])
        lix, liz = _world_to_cell(lx, lz, origin_x, origin_z, res, n)
        if not (0 <= lix < n and 0 <= liz < n):
            return None, 0.0
        if free[lix, liz]:
            scx, scz = _cell_center(lix, liz, origin_x, origin_z, res)
            snap = float(np.hypot(px - scx, pz - scz))
            return (lix, liz), snap
        stack_free_xz.pop()

    return None, 0.0


def _record_success_pose(memory: "MemorySystem", px: float, pz: float) -> None:
    """Push a successful query pose when it is over 0.2 m from stack top."""
    if memory is None:
        return
    stack = getattr(memory, "_stack_free_xz", None)
    if not isinstance(stack, list):
        stack = [(0.0, 0.0)]
        memory._stack_free_xz = stack
    pose = (float(px), float(pz))
    if not stack:
        stack.append(pose)
        return
    top = stack[-1]
    if top is None or len(top) < 2:
        stack.append(pose)
        return
    if float(np.hypot(pose[0] - float(top[0]), pose[1] - float(top[1]))) > 0.2:
        stack.append(pose)


def _forward_neighbor_order(yaw: Optional[float]):
    """
    8-邻格按与机器人前向对齐程度排序（对齐度高的优先入队）。

    导航平面约定 (与 nav_page / get_pose 一致):
      前向世界 (fx, fz) = (sin yaw, cos yaw)
    邻格 (dix, diz) 与前向点积越大越靠前。
    yaw is None 时保持固定顺序。
    """
    if yaw is None:
        return _NEIGH8
    fx = float(np.sin(yaw))
    fz = float(np.cos(yaw))
    return tuple(
        sorted(
            _NEIGH8,
            key=lambda d: -(fx * d[0] + fz * d[1]),
        )
    )


def _bfs_nearest_unexp(
    free: np.ndarray,
    unexp: np.ndarray,
    start: Tuple[int, int],
    neighbor_order=None,
    res: float = MEM_FRONTIER_RESOLUTION,
) -> Optional[Tuple[Tuple[int, int], float]]:
    """
    8-connected Dijkstra on FREE cells. First UNEXP reached wins.

    Orthogonal edges cost ``res``, diagonal edges cost ``sqrt(2) * res``.
    neighbor_order: 邻格 (dix,diz) 序列；前方优先时按前向点积降序。
    当累计距离相同时，先入堆的邻格优先（counter 打破平局），从而偏好前向 UNEXP。
    Returns ((ix, iz), accumulated_distance_m) or None.
    """
    n = free.shape[0]
    sx, sz = start
    if not free[sx, sz]:
        return None
    if unexp[sx, sz]:
        return (sx, sz), 0.0

    neigh = neighbor_order if neighbor_order is not None else _NEIGH8
    visited = np.zeros((n, n), dtype=bool)
    visited[sx, sz] = True

    # heap entries: (accumulated_distance_m, counter, cx, cz)
    # counter ensures stable tie-breaking among equal-distance entries
    heap: list = [(0.0, 0, sx, sz)]
    counter = 1

    while heap:
        acc_dist, _, cx, cz = heapq.heappop(heap)
        for dx, dz in neigh:
            nx, nz = cx + dx, cz + dz
            if nx < 0 or nx >= n or nz < 0 or nz >= n:
                continue
            if visited[nx, nz] or not free[nx, nz]:
                continue
            visited[nx, nz] = True

            # Orthogonal: |dx|+|dz| == 1; Diagonal: |dx|+|dz| == 2
            edge_cost = res if (abs(dx) + abs(dz) == 1) else _SQRT2 * res
            nd = acc_dist + edge_cost

            if unexp[nx, nz]:
                return (nx, nz), nd
            heapq.heappush(heap, (nd, counter, nx, nz))
            counter += 1
    return None


def get_nearest_frontier_grid(
    memory: "MemorySystem",
    pose_xz,
    *,
    yaw: Optional[float] = None,
    nav_y: float = 0.0,
    obstacle_points: Optional[np.ndarray] = None,
    robot_radius: float = MEM_FRONTIER_ROBOT_RADIUS,
    resolution: float = MEM_FRONTIER_RESOLUTION,
    win_radius: float = MEM_FRONTIER_WIN_RADIUS,
    kappa: float = MEM_FRONTIER_KAPPA,
    shape: str = MEM_FRONTIER_SHAPE,
    inflate_shape: str = MEM_FRONTIER_INFLATE_TYPE,
    sigma_walked: float = MEM_FRONTIER_SIGMA_WALKED,
    global_resolution: float = MEM_FRONTIER_GLOBAL_RESOLUTION,
) -> GridFrontierResult:
    """
    Local snapshot grid: FREE = ~inflate(obstacles), WALKED = kappa*sigma disks,
    UNEXP = FREE & ~WALKED. Geodesic nearest UNEXP via 8-connected Dijkstra on FREE.

    Global fallback is currently disabled. If no valid frontier is found in
    the local window, the query returns the local failure directly.

    Args:
        memory: MemorySystem with keyframes bound (for walked world params).
        pose_xz: (x, z) robot position on navigation plane.
        yaw: optional heading (rad). If set, BFS enqueues forward neighbors first
             so ties at equal geodesic distance prefer the robot's front.
        nav_y: height used only when building center_W for candidate query.
        obstacle_points: (N, 3) world points; only X,Z used. None => no OCC.
        robot_radius: inflation radius (m).
        resolution: cell size (m).
        win_radius: half-window size (m).
        kappa: walked disk radius multiplier.
        shape: "disk" or "square" — WALKED raster shape.
        inflate_shape: "circle" or "square" — OCC inflation kernel shape.
        sigma_walked: walked memory sigma (m). Used for square query half-side
                      = win_radius + 3*sigma_walked.
        global_resolution: cell size (m) for the global fallback grid.
    """
    t0 = time.perf_counter()
    t_occ_ms = 0.0
    t_walked_ms = 0.0
    t_cand_ms = 0.0
    t_paint_ms = 0.0
    t_bfs_ms = 0.0
    # Grid debug vars (populated after OCC + WALKED raster)
    occ_inf = None
    walked = None
    n = 0
    origin_x = 0.0
    origin_z = 0.0
    res = float(resolution)

    def _finish(**kwargs) -> GridFrontierResult:
        kwargs.setdefault("search_type", "local")
        kwargs.setdefault("elapsed_ms", (time.perf_counter() - t0) * 1000.0)
        kwargs.setdefault("t_occ_ms", t_occ_ms)
        kwargs.setdefault("t_walked_ms", t_walked_ms)
        kwargs.setdefault("t_cand_ms", t_cand_ms)
        kwargs.setdefault("t_paint_ms", t_paint_ms)
        kwargs.setdefault("t_bfs_ms", t_bfs_ms)
        kwargs.setdefault("occ_inf", occ_inf)
        kwargs.setdefault("walked_grid", walked)
        kwargs.setdefault("grid_n", n)
        kwargs.setdefault("grid_origin_x", origin_x)
        kwargs.setdefault("grid_origin_z", origin_z)
        kwargs.setdefault("grid_res", res)
        return GridFrontierResult(**kwargs)

    pose = np.asarray(pose_xz, dtype=np.float64).reshape(-1)
    if pose.size < 2:
        return _finish(found=False, message="invalid pose_xz")
    px, pz = float(pose[0]), float(pose[1])

    wr = float(win_radius)
    if res <= 0 or wr <= 0:
        return _finish(found=False, message="invalid resolution/win_radius")

    # --- OCC + inflate ---
    t_a = time.perf_counter()
    raw_occ = build_raw_occ_grid(
        (px, pz), obstacle_points,
        resolution=res,
        win_radius=wr,
    )
    if raw_occ is None:
        return _finish(found=False, message="invalid raw OCC parameters")
    occ = raw_occ.occ
    n = raw_occ.n
    origin_x = raw_occ.origin_x
    origin_z = raw_occ.origin_z
    n_cells = n * n

    rad_cells = int(np.ceil(float(robot_radius) / res)) if robot_radius > 0 else 0
    inflate_shape_l = (inflate_shape or "circle").lower()
    if inflate_shape_l == "circle":
        occ_inf = _inflate_circle(occ, rad_cells)
    else:
        occ_inf = _inflate_square(occ, rad_cells)
    free = ~occ_inf
    t_occ_ms = (time.perf_counter() - t_a) * 1000.0

    # --- WALKED from memory Gaussians ---
    # Split timing: cand = SharedKeyframes T_WC / world_params;
    # paint = disk raster (pure CPU, no KF lock).
    walked = np.zeros((n, n), dtype=bool)
    n_walked = 0
    shape_l = (shape or "disk").lower()
    cands = []

    t_a = time.perf_counter()
    if memory is not None and getattr(memory, "keyframes", None) is not None:
        center_W = np.array([px, float(nav_y), pz], dtype=np.float32)
        half_side = wr + 3.0 * float(sigma_walked)
        cands = memory.store.candidates_in_square(
            memory.keyframes,
            SID_WALKED,
            center_W,
            half_side,
            float(kappa),
        )
        n_walked = len(cands)
    t_cand_ms = (time.perf_counter() - t_a) * 1000.0

    t_a = time.perf_counter()
    for g in cands:
        cx = float(g.mu_W[0])
        cz = float(g.mu_W[2])
        R = float(kappa) * max(float(g.radius_W), 1e-12)
        if shape_l == "square":
            # half-side = kappa * sigma (edge length 2*kappa*sigma)
            _paint_square(walked, cx, cz, R, origin_x, origin_z, res)
        else:
            _paint_disk(walked, cx, cz, R, origin_x, origin_z, res)
    unexp = free & ~walked
    t_paint_ms = (time.perf_counter() - t_a) * 1000.0
    t_walked_ms = t_cand_ms + t_paint_ms

    # Mask outermost layer: edge cells have no obstacle data beyond the grid,
    # so they are not valid UNEXP frontier targets in local mode.
    unexp[0, :] = False
    unexp[-1, :] = False
    unexp[:, 0] = False
    unexp[:, -1] = False

    # --- Local Dijkstra ---
    local_message = ""
    neigh = _forward_neighbor_order(yaw)
    stack_free_xz = []
    if memory is not None:
        stack = getattr(memory, "_stack_free_xz", None)
        if isinstance(stack, list):
            stack_free_xz = stack

    t_a = time.perf_counter()
    if not np.any(unexp):
        local_message = "no unexplored free cells in window"
        # Still discard OCC history entries so global fallback sees a valid top.
        start, _snap = _resolve_start_cell(
            free, px, pz, origin_x, origin_z, res, n, stack_free_xz
        )
    else:
        start, snap_dist = _resolve_start_cell(
            free, px, pz, origin_x, origin_z, res, n, stack_free_xz
        )
        if start is None:
            local_message = "no free cell for start (robot in OCC, no free history)"
        else:
            hit = _bfs_nearest_unexp(free, unexp, start, neighbor_order=neigh, res=res)
            t_bfs_ms = (time.perf_counter() - t_a) * 1000.0
            if hit is not None:
                (fix, fiz), acc_dist = hit
                wx, wz = _cell_center(fix, fiz, origin_x, origin_z, res)
                # Include robot→start recovery cost so dist_m is never a fake 0
                # when start was recovered from history while robot sits in OCC.
                total_dist = float(snap_dist) + float(acc_dist)
                msg = "ok" if yaw is None else "ok (fwd-bias)"
                if snap_dist > 0.0:
                    msg = msg + " (from free history)"
                _record_success_pose(memory, px, pz)
                return _finish(
                    found=True,
                    xz=(float(wx), float(wz)),
                    dist_m=total_dist,
                    message=msg,
                    n_cells=n_cells,
                    n_walked=n_walked,
                    search_type="local",
                )
            local_message = "no reachable unexplored cell"
    t_bfs_ms = (time.perf_counter() - t_a) * 1000.0

    # --- Global Fallback (disabled: current implementation has known bugs) ---
    # Local coverage is sufficient for navigation, so do not enter the legacy
    # global-grid path below until it is fixed and explicitly re-enabled.
    return _finish(
        found=False,
        message=local_message,
        n_cells=n_cells,
        n_walked=n_walked,
        search_type="local",
    )

    # Legacy global fallback implementation (intentionally unreachable).
    xs_all = [px]
    zs_all = [pz]

    if obstacle_points is not None:
        pts = np.asarray(obstacle_points, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 3:
            xs_all.extend([float(pts[:, 0].min()), float(pts[:, 0].max())])
            zs_all.extend([float(pts[:, 2].min()), float(pts[:, 2].max())])

    global_cands = []
    if memory is not None and getattr(memory, "keyframes", None) is not None:
        global_cands = memory.store.candidates_by_sid(
            memory.keyframes, SID_WALKED,
        )
        for g in global_cands:
            xs_all.append(float(g.mu_W[0]))
            zs_all.append(float(g.mu_W[2]))

    # # DEBUG: global fallback entry point — shows why global search fails
    # print(
    #     f"[frontier-debug] global entry: local_msg=\"{local_message}\" "
    #     f"n_obs={obstacle_points.shape[0] if obstacle_points is not None else 0} "
    #     f"n_cands={len(global_cands)} n_pts_bounds={len(xs_all)} "
    #     f"x_range=[{min(xs_all):.1f},{max(xs_all):.1f}] "
    #     f"z_range=[{min(zs_all):.1f},{max(zs_all):.1f}]",
    #     flush=True,
    # )

    if len(xs_all) <= 1:  # only robot position, no other data
        return _finish(
            found=False,
            message=f"local: {local_message}; global: no data for bounds",
            n_cells=n_cells, n_walked=n_walked,
            search_type="local",
        )

    global_res = float(global_resolution)
    margin = float(robot_radius) + global_res
    global_min_x = min(xs_all) - margin
    global_max_x = max(xs_all) + margin
    global_min_z = min(zs_all) - margin
    global_max_z = max(zs_all) + margin

    global_nx = int(np.ceil((global_max_x - global_min_x) / global_res)) + 1
    global_nz = int(np.ceil((global_max_z - global_min_z) / global_res)) + 1
    global_n = max(global_nx, global_nz, 3)

    _MAX_GLOBAL_CELLS = 50000
    if global_n * global_n > _MAX_GLOBAL_CELLS:
        return _finish(
            found=False,
            message=f"local: {local_message}; global: grid too large ({global_n}x{global_n})",
            n_cells=n_cells, n_walked=n_walked,
            search_type="local",
        )

    global_origin_x = global_min_x
    global_origin_z = global_min_z

    # OCC + inflate
    occ_global = np.zeros((global_n, global_n), dtype=bool)
    if obstacle_points is not None:
        pts_g = np.asarray(obstacle_points, dtype=np.float64)
        if pts_g.ndim == 2 and pts_g.shape[0] > 0 and pts_g.shape[1] >= 3:
            xs_g = pts_g[:, 0]
            zs_g = pts_g[:, 2]
            ixs_g = np.floor((xs_g - global_origin_x) / global_res).astype(np.int32)
            izs_g = np.floor((zs_g - global_origin_z) / global_res).astype(np.int32)
            mask_g = (ixs_g >= 0) & (ixs_g < global_n) & (izs_g >= 0) & (izs_g < global_n)
            ixs_g = ixs_g[mask_g]
            izs_g = izs_g[mask_g]
            if ixs_g.size > 0:
                flat_g = ixs_g * global_n + izs_g
                counts_g = np.bincount(flat_g, minlength=global_n * global_n)
                occ_global = counts_g.reshape(global_n, global_n) >= 10

    rad_cells_global = int(np.ceil(float(robot_radius) / global_res)) if robot_radius > 0 else 0
    if inflate_shape_l == "circle":
        occ_inf_global = _inflate_circle(occ_global, rad_cells_global)
    else:
        occ_inf_global = _inflate_square(occ_global, rad_cells_global)
    free_global = ~occ_inf_global

    # Paint WALKED
    walked_global = np.zeros((global_n, global_n), dtype=bool)
    for g in global_cands:
        cx = float(g.mu_W[0])
        cz = float(g.mu_W[2])
        R = float(kappa) * max(float(g.radius_W), 1e-12)
        if shape_l == "square":
            _paint_square(walked_global, cx, cz, R, global_origin_x, global_origin_z, global_res)
        else:
            _paint_disk(walked_global, cx, cz, R, global_origin_x, global_origin_z, global_res)

    unexp_global = free_global & ~walked_global
    # NOTE: no edge masking in global mode — grid edges are true data boundaries

    if not np.any(unexp_global):
        return _finish(
            found=False,
            message=f"local: {local_message}; global: no unexplored free cells",
            n_cells=n_cells, n_walked=n_walked,
            search_type="global",
        )

    # Re-read the shared stack because the local branch may have popped OCC entries.
    if memory is not None:
        stack = getattr(memory, "_stack_free_xz", None)
        if isinstance(stack, list):
            stack_free_xz = stack

    gstart, gsnap = _resolve_start_cell(
        free_global, px, pz,
        global_origin_x, global_origin_z, global_res, global_n,
        stack_free_xz,
    )
    if gstart is None:
        return _finish(
            found=False,
            message=f"local: {local_message}; global: no free cell for start (robot in OCC, no free history)",
            n_cells=n_cells, n_walked=n_walked,
            search_type="global",
        )

    # # DEBUG: global fallback diagnostics
    # n_unexp_global = int(np.count_nonzero(unexp_global))
    # n_free_global = int(np.count_nonzero(free_global))
    # print(
    #     f"[frontier-debug] global grid={global_nx}x{global_nz} res={global_res:.2f} "
    #     f"free={n_free_global} unexp={n_unexp_global} "
    #     f"range_x=[{global_min_x:.1f},{global_max_x:.1f}] "
    #     f"range_z=[{global_min_z:.1f},{global_max_z:.1f}] "
    #     f"n_obs={obstacle_points.shape[0] if obstacle_points is not None else 0} "
    #     f"n_cands={len(global_cands)} gstart={gstart}",
    #     flush=True,
    # )

    ghit = _bfs_nearest_unexp(free_global, unexp_global, gstart,
                              neighbor_order=neigh, res=global_res)
    if ghit is None:
        return _finish(
            found=False,
            message=f"local: {local_message}; global: no reachable unexplored cell",
            n_cells=n_cells, n_walked=n_walked,
            search_type="global",
        )

    (gfix, gfiz), gacc_dist = ghit
    gwx, gwz = _cell_center(gfix, gfiz, global_origin_x, global_origin_z, global_res)
    _record_success_pose(memory, px, pz)
    return _finish(
        found=True,
        xz=(float(gwx), float(gwz)),
        dist_m=float(gsnap) + float(gacc_dist),
        message=f"global (local: {local_message})",
        n_cells=n_cells, n_walked=n_walked,
        search_type="global",
    )
