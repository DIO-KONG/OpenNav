"""Geometry helpers for memory: Sim3 maps, ball sampling, hits, clustering."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .types import WorldGaussian


def sim3_translation(T_WC) -> np.ndarray:
    """Extract translation (robot/camera world position) from lietorch Sim3 or data tensor."""
    data = _sim3_data(T_WC)
    return data[:3].astype(np.float32)


def sim3_scale(T_WC) -> float:
    data = _sim3_data(T_WC)
    return float(data[7])


def _sim3_data(T_WC) -> np.ndarray:
    """Return flat (8,) array: t(3) + q(4) + s(1).

    Note: if T_WC.data is a CUDA tensor, this triggers a D2H + stream sync.
    Prefer fetch_T_WC_data_batch() for multi-keyframe queries.
    """
    if hasattr(T_WC, "data"):
        data = T_WC.data
    else:
        data = T_WC
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if data.size < 8:
        raise ValueError(f"Sim3 data expected >=8 elements, got {data.size}")
    return data


def act_sim3_from_data(data: np.ndarray, points_c: np.ndarray) -> np.ndarray:
    """Apply Sim3 from flat (8,) CPU data to points (camera → world)."""
    single = np.asarray(points_c).ndim == 1
    pts = np.asarray(points_c, dtype=np.float64).reshape(-1, 3)
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    t = data[:3]
    q = data[3:7]
    s = float(data[7])
    R = _quat_to_rot(q)
    out = (s * (pts @ R.T)) + t
    out = out.astype(np.float32)
    return out[0] if single else out


def act_sim3_inverse_from_data(data: np.ndarray, points_w: np.ndarray) -> np.ndarray:
    """Apply Sim3 inverse from flat (8,) CPU data (world → camera)."""
    single = np.asarray(points_w).ndim == 1
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    t = data[:3]
    q = data[3:7]
    s = float(data[7])
    R = _quat_to_rot(q)
    out = ((pts - t) @ R) / max(s, 1e-12)
    out = out.astype(np.float32)
    return out[0] if single else out


def world_mu_radius_from_data(
    data: np.ndarray, mu_c: np.ndarray, radius: float
) -> Tuple[np.ndarray, float]:
    """μ_W, radius_W from one CPU Sim3 row (single D2H already done).

    Center keeps the original Sim3 transform. Radius is already stored in
    world units and must remain invariant under backend scale optimization.
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    mu_W = act_sim3_from_data(data, mu_c)
    radius_W = float(radius)
    return mu_W, radius_W


def fetch_T_WC_data_batch(keyframes, kf_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    """
    Load Sim3 data for many keyframe ids with as few CUDA→CPU syncs as possible.

    SharedKeyframes path: one index_select + **one** .cpu() for all unique ids.
    Fallback (fake/tests): one _sim3_data per unique id via __getitem__.

    Returns:
        {kf_id: np.ndarray shape (8,) float64}
    """
    # preserve first-seen order of unique ids
    unique: List[int] = []
    seen = set()
    for k in kf_ids:
        ki = int(k)
        if ki not in seen:
            seen.add(ki)
            unique.append(ki)
    if not unique:
        return {}

    T_buf = getattr(keyframes, "T_WC", None)
    lock = getattr(keyframes, "lock", None)
    n_size = getattr(keyframes, "n_size", None)

    # --- SharedKeyframes: (buffer, 1, 8) CUDA/CPU shared tensor ---
    if isinstance(T_buf, torch.Tensor) and T_buf.dim() >= 2:
        if n_size is not None:
            if lock is not None:
                with lock:
                    n = int(n_size.value)
            else:
                n = int(n_size.value)
        else:
            try:
                n = len(keyframes)
            except Exception:
                n = int(T_buf.shape[0])
        unique = [k for k in unique if 0 <= k < n]
        if not unique:
            return {}

        idx_t = torch.as_tensor(unique, dtype=torch.long, device=T_buf.device)
        # Clone under lock (short); .cpu() outside lock so sync does not hold RLock.
        if lock is not None:
            with lock:
                chunk = T_buf.index_select(0, idx_t).detach().clone()
        else:
            chunk = T_buf.index_select(0, idx_t).detach().clone()

        # Single D2H / stream sync for the whole batch
        arr = chunk.float().cpu().numpy().reshape(len(unique), -1)
        out: Dict[int, np.ndarray] = {}
        for i, kf in enumerate(unique):
            row = np.asarray(arr[i], dtype=np.float64).reshape(-1)
            if row.size >= 8:
                out[int(kf)] = row[:8].copy()
        return out

    # --- Fallback: list-like keyframes (unit tests, non-shared) ---
    out = {}
    try:
        n = len(keyframes)
    except Exception:
        n = 0
    for kf in unique:
        if kf < 0 or (n > 0 and kf >= n):
            continue
        try:
            T = keyframes[kf].T_WC
            out[int(kf)] = _sim3_data(T)
        except Exception:
            continue
    return out


def act_sim3_points(T_WC, points_c: np.ndarray) -> np.ndarray:
    """Apply Sim3 to (N,3) or (3,) points in camera frame → world."""
    return act_sim3_from_data(_sim3_data(T_WC), points_c)


def act_sim3_inverse_points(T_WC, points_w: np.ndarray) -> np.ndarray:
    """Apply Sim3 inverse: world → camera frame."""
    return act_sim3_inverse_from_data(_sim3_data(T_WC), points_w)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3x3 rotation (lietorch convention)."""
    x, y, z, w = [float(v) for v in q]
    # Normalize
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def sample_ball_uniform(
    center: np.ndarray, radius: float, n: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Volume-uniform samples in ball B(c, r). Returns (n, 3)."""
    if rng is None:
        rng = np.random.default_rng()
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if radius <= 0 or n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    # Gaussian direction → unit sphere
    dirs = rng.normal(size=(n, 3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs = dirs / norms
    u = rng.random(n)
    r = radius * np.cbrt(u)
    pts = center + dirs * r[:, None]
    return pts.astype(np.float32)


def hit_mask(
    samples: np.ndarray, mu_W: np.ndarray, radius_W: float, kappa: float
) -> np.ndarray:
    """Boolean (N,) whether each sample hits the circle (radius kappa*radius_W)."""
    d = np.linalg.norm(samples - mu_W.reshape(1, 3), axis=1)
    return d <= (kappa * max(radius_W, 1e-12))


def any_hit(
    sample: np.ndarray, circles: Sequence[WorldGaussian], kappa: float
) -> bool:
    for g in circles:
        if np.linalg.norm(sample - g.mu_W) <= kappa * max(g.radius_W, 1e-12):
            return True
    return False


def greedy_cluster(
    items: List[WorldGaussian], eps: float
) -> List[List[WorldGaussian]]:
    """
    Greedy clustering by center distance.
    Seeds sorted by (n_hits * w) descending; merge if ||μi-μj|| < eps.
    """
    if not items:
        return []
    remaining = sorted(items, key=lambda g: g.n_hits * g.w, reverse=True)
    clusters: List[List[WorldGaussian]] = []
    used = [False] * len(remaining)

    for i, seed in enumerate(remaining):
        if used[i]:
            continue
        cluster = [seed]
        used[i] = True
        for j in range(i + 1, len(remaining)):
            if used[j]:
                continue
            if np.linalg.norm(remaining[j].mu_W - seed.mu_W) < eps:
                # Also allow merge to any member of cluster (single-linkage to seed only
                # is simpler; use seed for stability as per framework greedy-from-seed)
                cluster.append(remaining[j])
                used[j] = True
        clusters.append(cluster)
    return clusters


def cluster_score(cluster: Sequence[WorldGaussian]) -> float:
    return float(sum(g.n_hits * g.w for g in cluster))


def cluster_center(cluster: Sequence[WorldGaussian]) -> np.ndarray:
    num = np.zeros(3, dtype=np.float64)
    den = 0.0
    for g in cluster:
        weight = max(g.n_hits * g.w, 1e-12)
        num += weight * g.mu_W.astype(np.float64)
        den += weight
    return (num / max(den, 1e-12)).astype(np.float32)


def sample_circle_section_disk(
    mu_W: np.ndarray,
    radius_W: float,
    height: float,
    n: int,
    thickness: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a thin horizontal disk (world Y = height) from a circle of radius radius_W.

    Horizontal (x, z) ~ Uniform disk of radius radius_W; y ~ Unif[height - t/2, height + t/2].

    Returns (M, 3) float32, M=n or 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    mu = np.asarray(mu_W, dtype=np.float64).reshape(3)
    r = max(float(radius_W), 1e-6)
    # Uniform disk in X-Z plane
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rad = r * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    x = mu[0] + rad * np.cos(ang)
    z = mu[2] + rad * np.sin(ang)
    half_t = 0.5 * max(float(thickness), 0.0)
    if half_t > 0:
        y = float(height) + rng.uniform(-half_t, half_t, size=n)
    else:
        y = np.full(n, float(height), dtype=np.float64)

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts
