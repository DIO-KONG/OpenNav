"""MemorySystem facade: write gates, walked policy, queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .frontier_grid import (
    GridFrontierResult,
    get_nearest_frontier_grid as _get_nearest_frontier_grid,
)
from .geometry import (
    cluster_center,
    cluster_score,
    greedy_cluster,
    sample_circle_section_disk,
)
from .persist import load_memory, save_memory
from .store import MemoryStore
from .types import (
    MemoryEvent,
    MemorySource,
    ObjectQueryResult,
    SID_WALKED,
    WalkedQueryResult,
)
from .vocab import SemanticVocab

# 几何/阈值统一来源: nav_constants.py (ROBOT_RADIUS 派生, 与 nav 联动)
from nav_constants import (
    MEM_ENABLE,
    MEM_PERSIST_ENABLE,
    MEM_DEBUG_EVERY,
    WALK_MIN_TRANS,
    MEM_WALKED_ENABLE,
    MEM_WALKED_RADIUS,
    MEM_WALKED_MERGE_RADIUS,
    MEM_WALKED_EDGE_MARGIN,
    MEM_WALKED_MIN_DT,
    MEM_WALKED_WEIGHT,
    MEM_OBJECT_RADIUS,
    MEM_OBJECT_MERGE_RADIUS,
    MEM_OBJECT_CLUSTER_EPS,
    MEM_OBJECT_WEIGHT,
    MEM_QUERY_RADIUS,
    MEM_QUERY_N_SAMPLES,
    MEM_QUERY_KAPPA,
    MEM_FRONTIER_ROBOT_RADIUS,
    MEM_FRONTIER_RESOLUTION,
    MEM_FRONTIER_WIN_RADIUS,
    MEM_FRONTIER_KAPPA,
    MEM_FRONTIER_SHAPE,
    MEM_FRONTIER_MAX_OBSTACLE_POINTS,
    MEM_VOCAB_ENTRIES,
    MEM_FRONTIER_INFLATE_TYPE,
    MEM_FRONTIER_SIGMA_WALKED,
    MEM_FRONTIER_GLOBAL_RESOLUTION,
)


def _pkg_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def _default_memory_cfg() -> Dict[str, Any]:
    """Assemble default cfg from nav_constants (geometry/阈值 + 语义词表, 单一真相源)."""
    cfg = {
        "enable": MEM_ENABLE,
        "debug_every": MEM_DEBUG_EVERY,
        "walked": {
            "enable": MEM_WALKED_ENABLE,
            "radius": MEM_WALKED_RADIUS,
            "min_trans": WALK_MIN_TRANS,
            "min_dt": MEM_WALKED_MIN_DT,
            "weight": MEM_WALKED_WEIGHT,
            "merge_radius": MEM_WALKED_MERGE_RADIUS,
        },
        "object": {
            "default_radius": MEM_OBJECT_RADIUS,
            "merge_radius": MEM_OBJECT_MERGE_RADIUS,
            "cluster_eps": MEM_OBJECT_CLUSTER_EPS,
            "weight": MEM_OBJECT_WEIGHT,
        },
        "query": {
            "radius": MEM_QUERY_RADIUS,
            "n_samples": MEM_QUERY_N_SAMPLES,
            "kappa": MEM_QUERY_KAPPA,
        },
        "frontier_grid": {
            "enable": True,
            "resolution": MEM_FRONTIER_RESOLUTION,
            "win_radius": MEM_FRONTIER_WIN_RADIUS,
            "kappa": MEM_FRONTIER_KAPPA,
            "shape": MEM_FRONTIER_SHAPE,
            "max_obstacle_points": MEM_FRONTIER_MAX_OBSTACLE_POINTS,
            "inflate_shape": MEM_FRONTIER_INFLATE_TYPE,
            "sigma_walked": MEM_FRONTIER_SIGMA_WALKED,
            "global_resolution": MEM_FRONTIER_GLOBAL_RESOLUTION,
            "robot_radius": MEM_FRONTIER_ROBOT_RADIUS,
        },
        "persist": {"enable": MEM_PERSIST_ENABLE},
    }
    # vocab 段由 nav_constants.py 的 MEM_VOCAB_ENTRIES 装配 (与 nav 统一单一真相源)
    cfg["vocab"] = {"entries": MEM_VOCAB_ENTRIES}
    return cfg


def _build_vocab(vocab_cfg) -> "SemanticVocab":
    """vocab_cfg: SemanticVocab | dict with 'entries'."""
    if isinstance(vocab_cfg, SemanticVocab):
        return vocab_cfg
    if isinstance(vocab_cfg, str):
        return SemanticVocab.from_yaml(vocab_cfg)
    vocab = SemanticVocab()
    for e in (vocab_cfg or {}).get("entries", []):
        vocab._add_entry(int(e["id"]), e["name"], e.get("aliases") or [])
    return vocab


def _merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


class MemorySystem:
    def __init__(
        self,
        cfg: Optional[dict] = None,
        vocab: Optional[SemanticVocab] = None,
        rng_seed: Optional[int] = None,
    ):
        self.cfg = _merge_dict(_default_memory_cfg(), cfg or {})
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = _build_vocab(self.cfg.get("vocab"))
        self.store = MemoryStore()
        self.keyframes = None
        self.write_enabled = True
        self._last_n_kf = 0
        self._last_walked_pos: Optional[np.ndarray] = None
        self._last_walked_t: Optional[float] = None
        self._stack_free_xz: List[Tuple[float, float]] = [(0.0, 0.0)]
        self._removed_last = 0
        self._removed_total = 0
        self._removed_kf_ids_last: List[int] = []
        self._rng = np.random.default_rng(rng_seed)

    @property
    def enable(self) -> bool:
        return bool(self.cfg.get("enable", True))

    def bind_keyframes(self, keyframes) -> None:
        self.keyframes = keyframes
        self._last_n_kf = len(keyframes) if keyframes is not None else 0

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = bool(enabled)

    def clear(self) -> None:
        self.store.clear()
        self._last_walked_pos = None
        self._last_walked_t = None
        self._stack_free_xz = [(0.0, 0.0)]
        self._removed_last = 0
        self._removed_total = 0
        self._removed_kf_ids_last = []

    def count(self, sid: Optional[int] = None) -> int:
        return self.store.count(sid)

    def sync_keyframe_count(self, n: Optional[int] = None) -> int:
        """If keyframe count dropped (pop_last), drop memories on removed indices."""
        if self.keyframes is None and n is None:
            return 0
        n_now = int(n if n is not None else len(self.keyframes))
        removed = 0
        removed_kf_ids: List[int] = []
        if n_now < self._last_n_kf:
            for k in range(n_now, self._last_n_kf):
                n_removed = self.store.delete_by_kf(k)
                removed += n_removed
                if n_removed > 0:
                    removed_kf_ids.append(int(k))
        if removed > 0:
            self._removed_last += int(removed)
            self._removed_total += int(removed)
            self._removed_kf_ids_last.extend(removed_kf_ids)
        self._last_n_kf = n_now
        return removed

    def collect_diagnostics(
        self,
        *,
        n_candidates: int = 0,
        frontier_found: bool = False,
        frontier_message: str = "",
        consume_deletions: bool = True,
    ) -> dict:
        """Return one structured memory/frontier diagnostic snapshot."""
        if self.keyframes is None:
            data = {
                "n_total": self.store.count(),
                "n_walked": self.store.count(SID_WALKED),
                "n_anchors": 0,
                "anchor_scale_min": None,
                "anchor_scale_max": None,
                "scale_ratio_min": None,
                "scale_ratio_max": None,
                "radius_W_min": None,
                "radius_W_max": None,
                "max_radius_mem_id": None,
                "max_radius_kf_id": None,
                "invalid_scale_count": 0,
                "scale_oob_count": 0,
                "radius_oob_count": 0,
            }
        else:
            data = self.store.diagnostics(self.keyframes)
        data.update({
            "stage": "frontier_snapshot",
            "n_candidates": int(n_candidates),
            "removed_last": int(self._removed_last),
            "removed_total": int(self._removed_total),
            "removed_kf_ids": list(self._removed_kf_ids_last),
            "frontier_found": bool(frontier_found),
            "frontier_message": str(frontier_message or ""),
        })
        if consume_deletions:
            self._removed_last = 0
            self._removed_kf_ids_last = []
        return data

    def drain_bounds_warnings(self) -> List[dict]:
        return self.store.drain_anomalies()

    def _latest_kf_id(self) -> Optional[int]:
        if self.keyframes is None:
            return None
        n = len(self.keyframes)
        if n <= 0:
            return None
        return n - 1

    def _merge_radius_for(self, sid: int) -> float:
        if sid == SID_WALKED:
            return float(self.cfg["walked"]["merge_radius"])
        return float(self.cfg["object"]["merge_radius"])

    def _default_radius_for(self, sid: int) -> float:
        if sid == SID_WALKED:
            return float(self.cfg["walked"]["radius"])
        return float(self.cfg["object"]["default_radius"])

    def ingest(self, event: MemoryEvent) -> bool:
        if not self.enable or not self.write_enabled:
            return False
        if self.keyframes is None:
            return False
        sid = self.vocab.resolve(event.semantic)
        if sid is None or sid == 0:
            return False
        kf_id = self._latest_kf_id()
        if kf_id is None:
            return False

        radius = event.radius if event.radius is not None else self._default_radius_for(sid)
        weight = float(event.weight)
        if weight <= 0:
            weight = 1.0

        m = self.store.insert_or_merge(
            sid=sid,
            position_W=event.position_W,
            radius=radius,
            weight=weight,
            timestamp=float(event.timestamp),
            source=event.source,
            kf_id=kf_id,
            keyframes=self.keyframes,
            merge_radius=self._merge_radius_for(sid),
        )
        return m is not None

    def maybe_write_walked(
        self, position_W: np.ndarray, timestamp: float = 0.0
    ) -> bool:
        if not self.enable or not self.write_enabled:
            return False
        if not self.cfg.get("walked", {}).get("enable", True):
            return False

        position_W = np.asarray(position_W, dtype=np.float32).reshape(3)
        wcfg = self.cfg["walked"]
        d_min = float(wcfg.get("min_trans", 0.1))
        dt_min = float(wcfg.get("min_dt", 1.0))

        # OR logic: write if moved enough OR enough time elapsed.
        # Avoids permanent starvation when robot is stationary.
        dist_ok = True
        time_ok = True
        if self._last_walked_pos is not None:
            dist_ok = bool(
                np.linalg.norm(position_W - self._last_walked_pos) >= d_min
            )
        if self._last_walked_t is not None and dt_min > 0:
            time_ok = bool(
                float(timestamp) - self._last_walked_t >= dt_min
            )
        if not dist_ok and not time_ok:
            return False

        ok = self.ingest(
            MemoryEvent(
                semantic=SID_WALKED,
                position_W=position_W,
                radius=float(wcfg.get("radius", 0.2)),
                weight=float(wcfg.get("weight", 1.0)),
                timestamp=float(timestamp),
                source=MemorySource.BEHAVIOR,
            )
        )
        if ok:
            self._last_walked_pos = position_W.copy()
            self._last_walked_t = float(timestamp)
        return ok

    def query_walked(
        self,
        position_W: np.ndarray,
        radius: Optional[float] = None,
        n_samples: Optional[int] = None,
        kappa: Optional[float] = None,
    ) -> WalkedQueryResult:
        """Deterministic point-in-circle test (no Monte-Carlo sampling).

        Returns score=1.0 (hit) if position_W lies *deep enough* inside any
        walked circle, else 0.0.

        边沿判定: walked 圆的有效半径 = kappa * radius_W。查询点要算"走过",
        其到圆边界的剩余余量必须 > MEM_WALKED_EDGE_MARGIN (默认=ROBOT_RADIUS),
        即要求 ||p - mu|| < kappa*radius_W - MEM_WALKED_EDGE_MARGIN。
        这样贴着记忆圆边沿 (机器人体积半径内到不了边界) 的点不会被误判为已走过。
        `n_samples` 仅作 API 兼容, 未使用。
        """
        empty = WalkedQueryResult(score=0.0, n_samples=0, n_hits=0, n_candidates=0)
        if not self.enable or self.keyframes is None:
            return empty

        qcfg = self.cfg["query"]
        r = float(radius if radius is not None else qcfg["radius"])
        k = float(kappa if kappa is not None else qcfg["kappa"])
        margin = float(MEM_WALKED_EDGE_MARGIN)
        if r <= 0:
            return empty

        cands = self.store.candidates_in_ball(
            self.keyframes, SID_WALKED, position_W, r, k
        )
        if not cands:
            return WalkedQueryResult(score=0.0, n_samples=0, n_hits=0, n_candidates=0)

        position_W = np.asarray(position_W, dtype=np.float64).reshape(3)
        for g in cands:
            r_eff = k * max(g.radius_W, 1e-12)
            # 到边界余量 = r_eff - ||p - mu||; 须 > margin 才算走过
            if np.linalg.norm(position_W - g.mu_W) <= r_eff - margin:
                return WalkedQueryResult(
                    score=1.0, n_samples=0, n_hits=1, n_candidates=len(cands)
                )
        return WalkedQueryResult(
            score=0.0, n_samples=0, n_hits=0, n_candidates=len(cands)
        )

    def query_object(
        self,
        semantic: Union[str, int],
        position_W: np.ndarray,
        radius: Optional[float] = None,
        n_samples: Optional[int] = None,
        kappa: Optional[float] = None,
        cluster_eps: Optional[float] = None,
    ) -> ObjectQueryResult:
        empty = ObjectQueryResult(found=False)
        if not self.enable or self.keyframes is None:
            return empty

        sid = self.vocab.resolve(semantic)
        if sid is None or sid == 0:
            return empty

        qcfg = self.cfg["query"]
        r = float(radius if radius is not None else qcfg["radius"])
        k = float(kappa if kappa is not None else qcfg["kappa"])
        eps = float(
            cluster_eps
            if cluster_eps is not None
            else self.cfg["object"]["cluster_eps"]
        )
        if r <= 0:
            return empty

        cands = self.store.candidates_in_ball(self.keyframes, sid, position_W, r, k)
        if not cands:
            return ObjectQueryResult(found=False, n_candidates=0)

        # Deterministic circle-overlap test (no sampling): does query ball
        # B(position_W, r) overlap circle(mu_W, k*radius_W)?
        position_W = np.asarray(position_W, dtype=np.float64).reshape(3)
        active = []
        for g in cands:
            if np.linalg.norm(position_W - g.mu_W) <= r + k * max(g.radius_W, 1e-12):
                g.n_hits = 1
                active.append(g)
        if not active:
            return ObjectQueryResult(
                found=False, n_candidates=len(cands), score=0.0
            )

        clusters = greedy_cluster(active, eps)
        best = max(clusters, key=cluster_score)
        return ObjectQueryResult(
            found=True,
            mu_W=cluster_center(best),
            score=cluster_score(best),
            n_gaussians_in_cluster=len(best),
            n_candidates=len(cands),
        )

    def export_centers_W(self) -> list:
        """Debug helper: list of dicts with world centers."""
        rows = []
        if self.keyframes is None:
            return rows
        from .geometry import fetch_T_WC_data_batch

        items = list(self.store.items())
        pose_map = fetch_T_WC_data_batch(
            self.keyframes, [int(m.kf_id) for m in items]
        )
        for m in items:
            wp = self.store.world_params(m, self.keyframes, pose_data_map=pose_map)
            if wp is None:
                continue
            mu_W, radius_W = wp
            rows.append(
                {
                    "id": m.id,
                    "kf_id": m.kf_id,
                    "sid": m.sid,
                    "name": self.vocab.name(m.sid),
                    "mu_W": mu_W.tolist(),
                    "radius_W": radius_W,
                    "w": m.w,
                }
            )
        return rows

    def get_nearest_frontier_grid(
        self,
        pose_xz,
        *,
        yaw: Optional[float] = None,
        nav_y: float = 0.0,
        obstacle_points: Optional[np.ndarray] = None,
        robot_radius: Optional[float] = None,
        resolution: Optional[float] = None,
        win_radius: Optional[float] = None,
        kappa: Optional[float] = None,
        shape: Optional[str] = None,
        inflate_shape: Optional[str] = None,
        sigma_walked: Optional[float] = None,
        global_resolution: Optional[float] = None,
        max_obstacle_points: Optional[int] = None,
    ) -> GridFrontierResult:
        """
        Local FREE \\ WALKED nearest frontier (geodesic BFS on FREE).

        yaw: optional heading (rad); if set, BFS prefers forward neighbors on
        equal geodesic distance. Defaults from cfg['frontier_grid'].
        """
        fcfg = self.cfg.get("frontier_grid") or {}
        if not self.enable or not fcfg.get("enable", True):
            return GridFrontierResult(found=False, message="disabled")
        if self.keyframes is None:
            return GridFrontierResult(found=False, message="no keyframes")

        return _get_nearest_frontier_grid(
            self,
            pose_xz,
            yaw=float(yaw) if yaw is not None else None,
            nav_y=float(nav_y),
            obstacle_points=obstacle_points,
            robot_radius=float(
                robot_radius
                if robot_radius is not None
                else fcfg.get("robot_radius", MEM_FRONTIER_ROBOT_RADIUS)
            ),
            resolution=float(
                resolution if resolution is not None else fcfg.get("resolution", 0.10)
            ),
            win_radius=float(
                win_radius if win_radius is not None else fcfg.get("win_radius", 6.0)
            ),
            kappa=float(kappa if kappa is not None else fcfg.get("kappa", 1.0)),
            shape=str(shape if shape is not None else fcfg.get("shape", "disk")),
                        inflate_shape=str(
                inflate_shape
                if inflate_shape is not None
                else fcfg.get("inflate_shape", "circle")
            ),
            sigma_walked=float(
                sigma_walked
                if sigma_walked is not None
                else fcfg.get("sigma_walked", 0.40)
            ),
            global_resolution=float(
                global_resolution
                if global_resolution is not None
                else fcfg.get("global_resolution", 0.30)
            ),
        )

    def sample_section_cloud(
        self,
        height: float = 0.0,
        *,
        semantic: Optional[Union[str, int]] = SID_WALKED,
        n_per_gauss: int = 128,
        thickness: float = 0.05,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample a thin horizontal section (world Y ≈ height) from memory circles.

        Each matching circle contributes a 2D uniform disk in the X–Z plane
        (radius = radius_W), with Y jittered in [height - t/2, height + t/2].
        Does **not** include SLAM reconstruction points — memory only.

        Returns:
            np.ndarray float32 shape (N, 3). Empty (0, 3) if no points.
        """
        empty = np.zeros((0, 3), dtype=np.float32)
        if not self.enable or self.keyframes is None:
            return empty
        if n_per_gauss <= 0:
            return empty

        if semantic is None:
            sid_filter = None
        else:
            sid_filter = self.vocab.resolve(semantic)
            if sid_filter is None:
                return empty
            if sid_filter == 0:
                return empty

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        # Filter then one batch T_WC D2H (same path as frontier candidates_in_ball)
        matched = []
        for m in self.store.items():
            if sid_filter is not None and m.sid != sid_filter:
                continue
            matched.append(m)
        if not matched:
            return empty
        from .geometry import fetch_T_WC_data_batch

        pose_map = fetch_T_WC_data_batch(
            self.keyframes, [int(m.kf_id) for m in matched]
        )
        chunks = []
        for m in matched:
            wp = self.store.world_params(m, self.keyframes, pose_data_map=pose_map)
            if wp is None:
                continue
            mu_W, radius_W = wp
            pts = sample_circle_section_disk(
                mu_W,
                radius_W,
                height=float(height),
                n=int(n_per_gauss),
                thickness=float(thickness),
                rng=rng,
            )
            if pts.shape[0] > 0:
                chunks.append(pts)

        if not chunks:
            return empty
        return np.concatenate(chunks, axis=0)

    def save(self, path: Union[str, Path]) -> None:
        save_memory(path, self.store, self.vocab)

    def load(self, path: Union[str, Path]) -> None:
        load_memory(path, self.store)
