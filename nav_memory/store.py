"""Sparse memory store: insert / merge / delete by keyframe."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .geometry import (
    act_sim3_inverse_from_data,
    fetch_T_WC_data_batch,
    world_mu_radius_from_data,
)
from .types import CircleMemory, MemorySource, SID_WALKED, WorldGaussian

from nav_constants import (
    MEM_ANCHOR_SCALE_RATIO_MAX,
    MEM_ANCHOR_SCALE_RATIO_MIN,
    MEM_WALKED_RADIUS_MAX,
    MEM_WALKED_RADIUS_MIN,
)


class MemoryStore:
    def __init__(self):
        self._items: List[CircleMemory] = []
        self._next_id = 0
        self._active_anomalies: Dict[tuple, dict] = {}
        self._pending_anomalies: List[dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Sequence[CircleMemory]:
        return self._items

    def clear(self) -> None:
        self._items.clear()
        self._next_id = 0
        self._active_anomalies.clear()
        self._pending_anomalies.clear()

    def _set_anomaly(self, key: tuple, payload: dict) -> None:
        """Activate one de-duplicated anomaly until the field recovers."""
        if key not in self._active_anomalies:
            self._pending_anomalies.append(dict(payload))
        self._active_anomalies[key] = dict(payload)

    def _clear_anomaly(self, key: tuple) -> None:
        self._active_anomalies.pop(key, None)

    def drain_anomalies(self) -> List[dict]:
        out = list(self._pending_anomalies)
        self._pending_anomalies.clear()
        return out

    def active_anomaly_counts(self) -> dict:
        fields = [str(v.get("field", "")) for v in self._active_anomalies.values()]
        return {
            "invalid_scale_count": sum(
                f in {"anchor_scale", "anchor_scale_ref", "anchor_missing"}
                for f in fields
            ),
            "scale_oob_count": sum(f == "scale_ratio" for f in fields),
            "radius_oob_count": sum(f == "radius_W" for f in fields),
        }

    def _validate_write_radius(self, sid: int, radius: float, kf_id: int) -> bool:
        key = ("write", int(sid), "radius_W")
        if sid != SID_WALKED:
            self._clear_anomaly(key)
            return bool(np.isfinite(radius) and radius > 0.0)
        valid = bool(
            np.isfinite(radius)
            and MEM_WALKED_RADIUS_MIN <= radius <= MEM_WALKED_RADIUS_MAX
        )
        if valid:
            self._clear_anomaly(key)
            return True
        self._set_anomaly(
            key,
            {
                "stage": "bounds_warning",
                "mem_id": None,
                "kf_id": int(kf_id),
                "field": "radius_W",
                "value": float(radius) if np.isfinite(radius) else str(radius),
                "lower": float(MEM_WALKED_RADIUS_MIN),
                "upper": float(MEM_WALKED_RADIUS_MAX),
                "action": "reject_write",
            },
        )
        return False

    def _checked_world_params(
        self,
        m: CircleMemory,
        data: Optional[np.ndarray],
    ) -> Optional[tuple]:
        """Return (mu_W, radius_W, scale, ratio), recording bounds anomalies."""
        invalid_key = (int(m.id), "anchor_scale")
        ref_key = (int(m.id), "anchor_scale_ref")
        ratio_key = (int(m.id), "scale_ratio")
        radius_key = (int(m.id), "radius_W")
        missing_key = (int(m.id), "anchor_missing")

        if data is None:
            self._set_anomaly(
                missing_key,
                {
                    "stage": "bounds_warning",
                    "mem_id": int(m.id),
                    "kf_id": int(m.kf_id),
                    "field": "anchor_missing",
                    "value": None,
                    "lower": 0,
                    "upper": None,
                    "action": "skip_memory",
                },
            )
            return None
        self._clear_anomaly(missing_key)

        data = np.asarray(data, dtype=np.float64).reshape(-1)
        scale = float(data[7]) if data.size >= 8 else float("nan")
        if not np.isfinite(scale) or scale <= 0.0:
            self._set_anomaly(
                invalid_key,
                {
                    "stage": "bounds_warning",
                    "mem_id": int(m.id),
                    "kf_id": int(m.kf_id),
                    "field": "anchor_scale",
                    "value": scale if np.isfinite(scale) else str(scale),
                    "lower": 0.0,
                    "upper": None,
                    "action": "skip_memory",
                },
            )
            self._clear_anomaly(ratio_key)
            return None
        self._clear_anomaly(invalid_key)

        ref = m.anchor_scale_ref
        if ref is None or not np.isfinite(float(ref)) or float(ref) <= 0.0:
            old_ref = ref
            m.anchor_scale_ref = scale
            ref = scale
            if old_ref is not None:
                self._set_anomaly(
                    ref_key,
                    {
                        "stage": "bounds_warning",
                        "mem_id": int(m.id),
                        "kf_id": int(m.kf_id),
                        "field": "anchor_scale_ref",
                        "value": str(old_ref),
                        "lower": 0.0,
                        "upper": None,
                        "action": "reset_ref_to_current",
                    },
                )
        else:
            self._clear_anomaly(ref_key)

        ratio = scale / float(ref)
        if not (MEM_ANCHOR_SCALE_RATIO_MIN <= ratio <= MEM_ANCHOR_SCALE_RATIO_MAX):
            self._set_anomaly(
                ratio_key,
                {
                    "stage": "bounds_warning",
                    "mem_id": int(m.id),
                    "kf_id": int(m.kf_id),
                    "field": "scale_ratio",
                    "value": float(ratio),
                    "lower": float(MEM_ANCHOR_SCALE_RATIO_MIN),
                    "upper": float(MEM_ANCHOR_SCALE_RATIO_MAX),
                    "action": "warn_keep_transform",
                },
            )
        else:
            self._clear_anomaly(ratio_key)

        radius_W = float(m.radius)
        if m.sid == SID_WALKED and not (
            np.isfinite(radius_W)
            and MEM_WALKED_RADIUS_MIN <= radius_W <= MEM_WALKED_RADIUS_MAX
        ):
            self._set_anomaly(
                radius_key,
                {
                    "stage": "bounds_warning",
                    "mem_id": int(m.id),
                    "kf_id": int(m.kf_id),
                    "field": "radius_W",
                    "value": radius_W if np.isfinite(radius_W) else str(radius_W),
                    "lower": float(MEM_WALKED_RADIUS_MIN),
                    "upper": float(MEM_WALKED_RADIUS_MAX),
                    "action": "skip_memory",
                },
            )
            return None
        self._clear_anomaly(radius_key)

        mu_W, radius_W = world_mu_radius_from_data(data, m.mu_c, radius_W)
        return mu_W, radius_W, scale, ratio

    def count(self, sid: Optional[int] = None) -> int:
        if sid is None:
            return len(self._items)
        return sum(1 for m in self._items if m.sid == sid)

    def delete_by_kf(self, kf_id: int) -> int:
        kf_id = int(kf_id)
        removed_ids = {int(m.id) for m in self._items if int(m.kf_id) == kf_id}
        if not removed_ids:
            return 0
        self._items = [m for m in self._items if int(m.kf_id) != kf_id]
        self._active_anomalies = {
            key: value
            for key, value in self._active_anomalies.items()
            if not (key and isinstance(key[0], int) and key[0] in removed_ids)
        }
        self._pending_anomalies = [
            value
            for value in self._pending_anomalies
            if value.get("mem_id") not in removed_ids
        ]
        return len(removed_ids)

    def get_T_WC(self, kf_id: int, keyframes) -> Optional[object]:
        n = len(keyframes)
        if kf_id < 0 or kf_id >= n:
            return None
        return keyframes[kf_id].T_WC

    def world_params(
        self,
        m: CircleMemory,
        keyframes,
        pose_data_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[tuple]:
        """Return (mu_W, radius_W) or None if kf invalid.

        pose_data_map: optional {kf_id: (8,) data} from fetch_T_WC_data_batch
        to avoid per-call CUDA→CPU sync.
        """
        if pose_data_map is not None:
            data = pose_data_map.get(int(m.kf_id))
            checked = self._checked_world_params(m, data)
            return None if checked is None else checked[:2]

        data_map = fetch_T_WC_data_batch(keyframes, [int(m.kf_id)])
        data = data_map.get(int(m.kf_id))
        checked = self._checked_world_params(m, data)
        return None if checked is None else checked[:2]

    def insert_or_merge(
        self,
        *,
        sid: int,
        position_W: np.ndarray,
        radius: float,
        weight: float,
        timestamp: float,
        source: MemorySource,
        kf_id: int,
        keyframes,
        merge_radius: float,
    ) -> Optional[CircleMemory]:
        """
        Merge with nearest same-semantic circle within merge_radius (world),
        else create new anchored at kf_id.
        """
        position_W = np.asarray(position_W, dtype=np.float32).reshape(3)
        kf_id = int(kf_id)

        # One batch D2H for new kf + all same-sid anchors
        kf_ids = [kf_id]
        for m in self._items:
            if m.sid == sid:
                kf_ids.append(int(m.kf_id))
        pose_map = fetch_T_WC_data_batch(keyframes, kf_ids)
        data_new = pose_map.get(kf_id)
        if data_new is None:
            return None
        data_new = np.asarray(data_new).reshape(-1)
        scale_new = float(data_new[7]) if data_new.size >= 8 else float("nan")
        if not np.isfinite(scale_new) or scale_new <= 0.0:
            self._set_anomaly(
                ("write", kf_id, "anchor_scale"),
                {
                    "stage": "bounds_warning",
                    "mem_id": None,
                    "kf_id": kf_id,
                    "field": "anchor_scale",
                    "value": scale_new if np.isfinite(scale_new) else str(scale_new),
                    "lower": 0.0,
                    "upper": None,
                    "action": "reject_write",
                },
            )
            return None
        self._clear_anomaly(("write", kf_id, "anchor_scale"))
        if not self._validate_write_radius(int(sid), float(radius), kf_id):
            return None

        best_idx = None
        best_dist = float("inf")
        best_mu_W = None

        for i, m in enumerate(self._items):
            if m.sid != sid:
                continue
            data = pose_map.get(int(m.kf_id))
            checked = self._checked_world_params(m, data)
            if checked is None:
                continue
            mu_W = checked[0]
            d = float(np.linalg.norm(mu_W - position_W))
            if d < merge_radius and d < best_dist:
                best_dist = d
                best_idx = i
                best_mu_W = mu_W

        if best_idx is not None:
            m = self._items[best_idx]
            w_old = float(m.w)
            dw = float(weight)
            mu_new = (w_old * best_mu_W + dw * position_W) / (w_old + dw)
            # Keep original kf_id; re-project with current T of that anchor.
            data_old = pose_map.get(int(m.kf_id))
            if data_old is None:
                return None
            m.mu_c = act_sim3_inverse_from_data(data_old, mu_new)
            # No enhance-on-revisit: keep original w / radius.
            m.t_last = float(timestamp)
            return m

        mu_c = act_sim3_inverse_from_data(data_new, position_W)
        m = CircleMemory(
            id=self._next_id,
            kf_id=kf_id,
            sid=int(sid),
            mu_c=mu_c,
            radius=float(radius),
            anchor_scale_ref=scale_new,
            w=float(weight),
            t_last=float(timestamp),
            source=source,
        )
        self._next_id += 1
        self._items.append(m)
        return m

    def candidates_in_ball(
        self,
        keyframes,
        sid: int,
        center_W: np.ndarray,
        radius: float,
        kappa: float,
    ) -> List[WorldGaussian]:
        """
        World-frame candidates in ball. Batch-loads unique T_WC (one .cpu() for
        SharedKeyframes) then pure-CPU Sim3 for each gaussian.
        """
        center_W = np.asarray(center_W, dtype=np.float32).reshape(3)
        matched: List[tuple] = []
        for i, m in enumerate(self._items):
            if m.sid != sid:
                continue
            matched.append((i, m))
        if not matched:
            return []

        pose_map = fetch_T_WC_data_batch(
            keyframes, [int(m.kf_id) for _, m in matched]
        )

        out: List[WorldGaussian] = []
        r_lim = float(radius)
        k = float(kappa)
        for i, m in matched:
            data = pose_map.get(int(m.kf_id))
            checked = self._checked_world_params(m, data)
            if checked is None:
                continue
            mu_W, radius_W = checked[:2]
            if np.linalg.norm(mu_W - center_W) <= r_lim + k * radius_W:
                out.append(
                    WorldGaussian(
                        store_index=i,
                        mu_W=mu_W,
                        radius_W=radius_W,
                        w=float(m.w),
                        sid=m.sid,
                        mem_id=m.id,
                    )
                )
        return out

    def candidates_by_sid(
        self,
        keyframes,
        sid: int,
    ) -> List[WorldGaussian]:
        """
        All world-frame candidates matching sid (no spatial filter).
        Batch-loads unique T_WC (single D2H for SharedKeyframes), then
        pure-CPU Sim3 for each gaussian.
        """
        matched: List[tuple] = []
        for i, m in enumerate(self._items):
            if m.sid != sid:
                continue
            matched.append((i, m))
        if not matched:
            return []

        pose_map = fetch_T_WC_data_batch(
            keyframes, [int(m.kf_id) for _, m in matched]
        )

        out: List[WorldGaussian] = []
        for i, m in matched:
            data = pose_map.get(int(m.kf_id))
            checked = self._checked_world_params(m, data)
            if checked is None:
                continue
            mu_W, radius_W = checked[:2]
            out.append(
                WorldGaussian(
                    store_index=i,
                    mu_W=mu_W,
                    radius_W=radius_W,
                    w=float(m.w),
                    sid=m.sid,
                    mem_id=m.id,
                )
            )
        return out

    def candidates_in_square(
        self,
        keyframes,
        sid: int,
        center_W: np.ndarray,
        half_side: float,
        kappa: float,
    ) -> List[WorldGaussian]:
        """
        World-frame candidates within an axis-aligned square on the X-Z plane.

        A gaussian is included if its center (mu_W) falls within the square
        of half-side ``half_side``, expanded by ``kappa * radius_W`` on each
        axis independently.  Y coordinate is ignored (navigation plane).

        Batch-loads unique T_WC (single D2H for SharedKeyframes).
        """
        center_W = np.asarray(center_W, dtype=np.float32).reshape(3)
        matched: List[tuple] = []
        for i, m in enumerate(self._items):
            if m.sid != sid:
                continue
            matched.append((i, m))
        if not matched:
            return []

        pose_map = fetch_T_WC_data_batch(
            keyframes, [int(m.kf_id) for _, m in matched]
        )

        out: List[WorldGaussian] = []
        half = float(half_side)
        k = float(kappa)
        cx = float(center_W[0])
        cz = float(center_W[2])
        for i, m in matched:
            data = pose_map.get(int(m.kf_id))
            checked = self._checked_world_params(m, data)
            if checked is None:
                continue
            mu_W, radius_W = checked[:2]
            margin = k * float(radius_W)
            if (abs(float(mu_W[0]) - cx) <= half + margin and
                    abs(float(mu_W[2]) - cz) <= half + margin):
                out.append(
                    WorldGaussian(
                        store_index=i,
                        mu_W=mu_W,
                        radius_W=radius_W,
                        w=float(m.w),
                        sid=m.sid,
                        mem_id=m.id,
                    )
                )
        return out

    def to_serializable(self) -> list:
        rows = []
        for m in self._items:
            rows.append(
                {
                    "id": m.id,
                    "kf_id": m.kf_id,
                    "sid": m.sid,
                    "mu_c": m.mu_c.tolist(),
                    "radius_W": m.radius,
                    "anchor_scale_ref": m.anchor_scale_ref,
                    "w": m.w,
                    "t_last": m.t_last,
                    "source": int(m.source),
                }
            )
        return rows

    def load_serializable(self, rows: list, version: int = 1) -> None:
        del version  # v1/v2 field compatibility is handled per row below.
        self.clear()
        max_id = -1
        for row in rows:
            m = CircleMemory(
                id=int(row["id"]),
                kf_id=int(row["kf_id"]),
                sid=int(row["sid"]),
                mu_c=np.asarray(row["mu_c"], dtype=np.float32),
                radius=float(
                    row.get("radius_W", row.get("radius", row.get("sigma", 0.2)))
                ),
                anchor_scale_ref=(
                    float(row["anchor_scale_ref"])
                    if row.get("anchor_scale_ref") is not None
                    else None
                ),
                w=float(row["w"]),
                t_last=float(row.get("t_last", 0.0)),
                source=MemorySource(int(row.get("source", MemorySource.MANUAL))),
            )
            self._items.append(m)
            max_id = max(max_id, m.id)
        self._next_id = max_id + 1

    def diagnostics(self, keyframes) -> dict:
        """Summarize live world parameters and active bounds anomalies."""
        items = list(self._items)
        pose_map = fetch_T_WC_data_batch(
            keyframes, [int(m.kf_id) for m in items]
        ) if items else {}
        scales: List[float] = []
        ratios: List[float] = []
        radii: List[float] = []
        max_item: Optional[CircleMemory] = None
        max_radius = float("-inf")

        for m in items:
            checked = self._checked_world_params(
                m, pose_map.get(int(m.kf_id))
            )
            if checked is None:
                continue
            _, radius_W, scale, ratio = checked
            scales.append(float(scale))
            ratios.append(float(ratio))
            radii.append(float(radius_W))
            if radius_W > max_radius:
                max_radius = float(radius_W)
                max_item = m

        counts = self.active_anomaly_counts()
        return {
            "n_total": len(items),
            "n_walked": sum(1 for m in items if m.sid == SID_WALKED),
            "n_anchors": len({int(m.kf_id) for m in items}),
            "anchor_scale_min": min(scales) if scales else None,
            "anchor_scale_max": max(scales) if scales else None,
            "scale_ratio_min": min(ratios) if ratios else None,
            "scale_ratio_max": max(ratios) if ratios else None,
            "radius_W_min": min(radii) if radii else None,
            "radius_W_max": max(radii) if radii else None,
            "max_radius_mem_id": int(max_item.id) if max_item is not None else None,
            "max_radius_kf_id": int(max_item.kf_id) if max_item is not None else None,
            **counts,
        }
