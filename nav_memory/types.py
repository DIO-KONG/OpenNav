"""Data types for the parallel semantic memory layer (method G)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

import numpy as np


class MemorySource(IntEnum):
    BEHAVIOR = 0
    VLM = 1
    MANUAL = 2


# Reserved semantic ids (must match nav_constants.py MEM_VOCAB_ENTRIES).
SID_NONE = 0
SID_WALKED = 1
SID_DUST_BIN = 2


@dataclass
class CircleMemory:
    """Anchored circle in keyframe camera frame C_k.

    mu_c: center in camera frame (3,); radius: invariant world-frame radius.
    anchor_scale_ref: keyframe Sim3 scale at write time, diagnostics only.
    """

    id: int
    kf_id: int
    sid: int
    mu_c: np.ndarray  # (3,) float32
    radius: float  # world radius (m); never multiplied by Sim3 scale
    w: float = 1.0
    t_last: float = 0.0
    source: MemorySource = MemorySource.MANUAL
    anchor_scale_ref: Optional[float] = None

    def __post_init__(self):
        self.mu_c = np.asarray(self.mu_c, dtype=np.float32).reshape(3)


@dataclass
class MemoryEvent:
    """World-frame write event. Caller supplies position_W."""

    semantic: Union[str, int]
    position_W: np.ndarray  # (3,)
    radius: Optional[float] = None
    weight: float = 1.0
    timestamp: float = 0.0
    source: MemorySource = MemorySource.MANUAL

    def __post_init__(self):
        self.position_W = np.asarray(self.position_W, dtype=np.float32).reshape(3)


@dataclass
class WalkedQueryResult:
    score: float
    n_samples: int
    n_hits: int
    n_candidates: int


@dataclass
class ObjectQueryResult:
    found: bool
    mu_W: Optional[np.ndarray] = None
    score: float = 0.0
    n_gaussians_in_cluster: int = 0
    n_candidates: int = 0

    def __post_init__(self):
        if self.mu_W is not None:
            self.mu_W = np.asarray(self.mu_W, dtype=np.float32).reshape(3)


@dataclass
class WorldGaussian:
    """Circle expressed in world frame for query-time use."""

    store_index: int
    mu_W: np.ndarray
    radius_W: float
    w: float
    sid: int
    mem_id: int
    n_hits: int = 0
