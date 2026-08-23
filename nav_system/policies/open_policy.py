#!/usr/bin/env python
"""Open 导航策略：开局强制扫描建图，每轮评估 presence 并在 5s / 80% 滑动窗口门控达标后才切入终调。"""
from __future__ import annotations

import time
from typing import Optional
from policies.object_policy import ObjectNavPolicy
from fsm.decision import FrameSnapshot


class OpenNavPolicy(ObjectNavPolicy):
    name: str = "open"
    default_boot_scan_done: bool = False  # 开局强制扫描建图

    PRESENCE_WINDOW: int = 10
    PRESENCE_SUSTAIN: float = 5.0
    PRESENCE_RATIO: float = 0.8

    def __init__(self):
        super().__init__()
        self._presence_samples: list[tuple[float, bool]] = []

    def should_query_presence(self, ctx: "NavContext", in_adjust: bool) -> bool:
        # 每轮都问 presence，持续累积滑动窗口
        return True

    def register_presence(self, is_present: bool, now: float) -> None:
        self._presence_samples.append((now, bool(is_present)))
        over = len(self._presence_samples) - (self.PRESENCE_WINDOW + 1)
        if over > 0:
            del self._presence_samples[:over]

    def _is_presence_locked(self, now: float) -> bool:
        n = len(self._presence_samples)
        if n < self.PRESENCE_WINDOW:
            return False
        win = self._presence_samples[-self.PRESENCE_WINDOW:]
        yes_count = sum(1 for _, p in win if p)
        ratio = yes_count / self.PRESENCE_WINDOW
        if ratio < self.PRESENCE_RATIO:
            return False
        return (now - win[0][0]) >= self.PRESENCE_SUSTAIN

    def should_accept_final_adjust(
        self, ctx: "NavContext", snapshot: FrameSnapshot, target_3d: tuple
    ) -> bool:
        # 必须满足滑动窗口门控 (5s 跨度 + 80% yes 率) 才允许切入 FINAL_ADJUST
        return self._is_presence_locked(snapshot.now)

    def reset_presence_window(self, reason: str = "") -> None:
        self._presence_samples.clear()
        if reason:
            print(f"\033[96m[nav_open][PRESENCE] 窗口清空 ({reason}) -> 样本重置\033[0m")
