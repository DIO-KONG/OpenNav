#!/usr/bin/env python
"""Frontier 纯几何探索导航策略：不调用 VLM 方向问询，INQUIRY 阶段直接从栅格中检索 Frontier。"""
from __future__ import annotations

import time
from policies.base_policy import BaseNavigationPolicy
from fsm.decision import FrameSnapshot, StateDecision
from nav_helpers import auto_trigger_patrol


class FrontierNavPolicy(BaseNavigationPolicy):
    name: str = "frontier"
    default_boot_scan_done: bool = False  # 开局强制扫描建图

    def __init__(self):
        self._frontier_query_t: float = 0.0

    def should_query_presence(self, ctx: "NavContext", in_adjust: bool) -> bool:
        return not (ctx.presence_confirmed and not in_adjust)

    def should_accept_final_adjust(
        self, ctx: "NavContext", snapshot: FrameSnapshot, target_3d: tuple
    ) -> bool:
        return True

    def handle_inquiry_step(
        self, ctx: "NavContext", snapshot: FrameSnapshot
    ) -> StateDecision:
        from fsm.states.patrol_plan import PatrolPlanState

        cur_pose = snapshot.cur_pose
        if cur_pose is None:
            return StateDecision(command_desc="frontier: pose lost, waiting")

        now = snapshot.now
        if now - self._frontier_query_t >= 1.0:
            self._frontier_query_t = now
            ft, _ = auto_trigger_patrol(
                ctx.services.slam, cur_pose, snapshot.nav_y, snapshot.obs_current
            )
            if ft is not None:
                ctx.set_patrol_goal(ft, "frontier")
                return StateDecision(
                    next_state=PatrolPlanState,
                    command_desc=f"frontier: selected {ft}",
                )
            else:
                return StateDecision(command_desc="frontier: no frontier, wait map")

        return StateDecision(command_desc="frontier: query cooldown")
