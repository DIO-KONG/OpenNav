#!/usr/bin/env python
"""扫描状态 (ScanningState)：原地旋转 360° 进行全景环境感知与建图。"""
from __future__ import annotations

import numpy as np
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import SCAN_ANGULAR


class ScanningState(BaseNavState):
    name: str = "SCANNING"

    def __init__(self):
        self.scan_accumulated: float = 0.0
        self.scan_prev_yaw: float | None = None

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.scan_accumulated = 0.0
        self.scan_prev_yaw = None
        ctx.policy.reset_presence_window("enter SCANNING")

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.inquiry import InquiryState
        from fsm.states.patrol_plan import PatrolPlanState
        from fsm.states.final_plan import FinalPlanState

        cur_pose = snapshot.cur_pose
        if cur_pose is None:
            return StateDecision(command_desc="scanning: pose lost, stop", reset_motion=True)

        if snapshot.halted:
            return StateDecision(command_desc="ESTOP (halt)", reset_motion=True)

        # 累计旋转角度
        if self.scan_prev_yaw is not None:
            d = cur_pose[2] - self.scan_prev_yaw
            d = (d + np.pi) % (2 * np.pi) - np.pi
            self.scan_accumulated += abs(d)
        self.scan_prev_yaw = cur_pose[2]

        if self.scan_accumulated >= 2 * np.pi:
            ctx.log_action("SCAN", "MotionThread", "scan finished 360°")
            if ctx.patrol_target is not None:
                ctx.clear_patrol(clear_target=False)
                ctx.clear_active_path()
                return StateDecision(
                    next_state=PatrolPlanState,
                    command_desc="scan completed -> resume patrol plan",
                    reset_motion=True,
                )
            elif ctx.final_target is not None:
                ctx.clear_final(clear_target=False)
                ctx.clear_active_path()
                return StateDecision(
                    next_state=FinalPlanState,
                    command_desc="scan completed -> resume final plan",
                    reset_motion=True,
                )
            else:
                ctx.auto_vlm_asked = False
                ctx.vlm_reask_pending = False
                return StateDecision(
                    next_state=InquiryState,
                    command_desc="scan completed -> inquiry",
                    reset_motion=True,
                )

        ctx.log_action("SCAN", "MotionThread(continuous cmd_vel)", f"angular={SCAN_ANGULAR:.2f}")
        return StateDecision(
            linear=0.0,
            angular=SCAN_ANGULAR,
            command_desc=f"scan ({np.degrees(self.scan_accumulated):.0f}/360°)",
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.scan_accumulated = 0.0
        self.scan_prev_yaw = None
        ctx.services.motion_thread.stop()
