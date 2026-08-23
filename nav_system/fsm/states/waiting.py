#!/usr/bin/env python
"""等待状态 (WaitingState)：等待 SLAM 与位姿就绪，触发开局建图扫描或初始探测。"""
from __future__ import annotations

from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import BOOT_SCAN_ENABLED


class WaitingState(BaseNavState):
    name: str = "WAITING"

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.services.motion_thread.stop()
        ctx.reset_smoothers()

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.scanning import ScanningState
        from fsm.states.inquiry import InquiryState

        cur_pose = snapshot.cur_pose
        mode = snapshot.slam_mode

        if not ctx.boot_scan_done and cur_pose is not None:
            if mode is not None and getattr(mode, "name", "") == "TRACKING":
                if BOOT_SCAN_ENABLED:
                    ctx.boot_scan_done = True
                    return StateDecision(
                        next_state=ScanningState,
                        command_desc="waiting -> scanning (boot scan)",
                        reset_motion=True,
                    )
                else:
                    ctx.boot_scan_done = True
                    ctx.auto_vlm_asked = False
                    ctx.vlm_reask_pending = False
                    return StateDecision(
                        next_state=InquiryState,
                        command_desc="waiting -> inquiry (boot scan disabled)",
                        reset_motion=True,
                    )

        return StateDecision(command_desc="idle (waiting for pose/ready)", reset_motion=True)
