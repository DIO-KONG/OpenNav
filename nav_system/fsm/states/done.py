#!/usr/bin/env python
"""完成状态 (DoneState)：导航任务成功结束。"""
from __future__ import annotations

from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision


class DoneState(BaseNavState):
    name: str = "DONE"

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.services.motion_thread.stop()
        ctx.clear_active_path()
        print("[nav_2d] ========================================")
        print("[nav_2d]        导航任务已圆满完成 (DONE)        ")
        print("[nav_2d] ========================================")

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        return StateDecision(command_desc="DONE (task completed)", reset_motion=True)

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        pass
