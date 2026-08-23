#!/usr/bin/env python
"""探索问询状态 (InquiryState)：委托当前 Policy 获取巡逻目标 (VLM 选方向 vs Frontier 选点)。"""
from __future__ import annotations

from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision


class InquiryState(BaseNavState):
    name: str = "VLM_INQUIRY"

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.services.motion_thread.stop()
        ctx.reset_smoothers()

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        return ctx.policy.handle_inquiry_step(ctx, snapshot)

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        pass
