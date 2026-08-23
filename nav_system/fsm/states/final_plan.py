#!/usr/bin/env python
"""最终目标规划状态 (FinalPlanState)：解析锁定目标的安全站位并进行全局 RRT* 规划。"""
from __future__ import annotations

import time
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_path import check_nav_point, resolve_nav_goal, plan_goal_resolution, plan_path


class FinalPlanState(BaseNavState):
    name: str = "FINAL_PLAN"
    PLAN_FAIL_MAX: int = 10

    def __init__(self):
        self.plan_fail_cnt: int = 0
        self.retry_t: float = 0.0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.plan_fail_cnt = 0
        ctx.services.motion_thread.stop()

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.final_follow import FinalFollowState
        from fsm.states.waiting import WaitingState

        cur_pose = snapshot.cur_pose
        target = ctx.final_target

        if cur_pose is None or target is None:
            return StateDecision(command_desc="final_plan: waiting pose/target", reset_motion=True)

        plan_start_2d = (
            (ctx.plan_pose[0], ctx.plan_pose[1]) if ctx.plan_pose is not None else None
        )

        final_nav_invalid = (
            ctx.final_goal_nav is not None
            and not check_nav_point(ctx.final_goal_nav, snapshot.obstacle_snapshot).safe
        )

        if ctx.final_goal_nav is None or final_nav_invalid:
            resolution = resolve_nav_goal(
                target, (cur_pose[0], cur_pose[1]), snapshot.obstacle_snapshot
            )
            path, actual_goal, selected = plan_goal_resolution(
                (cur_pose[0], cur_pose[1]),
                resolution,
                snapshot.obs_current,
                fixed_y=snapshot.nav_y,
                plan_start_2d=plan_start_2d,
                obstacle_snapshot=snapshot.obstacle_snapshot,
            )
            ctx.final_goal_nav = selected.goal
        else:
            path, actual_goal = plan_path(
                (cur_pose[0], cur_pose[1]),
                ctx.final_goal_nav,
                snapshot.obs_current,
                fixed_y=snapshot.nav_y,
                plan_start_2d=plan_start_2d,
                obstacle_snapshot=snapshot.obstacle_snapshot,
            )

        if path is not None and len(path) > 0:
            ctx.path = path
            ctx.path_idx = 0
            self.plan_fail_cnt = 0
            return StateDecision(
                next_state=FinalFollowState,
                command_desc=f"final_plan: path ready (len={len(path)})",
                reset_motion=True,
            )

        self.plan_fail_cnt += 1
        if self.plan_fail_cnt >= self.PLAN_FAIL_MAX:
            print(f"[nav_2d][WARN] FINAL 规划不成功 (连续 {self.plan_fail_cnt} 次失败) -> 释放目标回 WAITING")
            ctx.clear_final(clear_target=True)
            ctx.clear_active_path()
            return StateDecision(
                next_state=WaitingState,
                command_desc="final_plan failed max -> waiting",
                reset_motion=True,
            )

        return StateDecision(
            command_desc=f"final_plan: plan failed ({self.plan_fail_cnt})", reset_motion=True
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        pass
