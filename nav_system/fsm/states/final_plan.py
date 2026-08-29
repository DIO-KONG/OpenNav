#!/usr/bin/env python
"""最终目标规划状态 (FinalPlanState)：解析锁定目标的安全站位并进行全局 RRT* 规划。"""
from __future__ import annotations

from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_path import check_nav_point, resolve_nav_goal, plan_goal_resolution, plan_path
from nav_constants import PLAN_FAIL_MAX


class FinalPlanState(BaseNavState):
    name: str = "FINAL_PLAN"
    PLAN_RETRY_COOLDOWN: float = 2.0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.services.motion_thread.stop()

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.final_follow import FinalFollowState

        cur_pose = snapshot.cur_pose
        target = ctx.final_target
        now = snapshot.now

        if cur_pose is None or target is None:
            return StateDecision(command_desc="final_plan: waiting pose/target", reset_motion=True)

        if now < ctx.final_plan_retry_t:
            return StateDecision(command_desc="final plan retry cooldown", reset_motion=True)

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
            ctx.final_plan_fail_cnt = 0
            ctx.final_plan_retry_t = 0.0
            return StateDecision(
                next_state=FinalFollowState,
                command_desc=f"final_plan: path ready (len={len(path)})",
                reset_motion=True,
            )

        ctx.final_plan_fail_cnt += 1
        ctx.final_plan_retry_t = now + self.PLAN_RETRY_COOLDOWN
        if ctx.final_plan_fail_cnt >= PLAN_FAIL_MAX:
            ctx.final_plan_fail_cnt = 0
            ctx.clear_final(clear_target=False)
            ctx.clear_active_path()
            print(
                "[nav_2d][WARN] FINAL 路径连续失败，"
                "清除导航站位并等待地图更新后重新解析"
            )
            return StateDecision(
                command_desc="final_plan failed max; keep target, wait map",
                reset_motion=True,
            )

        return StateDecision(
            command_desc=f"final_plan: plan failed ({ctx.final_plan_fail_cnt})", reset_motion=True
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        pass
