#!/usr/bin/env python
"""巡逻规划状态 (PatrolPlanState)：解析巡逻目标的安全站位并进行 RRT* 路径规划。"""
from __future__ import annotations

import time
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_path import check_nav_point, resolve_nav_goal, plan_goal_resolution, plan_path
from nav_helpers import auto_trigger_patrol


class PatrolPlanState(BaseNavState):
    name: str = "PATROL_PLAN"
    PLAN_FAIL_MAX: int = 10

    def __init__(self):
        self.plan_fail_cnt: int = 0
        self.frontier_retry_t: float = 0.0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.plan_fail_cnt = 0
        ctx.services.motion_thread.stop()

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.patrol_follow import PatrolFollowState

        cur_pose = snapshot.cur_pose
        target = ctx.patrol_target

        if cur_pose is None or target is None:
            return StateDecision(command_desc="patrol_plan: waiting pose/target", reset_motion=True)

        # 检查现有安全站位是否依然有效
        patrol_nav_invalid = (
            ctx.patrol_goal_nav is not None
            and not check_nav_point(ctx.patrol_goal_nav, snapshot.obstacle_snapshot).safe
        )

        plan_start_2d = (
            (ctx.plan_pose[0], ctx.plan_pose[1]) if ctx.plan_pose is not None else None
        )

        if ctx.patrol_goal_nav is None or patrol_nav_invalid:
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
            ctx.patrol_goal_nav = selected.goal
        else:
            path, actual_goal = plan_path(
                (cur_pose[0], cur_pose[1]),
                ctx.patrol_goal_nav,
                snapshot.obs_current,
                fixed_y=snapshot.nav_y,
                plan_start_2d=plan_start_2d,
                obstacle_snapshot=snapshot.obstacle_snapshot,
            )

        ctx.patrol_goal_nav_target = target

        if path is not None and len(path) > 0:
            ctx.path = path
            ctx.path_idx = 0
            self.plan_fail_cnt = 0
            return StateDecision(
                next_state=PatrolFollowState,
                command_desc=f"patrol_plan: path ready (len={len(path)})",
                reset_motion=True,
            )

        # 规划失败处理
        self.plan_fail_cnt += 1
        if self.plan_fail_cnt >= self.PLAN_FAIL_MAX:
            ft, _ = auto_trigger_patrol(
                ctx.services.slam, cur_pose, snapshot.nav_y, snapshot.obs_current
            )
            if ft is not None:
                ctx.patrol_target = ft
                ctx.patrol_source = "frontier"
                ctx.goal_source = "frontier"
                ctx.clear_active_path()
                ctx.clear_patrol(clear_target=False)
                self.plan_fail_cnt = 0
                return StateDecision(
                    command_desc=f"patrol_plan: RRT failed -> fallback frontier {ft}"
                )
            elif time.time() - self.frontier_retry_t >= 2.0:
                self.frontier_retry_t = time.time()
                print(f"[nav_2d][WARN] PATROL 规划不成功 (连续 {self.plan_fail_cnt} 次失败) 且无 frontier，等待地图更新")

        return StateDecision(
            command_desc=f"patrol_plan: plan failed ({self.plan_fail_cnt})",
            reset_motion=True,
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        pass
