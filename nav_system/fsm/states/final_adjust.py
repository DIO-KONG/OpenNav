#!/usr/bin/env python
"""最终目标微调状态 (FinalAdjustState)：视觉伺服持续跟踪目标，执行 EMA 3D 坐标滤波并触发锁定。"""
from __future__ import annotations

import time
import numpy as np
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import (
    FINAL_ADJUST_TIME,
    FINAL_ADJUST_MIN_FRAMES,
    FINAL_ADJUST_LOCK_DIST,
    FINAL_ADJUST_FORCE_TIME,
    FINAL_ADJUST_SMOOTH_ALPHA,
    FINAL_ADJUST_REJECT_DIST,
)
from nav_path import check_nav_point, resolve_nav_goal, plan_goal_resolution, plan_path, follow_path_step
from nav_helpers import _cmd_str


class FinalAdjustState(BaseNavState):
    name: str = "FINAL_ADJUST"

    def __init__(self):
        self.enter_t: float = 0.0
        self.valid_frames: int = 0
        self.plan_fail_cnt: int = 0
        self.last_replan_t: float = 0.0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.enter_t = snapshot.now
        self.valid_frames = 1
        self.plan_fail_cnt = 0
        self.last_replan_t = snapshot.now
        ctx.clear_active_path()
        ctx.reset_smoothers()
        ctx.policy.reset_presence_window("enter FINAL_ADJUST")

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.final_plan import FinalPlanState
        from fsm.states.final_follow import FinalFollowState
        from fsm.states.escape import EscapeState

        cur_pose = snapshot.cur_pose
        now = snapshot.now

        if snapshot.halted:
            return StateDecision(command_desc="ESTOP (halt)", reset_motion=True)

        if cur_pose is None or ctx.final_target is None:
            return StateDecision(command_desc="final_adjust: waiting pose/target", reset_motion=True)

        target = ctx.final_target

        # 1. 判定终调锁定条件 (由时间、有效帧数、距离及超时兜底控制)
        nav_pt = ctx.final_goal_nav if ctx.final_goal_nav is not None else target
        d_nav = np.hypot(nav_pt[0] - cur_pose[0], nav_pt[1] - cur_pose[1])
        ctx.d_tgt = d_nav

        elapsed = now - self.enter_t
        reached_min_frames = self.valid_frames >= FINAL_ADJUST_MIN_FRAMES
        close_enough = d_nav <= FINAL_ADJUST_LOCK_DIST
        force_timeout = elapsed >= FINAL_ADJUST_FORCE_TIME

        if ((elapsed >= FINAL_ADJUST_TIME and reached_min_frames and close_enough) or force_timeout):
            ctx.final_from_reloc = False
            return StateDecision(
                next_state=FinalFollowState,
                command_desc=f"final_adjust locked ({self.valid_frames} frames / {elapsed:.1f}s) -> follow",
                reset_motion=True,
            )

        # 2. 障碍物入侵检测
        robot_check = check_nav_point((cur_pose[0], cur_pose[1]), snapshot.obstacle_snapshot)
        ctx.d_robot = robot_check.clearance
        if not robot_check.safe:
            ctx.resume_state_cls = FinalAdjustState
            ctx.escape_target = nav_pt
            ctx.escape_semantic_target = target
            ctx.escape_replan_from_semantic = False
            ctx.clear_active_path()
            return StateDecision(
                next_state=EscapeState,
                command_desc=f"intrusion in adjust (clr={robot_check.clearance:.2f}m) -> escape",
                reset_motion=True,
            )

        # 3. 动态规划与连接跟踪
        path = ctx.path
        if path is None or len(path) == 0:
            plan_start_2d = (
                (ctx.plan_pose[0], ctx.plan_pose[1]) if ctx.plan_pose is not None else None
            )
            if ctx.final_goal_nav is None or not check_nav_point(ctx.final_goal_nav, snapshot.obstacle_snapshot).safe:
                resolution = resolve_nav_goal(target, (cur_pose[0], cur_pose[1]), snapshot.obstacle_snapshot)
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
            else:
                self.plan_fail_cnt += 1
                return StateDecision(command_desc="final_adjust: path connecting...", reset_motion=True)

        # 4. 纯追踪低速逼近
        cmd, next_idx, lookahead = follow_path_step(
            ctx.path,
            ctx.path_idx,
            cur_pose,
            yaw_smoother=ctx.yaw_smoother,
            output_smoother=ctx.out_smoother,
            obstacle_points=snapshot.obs_current,
            obstacle_snapshot=snapshot.obstacle_snapshot,
        )
        ctx.path_idx = next_idx
        ctx.lookahead_target = lookahead

        ctx.log_action("ADJUST", "MotionThread", f"adjusting target={target} frames={self.valid_frames}")
        return StateDecision(
            linear=cmd[0],
            angular=cmd[1],
            command_desc=f"adjust [{_cmd_str(cmd[0], cmd[1])}] rem={d_nav:.2f}m",
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.services.motion_thread.stop()
