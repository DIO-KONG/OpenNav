#!/usr/bin/env python
"""最终目标微调状态 (FinalAdjustState)：视觉伺服持续跟踪目标，执行 EMA 3D 坐标滤波并触发锁定。"""
from __future__ import annotations

import numpy as np
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import (
    FINAL_ADJUST_TIME,
    FINAL_ADJUST_MIN_FRAMES,
    FINAL_ADJUST_LOCK_DIST,
    FINAL_ADJUST_FORCE_TIME,
    FOLLOW_STRAIGHT_ALPHA,
    FOLLOW_DEADBAND,
    FOLLOW_WAYPOINT_THRESHOLD,
    PATROL_ARRIVE_EPS,
    ROBOT_RADIUS,
    PLAN_FAIL_MAX,
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
        if ctx.resume_from_escape:
            # legacy 恢复 ADJUST 不重进终调：保留 path / 计时 / 帧数。
            ctx.resume_from_escape = False
            ctx.reset_smoothers()
            return
        self.enter_t = snapshot.now
        self.valid_frames = 1
        self.plan_fail_cnt = 0
        ctx.final_adj_plan_fail_cnt = 0
        self.last_replan_t = snapshot.now
        ctx.clear_active_path()
        ctx.reset_smoothers()
        ctx.policy.reset_presence_window("enter FINAL_ADJUST")

    def _lock_if_ready(
        self,
        ctx: "NavContext",
        d_nav: float,
        now: float,
        *,
        linear: float = 0.0,
        angular: float = 0.0,
        reset_motion: bool = False,
    ) -> StateDecision | None:
        """与 legacy `_lock_final_adjust_if_ready` 对齐：时间/帧数/距离或 FORCE_TIMEOUT。"""
        from fsm.states.final_follow import FinalFollowState

        elapsed = now - self.enter_t
        reached_min = self.valid_frames >= FINAL_ADJUST_MIN_FRAMES
        close_enough = d_nav <= FINAL_ADJUST_LOCK_DIST
        force_timeout = elapsed >= FINAL_ADJUST_FORCE_TIME
        if not ((elapsed >= FINAL_ADJUST_TIME and reached_min and close_enough) or force_timeout):
            return None
        if force_timeout and not close_enough and ctx.final_target is not None:
            print(
                f"\033[93m[nav_2d] FINAL_ADJUST 超时强制锁定 "
                f"(目标仍远 {d_nav:.2f}m, 帧数 {self.valid_frames}, 终点 "
                f"({ctx.final_target[0]:.2f},{ctx.final_target[1]:.2f}))\033[0m"
            )
        ctx.final_from_reloc = False
        return StateDecision(
            next_state=FinalFollowState,
            linear=linear,
            angular=angular,
            reset_motion=reset_motion,
            command_desc=f"final_adjust locked ({self.valid_frames} frames / {elapsed:.1f}s) -> follow",
        )

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.final_plan import FinalPlanState
        from fsm.states.escape import EscapeState

        cur_pose = snapshot.cur_pose
        now = snapshot.now

        if snapshot.halted:
            return StateDecision(command_desc="ESTOP (halt)", reset_motion=True)

        if cur_pose is None or ctx.final_target is None:
            return StateDecision(command_desc="final_adjust: waiting pose/target", reset_motion=True)

        target = ctx.final_target
        nav_pt = ctx.final_goal_nav if ctx.final_goal_nav is not None else target
        d_tgt = float(np.hypot(nav_pt[0] - cur_pose[0], nav_pt[1] - cur_pose[1]))
        ctx.d_tgt = d_tgt

        path = ctx.path
        if path is None or len(path) == 0:
            # 本帧只规划，下一帧再跟随（legacy: path is None 分支不落入 follow）
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
                if len(path) > 1:
                    p0_dist = float(np.hypot(
                        float(path[0][0]) - float(cur_pose[0]),
                        float(path[0][1]) - float(cur_pose[1]),
                    ))
                    if p0_dist < FOLLOW_WAYPOINT_THRESHOLD:
                        ctx.path_idx = 1
                self.plan_fail_cnt = 0
                ctx.final_adj_plan_fail_cnt = 0
                # 规划帧不停底盘：legacy FINAL_ADJUST 属 moving state，上一拍 cmd_vel 继续。
                return StateDecision(command_desc="plan pending (adjust)")

            ctx.final_adj_plan_fail_cnt += 1
            if ctx.final_adj_plan_fail_cnt >= PLAN_FAIL_MAX:
                ctx.final_adj_plan_fail_cnt = 0
                ctx.final_goal_nav = None
                ctx.clear_active_path()
                return StateDecision(
                    next_state=FinalPlanState,
                    command_desc="final_adjust: repeated plan failure -> final_plan",
                    reset_motion=True,
                )
            return StateDecision(
                command_desc=f"final_adjust: path connecting ({ctx.final_adj_plan_fail_cnt})",
                reset_motion=True,
            )

        d_end = float(np.hypot(path[-1][0] - cur_pose[0], path[-1][1] - cur_pose[1]))

        # 有路径后才判入侵（legacy: path is None 只规划，不落入 follow/escape）
        robot_check = check_nav_point(
            (cur_pose[0], cur_pose[1]),
            snapshot.obstacle_snapshot,
            clearance=ROBOT_RADIUS,
        )
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

        # 到达 GOAL_NAV / 路径末点：先停车再判断锁定
        if d_tgt <= PATROL_ARRIVE_EPS or d_end <= PATROL_ARRIVE_EPS:
            locked = self._lock_if_ready(ctx, d_tgt, now, reset_motion=True)
            if locked is not None:
                return locked
            return StateDecision(command_desc="hold (adjust, at GOAL_NAV)", reset_motion=True)

        path_check = ctx.runtime.validate_remaining_path(
            path,
            ctx.path_idx,
            (cur_pose[0], cur_pose[1]),
            snapshot.obstacle_snapshot,
        )
        if not path_check.safe:
            if now - self.last_replan_t >= 1.0:
                self.last_replan_t = now
                if ctx.final_goal_nav is not None and not check_nav_point(
                    ctx.final_goal_nav, snapshot.obstacle_snapshot
                ).safe:
                    ctx.final_goal_nav = None
                ctx.clear_active_path()
                return StateDecision(
                    command_desc=f"final_adjust: path blocked ({path_check.reason}) -> replan",
                    reset_motion=True,
                )
            return StateDecision(command_desc="stop (replan cooldown)", reset_motion=True)

        sm_yaw = ctx.yaw_smoother.update(cur_pose[2])
        sm_pose = (cur_pose[0], cur_pose[1], sm_yaw)
        cmd, next_idx, lookahead, alpha = follow_path_step(path, ctx.path_idx, sm_pose)
        deadband = 0.0 if abs(alpha) > FOLLOW_STRAIGHT_ALPHA else FOLLOW_DEADBAND
        cmd = (cmd[0], ctx.out_smoother.update(cmd[1], deadband=deadband))
        ctx.path_idx = next_idx
        ctx.lookahead_target = lookahead

        if ctx.path_idx >= len(path):
            if d_tgt <= PATROL_ARRIVE_EPS:
                locked = self._lock_if_ready(ctx, d_tgt, now, reset_motion=True)
                if locked is not None:
                    return locked
                return StateDecision(command_desc="hold (adjust, at GOAL_NAV)", reset_motion=True)
            ctx.clear_active_path()
            return StateDecision(command_desc="adjust: path done, replan pending")

        # 未到 GOAL_NAV：跟随后再判断锁定（含 FORCE_TIMEOUT），保留本帧跟随速度
        locked = self._lock_if_ready(ctx, d_tgt, now, linear=cmd[0], angular=cmd[1])
        if locked is not None:
            return locked

        ctx.log_action("ADJUST", "MotionThread", f"adjusting target={target} frames={self.valid_frames}")
        return StateDecision(
            linear=cmd[0],
            angular=cmd[1],
            command_desc=f"adjust [{_cmd_str(cmd[0], cmd[1])}] rem={d_tgt:.2f}m",
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        # 不在此处停车：切 FINAL_FOLLOW 时需保持本帧跟随速度；
        # 切 ESCAPE / FINAL_PLAN 的决策已带 reset_motion=True。
        pass
