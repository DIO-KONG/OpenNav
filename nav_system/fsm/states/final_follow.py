#!/usr/bin/env python
"""最终目标跟随状态 (FinalFollowState)：执行最终航线跟踪，安全到达目标站位并转入完成态。"""
from __future__ import annotations

import time
import numpy as np
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import (
    ROBOT_RADIUS,
    PATROL_ARRIVE_EPS,
    PRETURN_MIN_TURN_DEG,
    PRETURN_RETREAT_DISTANCE,
    PRETURN_RETREAT_SPEED,
    PRETURN_RETREAT_TIMEOUT,
    FOLLOW_STRAIGHT_ALPHA,
    FOLLOW_DEADBAND,
)
from nav_control import _angle_diff
from nav_path import check_nav_point, check_nav_segment, follow_path_step
from nav_helpers import _cmd_str
from nav_memory.frontier_grid import detect_front_wall


class FinalFollowState(BaseNavState):
    name: str = "FINAL_FOLLOW"

    def __init__(self):
        self.preturn_active: bool = False
        self.preturn_start_pose: tuple | None = None
        self.preturn_slam_anchor: tuple | None = None
        self.preturn_odom_anchor: tuple | None = None
        self.preturn_active_time: float = 0.0
        self.preturn_last_tick: float = 0.0
        self.last_replan_t: float = 0.0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.preturn_active = False
        self.preturn_start_pose = None
        self.preturn_slam_anchor = None
        self.preturn_odom_anchor = None
        self.preturn_active_time = 0.0
        self.preturn_last_tick = snapshot.now
        ctx.reset_smoothers()

        cur_pose = snapshot.cur_pose
        path = ctx.path
        if (
            cur_pose is not None
            and path is not None
            and len(path) >= 2
            and snapshot.odom_pose is not None
            and snapshot.slam_mode is not None
            and getattr(snapshot.slam_mode, "name", "") == "TRACKING"
        ):
            dx = float(path[1][0]) - float(cur_pose[0])
            dz = float(path[1][1]) - float(cur_pose[1])
            if np.hypot(dx, dz) > 1e-6:
                target_yaw = float(np.arctan2(dx, dz))
                turn_deg = abs(float(np.degrees(_angle_diff(target_yaw, float(cur_pose[2])))))
                if turn_deg >= PRETURN_MIN_TURN_DEG:
                    wall = detect_front_wall((cur_pose[0], cur_pose[1]), cur_pose[2], snapshot.obs_current)
                    if wall.detected:
                        self.preturn_start_pose = (float(cur_pose[0]), float(cur_pose[1]), float(cur_pose[2]))
                        self.preturn_slam_anchor = self.preturn_start_pose
                        self.preturn_odom_anchor = tuple(float(v) for v in snapshot.odom_pose)
                        self.preturn_active_time = 0.0
                        self.preturn_last_tick = snapshot.now
                        self.preturn_active = True
                        ctx.post_escape = False

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.done import DoneState
        from fsm.states.escape import EscapeState
        from fsm.states.final_plan import FinalPlanState

        cur_pose = snapshot.cur_pose
        now = snapshot.now

        if snapshot.halted:
            self.preturn_last_tick = now
            return StateDecision(command_desc="ESTOP (halt)", reset_motion=True)

        if cur_pose is None:
            return StateDecision(command_desc="final_follow: pose lost, stop", reset_motion=True)

        # ----------------------------------------------------
        # 1. 预转向回退子阶段
        # ----------------------------------------------------
        if self.preturn_active:
            if snapshot.odom_pose is None or self.preturn_start_pose is None:
                self.preturn_active = False
                ctx.path_idx = 0
                return StateDecision(command_desc="preturn aborted (odom lost)", reset_motion=True)

            dt = max(0.0, now - self.preturn_last_tick)
            self.preturn_last_tick = now
            self.preturn_active_time += dt

            start_x, start_z, start_yaw = self.preturn_start_pose
            move_x = float(cur_pose[0]) - start_x
            move_z = float(cur_pose[1]) - start_z
            back_progress = max(0.0, -(move_x * np.sin(start_yaw) + move_z * np.cos(start_yaw)))

            if back_progress >= PRETURN_RETREAT_DISTANCE:
                self.preturn_active = False
                if self.preturn_slam_anchor and self.preturn_odom_anchor:
                    ctx.slam_anchor = self.preturn_slam_anchor
                    ctx.odom_anchor = self.preturn_odom_anchor
                    ctx.post_escape = True
                    ctx.post_escape_t = now

                blocked = True
                if ctx.path is not None and len(ctx.path) >= 2:
                    try:
                        blocked = not check_nav_segment((cur_pose[0], cur_pose[1]), ctx.path[1], snapshot.obstacle_snapshot).safe
                    except Exception as exc:
                        print(f"[PRETURN] 连接段复检异常 {exc!r}, 按阻挡处理")
                        blocked = True

                if blocked:
                    ctx.clear_active_path()
                    return StateDecision(
                        next_state=FinalPlanState,
                        command_desc="preturn done; reconnect blocked -> replan",
                        reset_motion=True,
                    )
                else:
                    ctx.path_idx = 1
                    ctx.reset_smoothers()
                    return StateDecision(command_desc="preturn done; follow path[1]")

            if self.preturn_active_time >= PRETURN_RETREAT_TIMEOUT:
                self.preturn_active = False
                ctx.path_idx = 0
                return StateDecision(command_desc="preturn timeout; follow original path")

            return StateDecision(
                linear=-PRETURN_RETREAT_SPEED,
                angular=0.0,
                command_desc=f"preturn retreat back={back_progress:.2f}/{PRETURN_RETREAT_DISTANCE:.2f}m",
            )

        # ----------------------------------------------------
        # 2. 到达判定
        # ----------------------------------------------------
        path = ctx.path
        if path is None or len(path) == 0:
            return StateDecision(next_state=FinalPlanState, command_desc="no path -> final_plan", reset_motion=True)

        nav_pt = ctx.final_goal_nav if ctx.final_goal_nav is not None else ctx.final_target
        if nav_pt is None:
            return StateDecision(command_desc="no target -> stop", reset_motion=True)

        d_tgt = np.hypot(nav_pt[0] - cur_pose[0], nav_pt[1] - cur_pose[1])
        d_end = np.hypot(path[-1][0] - cur_pose[0], path[-1][1] - cur_pose[1])
        ctx.d_tgt = d_tgt

        if d_tgt <= PATROL_ARRIVE_EPS or d_end <= PATROL_ARRIVE_EPS or ctx.path_idx >= len(path):
            ctx.clear_active_path()
            ctx.invalidate_vlm("done")
            return StateDecision(
                next_state=DoneState,
                command_desc="FINAL_FOLLOW -> DONE (goal reached)",
                reset_motion=True,
            )

        # ----------------------------------------------------
        # 3. 障碍物入侵检测
        # ----------------------------------------------------
        robot_check = check_nav_point((cur_pose[0], cur_pose[1]), snapshot.obstacle_snapshot, clearance=ROBOT_RADIUS)
        ctx.d_robot = robot_check.clearance
        if not robot_check.safe:
            ctx.resume_state_cls = FinalFollowState
            ctx.escape_target = nav_pt
            ctx.escape_semantic_target = ctx.final_target
            ctx.escape_replan_from_semantic = False
            ctx.clear_active_path()
            return StateDecision(
                next_state=EscapeState,
                command_desc=f"intrusion detected (clr={robot_check.clearance:.2f}m) -> escape",
                reset_motion=True,
            )

        # ----------------------------------------------------
        # 4. 剩余路径碰撞复检
        # ----------------------------------------------------
        path_check = ctx.runtime.validate_remaining_path(
            path, ctx.path_idx, (cur_pose[0], cur_pose[1]),
            snapshot.obstacle_snapshot,
        )
        if not path_check.safe:
            if now - self.last_replan_t >= 1.0:
                self.last_replan_t = now
                if ctx.final_goal_nav is not None and not check_nav_point(ctx.final_goal_nav, snapshot.obstacle_snapshot).safe:
                    ctx.final_goal_nav = None
                ctx.clear_active_path()
                return StateDecision(
                    next_state=FinalPlanState,
                    command_desc=f"final path blocked ({path_check.reason}) -> replan",
                    reset_motion=True,
                )

        # ----------------------------------------------------
        # 5. 纯追踪控制计算
        # ----------------------------------------------------
        sm_yaw = ctx.yaw_smoother.update(cur_pose[2])
        sm_pose = (cur_pose[0], cur_pose[1], sm_yaw)
        cmd, next_idx, lookahead, alpha = follow_path_step(
            path,
            ctx.path_idx,
            sm_pose,
        )
        deadband = 0.0 if abs(alpha) > FOLLOW_STRAIGHT_ALPHA else FOLLOW_DEADBAND
        cmd = (cmd[0], ctx.out_smoother.update(cmd[1], deadband=deadband))
        ctx.path_idx = next_idx
        ctx.lookahead_target = lookahead

        ctx.log_action("FINAL_FOLLOW", "MotionThread", f"following pts={len(path)} idx={ctx.path_idx}")
        return StateDecision(
            linear=cmd[0],
            angular=cmd[1],
            command_desc=f"final [{_cmd_str(cmd[0], cmd[1])}] rem={d_tgt:.2f}m",
        )

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.preturn_active = False
        ctx.services.motion_thread.stop()
