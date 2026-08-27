#!/usr/bin/env python
"""脱困状态 (EscapeState)：机器人本体侵入障碍物时的受控直线对准/后退脱困子状态机。"""
from __future__ import annotations

import numpy as np
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import (
    FOLLOW_LIN_MAX,
    ESCAPE_PATH0_ARRIVE_EPS,
)
from nav_path import (
    check_nav_point,
    resolve_nav_goal,
    plan_goal_resolution,
    plan_path,
    make_escape_path_start_line,
    escape_to_path_start_velocity,
    escape_align_to_yaw_velocity,
    find_escape_dir,
    escape_velocity,
)


class EscapeState(BaseNavState):
    name: str = "ESCAPE"
    PLAN_FAIL_MAX: int = 10

    def __init__(self):
        self.phase: str = "plan"          # "plan" | "path0" | "path1_align" | "fallback_retreat"
        self.path0_drive_mode: str | None = None
        self.path0_anchor: tuple | None = None
        self.path0_path: list | None = None
        self.path0_line: dict | None = None
        self.path0_aligned: bool = False
        self.path1_yaw: float | None = None
        self.escape_dir: tuple | None = None
        self.plan_fail_cnt: int = 0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.phase = "plan"
        self.path0_drive_mode = None
        self.path0_anchor = None
        self.path0_path = None
        self.path0_line = None
        self.path0_aligned = False
        self.path1_yaw = None
        self.escape_dir = None
        self.plan_fail_cnt = 0
        ctx.escaping_active = True
        ctx.clear_active_path()
        ctx.services.motion_thread.stop()
        ctx.reset_smoothers()

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.waiting import WaitingState
        from fsm.states.final_plan import FinalPlanState
        from fsm.states.patrol_plan import PatrolPlanState

        cur_pose = snapshot.cur_pose
        target = ctx.escape_target
        resume_cls = ctx.resume_state_cls

        if snapshot.halted:
            return StateDecision(command_desc="ESTOP (escape paused)", reset_motion=True)

        if cur_pose is None:
            return StateDecision(command_desc="escape: waiting pose", reset_motion=True)

        if target is None or resume_cls is None:
            return StateDecision(
                next_state=WaitingState,
                command_desc="escape: missing context -> waiting",
                reset_motion=True,
            )

        fixed_y = snapshot.nav_y
        esc_obs = snapshot.obs_current
        esc_tree = snapshot.obstacle_snapshot.tree
        robot_check = check_nav_point((cur_pose[0], cur_pose[1]), snapshot.obstacle_snapshot)

        # ----------------------------------------------------
        # 子阶段 1: plan (带同目标重规划)
        # ----------------------------------------------------
        if self.phase == "plan":
            new_path, actual_goal = None, None
            target_check = check_nav_point(target, snapshot.obstacle_snapshot)

            if not target_check.safe and ctx.escape_semantic_target is not None:
                resolution = resolve_nav_goal(
                    ctx.escape_semantic_target, (cur_pose[0], cur_pose[1]), snapshot.obstacle_snapshot
                )
                plan_start_2d = (
                    (ctx.plan_pose[0], ctx.plan_pose[1]) if ctx.plan_pose is not None else None
                )
                resolved_path, actual_goal, selected = plan_goal_resolution(
                    (cur_pose[0], cur_pose[1]),
                    resolution,
                    esc_obs,
                    fixed_y=fixed_y,
                    plan_start_2d=plan_start_2d,
                    obstacle_snapshot=snapshot.obstacle_snapshot,
                )
                if selected.goal is not None and resolved_path is not None:
                    target = selected.goal
                    ctx.escape_target = target
                    new_path = resolved_path
                    ctx.escape_replan_from_semantic = False
                else:
                    ctx.escape_replan_from_semantic = True

            if ctx.escape_replan_from_semantic and not check_nav_point(target, snapshot.obstacle_snapshot).safe:
                new_path = None
            elif new_path is None:
                plan_start_2d = (
                    (ctx.plan_pose[0], ctx.plan_pose[1]) if ctx.plan_pose is not None else None
                )
                new_path, actual_goal = plan_path(
                    (cur_pose[0], cur_pose[1]),
                    target,
                    esc_obs,
                    fixed_y=fixed_y,
                    plan_start_2d=plan_start_2d,
                    obstacle_snapshot=snapshot.obstacle_snapshot,
                )

            path0_line = (
                make_escape_path_start_line(cur_pose, new_path[0])
                if new_path is not None and len(new_path) >= 1
                else None
            )

            if new_path is not None and len(new_path) >= 2 and path0_line is not None:
                ctx.path = new_path
                ctx.path_idx = 0
                self.plan_fail_cnt = 0
                self.phase = "path0"
                self.path0_drive_mode = path0_line["drive_mode"]
                self.path0_anchor = (float(new_path[0][0]), float(new_path[0][1]))
                self.path0_path = new_path
                self.path0_line = path0_line
                self.path0_aligned = False
                self.path1_yaw = None
                self.escape_dir = None
                ctx.reset_smoothers()
                return StateDecision(command_desc="intrusion: path_new ready, align/drive path0")
            else:
                ctx.clear_active_path()
                self.plan_fail_cnt += 1
                if self.plan_fail_cnt >= self.PLAN_FAIL_MAX:
                    self.phase = "fallback_retreat"
                    self.escape_dir = find_escape_dir((cur_pose[0], cur_pose[1]), esc_tree, fixed_y=fixed_y)
                    return StateDecision(command_desc="intrusion: plan failed max -> fallback retreat")
                return StateDecision(
                    command_desc=f"intrusion plan retry ({self.plan_fail_cnt})", reset_motion=True
                )

        # ----------------------------------------------------
        # 子阶段 2: path0 直行对准行驶
        # ----------------------------------------------------
        elif self.phase == "path0":
            path0_valid = (
                ctx.path is not None
                and len(ctx.path) >= 2
                and self.path0_path is ctx.path
                and self.path0_line is not None
                and self.path0_anchor == (float(ctx.path[0][0]), float(ctx.path[0][1]))
            )
            if not path0_valid:
                ctx.clear_active_path()
                self.phase = "plan"
                return StateDecision(command_desc="intrusion: path0 invalidated -> replan", reset_motion=True)

            p0_dist = float(
                np.hypot(self.path0_anchor[0] - float(cur_pose[0]), self.path0_anchor[1] - float(cur_pose[1]))
            )
            line_start = self.path0_line["start"]
            line_unit = self.path0_line["unit"]
            travel_x = float(cur_pose[0]) - float(line_start[0])
            travel_z = float(cur_pose[1]) - float(line_start[1])
            line_progress = travel_x * float(line_unit[0]) + travel_z * float(line_unit[1])
            line_remaining = float(self.path0_line["distance"]) - line_progress

            line_reached = (
                line_remaining <= ESCAPE_PATH0_ARRIVE_EPS or p0_dist <= ESCAPE_PATH0_ARRIVE_EPS
            )

            if line_reached and robot_check.safe:
                p1_dx = float(ctx.path[1][0]) - float(ctx.path[0][0])
                p1_dz = float(ctx.path[1][1]) - float(ctx.path[0][1])
                if np.hypot(p1_dx, p1_dz) <= 1e-6:
                    ctx.path_idx = 1
                    ctx.clear_escape()
                    return StateDecision(
                        next_state=resume_cls,
                        command_desc="intrusion: path0 reached -> resume path",
                        reset_motion=True,
                    )
                else:
                    self.path1_yaw = float(np.arctan2(p1_dx, p1_dz))
                    self.phase = "path1_align"
                    return StateDecision(command_desc="intrusion: path0 reached -> align path1", reset_motion=True)
            elif line_reached:
                ctx.clear_active_path()
                self.phase = "plan"
                return StateDecision(command_desc="intrusion: path0 unsafe -> replan", reset_motion=True)
            else:
                el, ea, alpha, self.path0_aligned = escape_to_path_start_velocity(
                    cur_pose, self.path0_line, line_remaining, self.path0_aligned
                )
                return StateDecision(
                    linear=el,
                    angular=ea,
                    command_desc=f"intrusion path0[{self.path0_drive_mode}] rem={line_remaining:.2f}m",
                )

        # ----------------------------------------------------
        # 子阶段 3: path1_align 对准下一点航向
        # ----------------------------------------------------
        elif self.phase == "path1_align":
            path1_valid = (
                ctx.path is not None
                and len(ctx.path) >= 2
                and self.path0_path is ctx.path
                and self.path1_yaw is not None
            )
            if not path1_valid:
                ctx.clear_active_path()
                self.phase = "plan"
                return StateDecision(command_desc="intrusion: path1 align invalid -> replan", reset_motion=True)

            el, ea, alpha, p1_aligned = escape_align_to_yaw_velocity(cur_pose, self.path1_yaw)
            if p1_aligned:
                ctx.path_idx = 1
                ctx.clear_escape()
                return StateDecision(
                    next_state=resume_cls,
                    command_desc="intrusion: path1 aligned -> resume follow",
                    reset_motion=True,
                )
            return StateDecision(
                linear=el,
                angular=ea,
                command_desc=f"intrusion path1_align err={np.degrees(alpha):.1f}°",
            )

        # ----------------------------------------------------
        # 子阶段 4: fallback_retreat 远离最近障碍盲退
        # ----------------------------------------------------
        elif self.phase == "fallback_retreat":
            if robot_check.safe:
                self.escape_dir = None
                self.plan_fail_cnt = 0
                if ctx.escape_replan_from_semantic:
                    if resume_cls.__name__.startswith("Final"):
                        ctx.clear_final(clear_target=False)
                        ctx.clear_escape()
                        return StateDecision(
                            next_state=FinalPlanState,
                            command_desc="intrusion fallback clear -> reselect final goal",
                            reset_motion=True,
                        )
                    else:
                        ctx.clear_patrol(clear_target=False)
                        ctx.clear_escape()
                        return StateDecision(
                            next_state=PatrolPlanState,
                            command_desc="intrusion fallback clear -> reselect patrol goal",
                            reset_motion=True,
                        )
                self.phase = "plan"
                return StateDecision(command_desc="intrusion fallback clear -> replan", reset_motion=True)
            else:
                if self.escape_dir is None:
                    self.escape_dir = find_escape_dir((cur_pose[0], cur_pose[1]), esc_tree, fixed_y=fixed_y)
                if self.escape_dir is not None:
                    el, ea = escape_velocity(cur_pose, self.escape_dir)
                    drive = "obstacle_away"
                else:
                    el, ea = -FOLLOW_LIN_MAX * 0.5, 0.0
                    drive = "blind_reverse"
                return StateDecision(
                    linear=el,
                    angular=ea,
                    command_desc=f"intrusion fallback[{drive}] clr={robot_check.clearance:.2f}m",
                )

        return StateDecision(command_desc="intrusion: unknown state", reset_motion=True)

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.escaping_active = False
        ctx.services.motion_thread.stop()
