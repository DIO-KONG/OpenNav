#!/usr/bin/env python
"""重定位恢复状态 (RelocState)：SLAM 丢失跟踪时的自动恢复子状态机 (停车 -> 后退离墙 -> 原地绕圈 -> 目标截胡/恢复)。"""
from __future__ import annotations

import time
import numpy as np
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import (
    RELOC_STOP_TIME,
    RELOC_BACK_TIME,
    FOLLOW_LIN_MAX,
    SCAN_ANGULAR,
    OBJ_DET_PERIOD,
    VLM_DETECT_TARGET_DEFAULT,
    VLM_PRESENCE_PROMPT,
)
from nav_vlm import qwen_bbox_to_pixel
from nav_helpers import _presence_is_yes, extract_target_3d_from_snapshot, _get_accumulated_map


class RelocState(BaseNavState):
    name: str = "RELOC"
    RELOC_MAX_TIME: float = 120.0

    def __init__(self):
        self.phase: str = "stop"          # "stop" | "back" | "spin" | "detect_hold"
        self.phase_t: float = 0.0
        self.start_t: float = 0.0
        self.last_det_t: float = 0.0

    def on_enter(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        self.phase = "stop"
        self.phase_t = snapshot.now
        self.start_t = snapshot.now
        self.last_det_t = 0.0
        ctx.invalidate_vlm("enter_reloc")
        ctx.policy.reset_presence_window("enter RELOC")
        ctx.services.motion_thread.stop()
        ctx.clear_active_path()
        ctx.reset_smoothers()
        print("[RELOC] 进入 RELOC 自动恢复: 停车")

    def on_update(self, ctx: "NavContext", snapshot: FrameSnapshot) -> StateDecision:
        from fsm.states.final_adjust import FinalAdjustState
        from fsm.states.final_follow import FinalFollowState
        from fsm.states.patrol_plan import PatrolPlanState
        from fsm.states.waiting import WaitingState
        from fsm.states.inquiry import InquiryState

        now = snapshot.now
        mode = snapshot.slam_mode
        cur_pose = snapshot.cur_pose

        # ---- 1. 检查 SLAM 是否已自主恢复 TRACKING ----
        if mode is not None and getattr(mode, "name", "") != "RELOC":
            prev_cls = ctx.reloc_prev_state_cls
            ctx.boot_scan_done = True

            if prev_cls is not None and prev_cls.__name__.startswith("Final"):
                ctx.clear_active_path()
                return StateDecision(
                    next_state=FinalFollowState,
                    command_desc="RELOC recovered -> resume FinalFollow",
                    reset_motion=True,
                )
            elif prev_cls is not None and prev_cls.__name__.startswith("Patrol"):
                ctx.clear_active_path()
                return StateDecision(
                    next_state=PatrolPlanState,
                    command_desc="RELOC recovered -> resume PatrolPlan",
                    reset_motion=True,
                )
            else:
                target_cls = prev_cls if prev_cls is not None else WaitingState
                return StateDecision(
                    next_state=target_cls,
                    command_desc=f"RELOC recovered -> {target_cls.name}",
                    reset_motion=True,
                )

        if snapshot.halted:
            return StateDecision(command_desc="ESTOP (halt)", reset_motion=True)

        elapsed = now - self.start_t

        # 超过最大重定位时间放弃自动接管
        if elapsed > self.RELOC_MAX_TIME:
            ctx.reloc_giveup = True
            return StateDecision(
                next_state=WaitingState,
                command_desc=f"RELOC timeout ({elapsed:.0f}s) -> waiting",
                reset_motion=True,
            )

        # ---- 2. 子阶段切换机 ----
        if self.phase == "stop":
            if now - self.phase_t > RELOC_STOP_TIME:
                self.phase = "back"
                self.phase_t = now
                print("[RELOC] 阶段 stop -> back (直线后退离墙)")
            return StateDecision(command_desc=f"RELOC [stop] ({elapsed:.1f}s)", reset_motion=True)

        elif self.phase == "back":
            if now - self.phase_t > RELOC_BACK_TIME:
                self.phase = "spin"
                self.phase_t = now
                ctx.services.motion_thread.stop()
                print("[RELOC] 阶段 back -> spin (原地绕圈找 tracking)")
                return StateDecision(command_desc="RELOC [back->spin]", reset_motion=True)
            return StateDecision(
                linear=-FOLLOW_LIN_MAX,
                angular=0.0,
                command_desc=f"RELOC [back] ({elapsed:.1f}s)",
            )

        elif self.phase in ("spin", "detect_hold"):
            # 处理异步 VLM 结果 (spin 期间视觉目标截胡)
            # RELOC 是高优先级恢复态，必须像 legacy 的全局结果处理一样
            # 取走并释放其它状态遗留的结果（例如刚进入 RELOC 前提交的
            # auto_detect/direction），否则 worker 会一直 busy，RELOC 无法
            # 提交自己的 presence 请求。
            vlm_res = ctx.services.vlm_worker.poll()
            if vlm_res is not None and vlm_res.epoch == ctx.vlm_epoch:
                if vlm_res.kind == "reloc_presence":
                    present = vlm_res.error is None and _presence_is_yes(vlm_res.answer)
                    ctx.policy.register_presence(present, now)
                    if present:
                        self.phase = "detect_hold"
                        self.phase_t = now
                        ctx.services.motion_thread.stop()
                        print("[RELOC][PRESENCE] 画面存在目标 -> 停止旋转并检测 bbox")
                    else:
                        print("[RELOC][PRESENCE] 画面无目标 -> 继续旋转")

                elif vlm_res.kind == "reloc_detect":
                    if vlm_res.dets and not vlm_res.error:
                        vb = vlm_res.dets[0].get("bbox_2d")
                        if vb and len(vb) == 4:
                            h, w = vlm_res.context["image_shape"]
                            x1, y1, x2, y2 = qwen_bbox_to_pixel(vb, w, h)
                            map_fallback = (
                                _get_accumulated_map(ctx.services.slam)
                                if vlm_res.context.get("allow_map_fallback")
                                else None
                            )
                            t3d = extract_target_3d_from_snapshot(
                                (x1, y1, x2, y2),
                                vlm_res.context.get("pointcloud_2d"),
                                vlm_res.context.get("image_shape"),
                                vlm_res.context.get("camera_pos_3d"),
                                nav_y=vlm_res.context.get("nav_y"),
                                allow_map_fallback=bool(vlm_res.context.get("allow_map_fallback")),
                                map_points=map_fallback,
                            )
                            if t3d is not None and ctx.policy.should_accept_final_adjust(ctx, snapshot, t3d):
                                ctx.invalidate_vlm("reloc_target_locked")
                                ctx.final_target = (t3d[0], t3d[2])
                                ctx.final_goal_nav = None
                                ctx.final_from_reloc = True
                                ctx.goal_source = "vlm_det"
                                ctx.vlm_latest = (vlm_res.dets, vlm_res.masks)
                                ctx.final_bbox_anchor = {
                                    "target_world": tuple(float(v) for v in t3d[:3]),
                                    "bbox_px": (x1, y1, x2, y2),
                                    "image_shape": tuple(vlm_res.context["image_shape"]),
                                    "camera_pose": tuple(vlm_res.context["camera_pose_3d"]),
                                    "detection": dict(vlm_res.dets[0]),
                                }
                                print(f"[RELOC] 检测到目标 ({t3d[0]:.2f},{t3d[2]:.2f}) -> 挑出 RELOC 转 FINAL_ADJUST")
                                return StateDecision(
                                    next_state=FinalAdjustState,
                                    command_desc="RELOC -> FinalAdjust (target found)",
                                    reset_motion=True,
                                )
                    self.phase = "spin"

            # 周期触发异步检测
            if self.phase == "spin" and now - self.last_det_t >= OBJ_DET_PERIOD and snapshot.img is not None and cur_pose is not None:
                self.last_det_t = now
                ctx.submit_vlm_job(
                    kind="reloc_presence",
                    img=snapshot.img,
                    target=VLM_DETECT_TARGET_DEFAULT,
                    prompt=VLM_PRESENCE_PROMPT,
                    cur_pose=cur_pose,
                    nav_y=snapshot.nav_y,
                    step=snapshot.step,
                    current_state_name="RELOC_spin",
                    allow_map_fallback=True,
                )
            elif self.phase == "detect_hold" and not ctx.services.vlm_worker.busy() and snapshot.img is not None and cur_pose is not None:
                ctx.submit_vlm_job(
                    kind="reloc_detect",
                    img=snapshot.img,
                    target=VLM_DETECT_TARGET_DEFAULT,
                    prompt=None,
                    cur_pose=cur_pose,
                    nav_y=snapshot.nav_y,
                    step=snapshot.step,
                    current_state_name="RELOC_detect_hold",
                    allow_map_fallback=True,
                )

            if self.phase == "spin":
                return StateDecision(
                    linear=0.0,
                    angular=SCAN_ANGULAR,
                    command_desc=f"RELOC [spin] ({elapsed:.1f}s)",
                )
            else:
                return StateDecision(command_desc=f"RELOC [detect_hold] ({elapsed:.1f}s)", reset_motion=True)

        return StateDecision(command_desc="RELOC (recovering)", reset_motion=True)

    def on_exit(self, ctx: "NavContext", snapshot: FrameSnapshot) -> None:
        ctx.services.motion_thread.stop()
