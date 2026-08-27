#!/usr/bin/env python
"""Object 导航策略：开局扫描建图，随后进行 VLM 方向问询，目标检测单次确认即切入终调。"""
from __future__ import annotations

import time
from typing import Optional
from policies.base_policy import BaseNavigationPolicy
from fsm.decision import FrameSnapshot, StateDecision
from nav_constants import (
    VLM_DETECT_TARGET_DEFAULT,
    AUTO_VLM_DIR_PROMPT,
    AUTO_VLM_RETRY_PERIOD,
    PATROL_ARRIVE_EPS,
)
from nav_helpers import (
    key_to_bbox,
    extract_target_3d_from_snapshot,
    _get_accumulated_map,
    _goal_walked,
    parse_vlm_direction,
    auto_trigger_patrol,
)


class ObjectNavPolicy(BaseNavigationPolicy):
    name: str = "object"
    default_boot_scan_done: bool = False  # 开局强制扫描建图

    def should_query_presence(self, ctx: "NavContext", in_adjust: bool) -> bool:
        # 一旦画面确认过目标存在，在非 adjust 巡逻期间不再重复携带 presence
        return not (ctx.presence_confirmed and not in_adjust)

    def should_accept_final_adjust(
        self, ctx: "NavContext", snapshot: FrameSnapshot, target_3d: tuple
    ) -> bool:
        # 单次有效 3D 检测命中即接受切入 FINAL_ADJUST
        return True

    def handle_inquiry_step(
        self, ctx: "NavContext", snapshot: FrameSnapshot
    ) -> StateDecision:
        from fsm.states.patrol_plan import PatrolPlanState

        now = snapshot.now
        img = snapshot.img
        cur_pose = snapshot.cur_pose
        nav_y = snapshot.nav_y

        if cur_pose is None or img is None:
            return StateDecision(command_desc="inquiry: waiting pose/img")

        # 1. 检查是否有已完成的异步方向问询结果
        vlm_res = ctx.services.vlm_worker.poll(kind="direction")
        vlm_key = None
        vlm_dir_ctx = None

        if vlm_res is not None and vlm_res.kind == "direction":
            ctx.auto_vlm_ask_t = float(vlm_res.finished_at)
            if vlm_res.epoch == ctx.vlm_epoch and not vlm_res.error:
                parsed_dir = parse_vlm_direction(vlm_res.answer)
                if parsed_dir == "f":
                    ft, _ = auto_trigger_patrol(
                        ctx.services.slam, cur_pose, nav_y, snapshot.obs_current
                    )
                    if ft is not None:
                        ctx.set_patrol_goal(ft, "frontier")
                        return StateDecision(
                            next_state=PatrolPlanState,
                            command_desc=f"inquiry: vlm=F -> frontier {ft}",
                        )
                elif parsed_dir in ("j", "k", "l"):
                    vlm_key = parsed_dir
                    vlm_dir_ctx = vlm_res.context
                else:
                    # legacy 在无法解析方向时会释放本轮请求并按节流重试；
                    # 若保留 True，后续帧会一直等待一个已经消费的结果。
                    ctx.auto_vlm_asked = False
                    return StateDecision(
                        command_desc="inquiry: invalid vlm direction, retry"
                    )
            else:
                ctx.auto_vlm_asked = False

        # 2. 若当前未在问询，按节流发起异步方向问询
        if not ctx.auto_vlm_asked:
            if now - ctx.auto_vlm_ask_t >= AUTO_VLM_RETRY_PERIOD:
                if ctx.submit_vlm_job(
                    kind="direction",
                    img=img,
                    target=None,
                    prompt=AUTO_VLM_DIR_PROMPT,
                    cur_pose=cur_pose,
                    nav_y=nav_y,
                    step=snapshot.step,
                    current_state_name="VLM_INQUIRY",
                    allow_map_fallback=True,
                ):
                    ctx.auto_vlm_asked = True
                    return StateDecision(command_desc="inquiry: asking vlm direction")

        # 3. 解析方向对应的 3D 世界坐标
        if vlm_key is not None and vlm_dir_ctx is not None:
            dir_h, dir_w = vlm_dir_ctx["image_shape"]
            bbox = key_to_bbox(vlm_key, dir_h, dir_w)
            if bbox is not None:
                map_fallback = (
                    _get_accumulated_map(ctx.services.slam)
                    if vlm_dir_ctx.get("allow_map_fallback")
                    else None
                )
                t3d = extract_target_3d_from_snapshot(
                    bbox,
                    vlm_dir_ctx.get("pointcloud_2d"),
                    vlm_dir_ctx.get("image_shape"),
                    vlm_dir_ctx.get("camera_pos_3d"),
                    nav_y=vlm_dir_ctx.get("nav_y"),
                    allow_map_fallback=bool(vlm_dir_ctx.get("allow_map_fallback")),
                    map_points=map_fallback,
                )
                if t3d is not None:
                    tx, tz = float(t3d[0]), float(t3d[2])
                    d_tgt = ((tx - cur_pose[0]) ** 2 + (tz - cur_pose[1]) ** 2) ** 0.5
                    walked = _goal_walked(ctx.services.slam.memory, tx, tz, nav_y)

                    if walked or d_tgt < PATROL_ARRIVE_EPS:
                        if ctx.vlm_reask_pending:
                            ft, _ = auto_trigger_patrol(
                                ctx.services.slam, cur_pose, nav_y, snapshot.obs_current
                            )
                            if ft is not None:
                                ctx.set_patrol_goal(ft, "frontier")
                                return StateDecision(
                                    next_state=PatrolPlanState,
                                    command_desc="inquiry: reask reached -> frontier",
                                )
                        else:
                            ctx.auto_vlm_asked = False
                            ctx.vlm_reask_pending = True
                            return StateDecision(
                                command_desc="inquiry: dir reached -> re-ask vlm"
                            )
                    else:
                        ctx.set_patrol_goal((tx, tz), "vlm_dir")
                        return StateDecision(
                            next_state=PatrolPlanState,
                            command_desc=f"inquiry: target=({tx:.2f},{tz:.2f})",
                        )

        return StateDecision(command_desc="inquiry: waiting vlm answer")
