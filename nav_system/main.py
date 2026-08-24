#!/usr/bin/env python
"""导航主程序 (Main)：基于规范有限状态机 (FSM) 与策略模式 (Policy) 的统一导航调度入口。"""
from __future__ import annotations

import argparse
import os
import sys
import time
import cv2
import numpy as np
import rospy

# 保证当前模块与项目根目录均在 sys.path 中
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) if os.path.basename(_CURRENT_DIR) == "nav_system" else _CURRENT_DIR
for p in (_CURRENT_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from nav_constants import (
    VLM_URL_DEFAULT,
    VLM_DETECT_TARGET_DEFAULT,
    VLM_PRESENCE_PROMPT,
    OBJ_DET_PERIOD,
    FINAL_ADJUST_DET_PERIOD,
    FINAL_ADJUST_SMOOTH_ALPHA,
    FINAL_ADJUST_REJECT_DIST,
    ROBOT_RADIUS,
)
from nav_control import mock
from nav_helpers import (
    get_keyframe_pose,
    build_obstacle_snapshot,
    query_frontier_for_viz,
    _presence_is_yes,
    extract_target_3d_from_snapshot,
    _get_accumulated_map,
)
from nav_vlm import qwen_bbox_to_pixel
from nav_page import draw_debug_overlay, draw_map_view

from nav_services import bootstrap_services, NavServices
from fsm.context import NavContext
from fsm.engine import NavigationStateMachine
from fsm.decision import FrameSnapshot, StateDecision
from fsm.states.waiting import WaitingState
from fsm.states.reloc import RelocState
from fsm.states.final_adjust import FinalAdjustState
from fsm.states.done import DoneState

from policies.base_policy import BaseNavigationPolicy
from policies.object_policy import ObjectNavPolicy
from policies.open_policy import OpenNavPolicy
from policies.frontier_policy import FrontierNavPolicy


def create_policy(mode: str) -> BaseNavigationPolicy:
    """根据运行参数构造对应的导航策略。"""
    mode_lower = mode.lower().strip()
    if mode_lower == "open":
        return OpenNavPolicy()
    elif mode_lower == "frontier":
        return FrontierNavPolicy()
    elif mode_lower in ("object", "auto", "default"):
        return ObjectNavPolicy()
    else:
        print(f"[main][WARN] 未知 mode '{mode}', 回退到 'object' 模式")
        return ObjectNavPolicy()


def handle_global_auto_detection(
    ctx: NavContext, snapshot: FrameSnapshot, fsm: NavigationStateMachine
) -> None:
    """全局异步目标检测任务提交与结果消费 (视觉截胡切入 FINAL_ADJUST)。"""
    now = snapshot.now
    img = snapshot.img
    cur_pose = snapshot.cur_pose
    in_adjust = isinstance(fsm.current_state, FinalAdjustState)
    in_reloc = isinstance(fsm.current_state, RelocState)
    in_done = isinstance(fsm.current_state, DoneState)

    if in_reloc or in_done:
        return

    # 1. 消费已完成的异步检测任务
    vlm_res = ctx.services.vlm_worker.poll()
    if vlm_res is not None and vlm_res.kind == "auto_detect":
        ctx.auto_det_t = float(vlm_res.finished_at)
        if vlm_res.epoch == ctx.vlm_epoch and not vlm_res.error:
            res_ctx = vlm_res.context
            need_presence = bool(res_ctx.get("need_presence", True))
            present = (not need_presence) or _presence_is_yes(vlm_res.answer)

            if need_presence:
                ctx.policy.register_presence(present, now)

            if present and not in_adjust:
                ctx.presence_confirmed = True

            if present and vlm_res.dets:
                vb = vlm_res.dets[0].get("bbox_2d")
                if vb and len(vb) == 4:
                    h, w = res_ctx["image_shape"]
                    x1, y1, x2, y2 = qwen_bbox_to_pixel(vb, w, h)
                    map_fallback = (
                        _get_accumulated_map(ctx.services.slam)
                        if res_ctx.get("allow_map_fallback")
                        else None
                    )
                    t3d = extract_target_3d_from_snapshot(
                        (x1, y1, x2, y2),
                        res_ctx.get("pointcloud_2d"),
                        res_ctx.get("image_shape"),
                        res_ctx.get("camera_pos_3d"),
                        nav_y=res_ctx.get("nav_y"),
                        allow_map_fallback=bool(res_ctx.get("allow_map_fallback")),
                        map_points=map_fallback,
                    )
                    if t3d is not None:
                        # 正在终调中：执行 EMA 平滑滤波
                        if in_adjust:
                            if ctx.final_from_reloc or ctx.final_target is None:
                                ctx.final_target = (t3d[0], t3d[2])
                                ctx.final_goal_nav = None
                                fsm.current_state.valid_frames += 1
                                ctx.clear_active_path()
                            else:
                                dx = t3d[0] - ctx.final_target[0]
                                dz = t3d[2] - ctx.final_target[1]
                                jump = (dx * dx + dz * dz) ** 0.5
                                if jump <= FINAL_ADJUST_REJECT_DIST:
                                    ctx.final_target = (
                                        FINAL_ADJUST_SMOOTH_ALPHA * t3d[0]
                                        + (1.0 - FINAL_ADJUST_SMOOTH_ALPHA) * ctx.final_target[0],
                                        FINAL_ADJUST_SMOOTH_ALPHA * t3d[2]
                                        + (1.0 - FINAL_ADJUST_SMOOTH_ALPHA) * ctx.final_target[1],
                                    )
                                    ctx.final_goal_nav = None
                                    fsm.current_state.valid_frames += 1
                                    ctx.clear_active_path()
                                else:
                                    print(f"[nav_auto][DET] EMA 拒收离群测量 (跳变 {jump:.2f}m > {FINAL_ADJUST_REJECT_DIST:.2f}m)")

                        # 巡逻或探索中检测到目标：经由 Policy 门控裁决是否切入终调
                        elif ctx.policy.should_accept_final_adjust(ctx, snapshot, t3d):
                            ctx.invalidate_vlm("new_final_target")
                            ctx.final_target = (t3d[0], t3d[2])
                            ctx.final_goal_nav = None
                            ctx.goal_source = "vlm_det"
                            ctx.vlm_latest = (vlm_res.dets, vlm_res.masks)
                            ctx.final_bbox_anchor = {
                                "target_world": tuple(float(v) for v in t3d[:3]),
                                "bbox_px": (x1, y1, x2, y2),
                                "image_shape": tuple(res_ctx["image_shape"]),
                                "camera_pose": tuple(res_ctx["camera_pose_3d"]),
                                "detection": dict(vlm_res.dets[0]),
                            }
                            print(f"[nav_auto][DET] VLM 目标 ({t3d[0]:.2f},{t3d[2]:.2f}) -> FINAL_ADJUST")
                            fsm.change_state(FinalAdjustState, snapshot)

    # 2. 周期提交检测任务
    det_period = FINAL_ADJUST_DET_PERIOD if in_adjust else OBJ_DET_PERIOD
    should_det = (ctx.final_target is None and not in_adjust) or in_adjust
    if (
        now - ctx.auto_det_t >= det_period
        and should_det
        and img is not None
        and cur_pose is not None
    ):
        need_presence = ctx.policy.should_query_presence(ctx, in_adjust)
        ctx.submit_vlm_job(
            kind="auto_detect",
            img=img,
            target=VLM_DETECT_TARGET_DEFAULT,
            prompt=VLM_PRESENCE_PROMPT if need_presence else None,
            cur_pose=cur_pose,
            nav_y=snapshot.nav_y,
            step=snapshot.step,
            current_state_name=fsm.current_state.name,
            allow_map_fallback=in_adjust,
            need_presence=need_presence,
        )


def main():
    parser = argparse.ArgumentParser(description="模块化导航主程序 (FSM 架构)")
    parser.add_argument(
        "--mode",
        default="object",
        choices=["object", "open", "frontier"],
        help="导航任务模式 (默认: object)",
    )
    parser.add_argument(
        "--vlm-url",
        default=VLM_URL_DEFAULT,
        help=f"VLM 服务地址 (默认: {VLM_URL_DEFAULT})",
    )
    args = parser.parse_args()

    print(f"[main] 正在启动导航系统，模式: {args.mode.upper()}")
    policy = create_policy(args.mode)
    services = bootstrap_services(vlm_url=args.vlm_url)
    context = NavContext(services, policy)
    fsm = NavigationStateMachine(context=context, initial_state_cls=WaitingState)

    # 障碍物快照与导航高度缓存
    nav_y = 0.0
    obstacle_points_id = None
    obstacle_revision = 0
    obstacle_snapshot = build_obstacle_snapshot(None, fixed_y=0.0, revision=0)

    step = 0
    prev_t = time.time()
    fps = 0.0

    print("[main] 有限状态机与后台服务已就绪，进入 10Hz 主循环...")

    try:
        while not rospy.is_shutdown() and not services.nav_stop_event.is_set():
            step += 1
            now = time.time()
            dt = max(1e-4, now - prev_t)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_t = now

            # ---- 1. 采集当前帧基础数据 ----
            img = services.slam.get_img()
            raw_pose = services.slam.get_pose()
            kf_pose = get_keyframe_pose(services.slam)
            slam_mode = services.slam.get_mode()
            odom_now = services.odom_holder.get()

            # 更新导航高度 Y
            if raw_pose is not None:
                pose_full = services.slam.get_pose_full()
                if pose_full is not None:
                    nav_y = float(pose_full[1])
            services.pc_cache.set_nav_y(nav_y)

            # 点云缓存与不可变障碍物快照构建
            map_pts, obs_current, n_map = services.pc_cache.snapshot()
            if id(obs_current) != obstacle_points_id:
                obstacle_points_id = id(obs_current)
                obstacle_revision += 1
                obstacle_snapshot = build_obstacle_snapshot(
                    obs_current, fixed_y=nav_y, revision=obstacle_revision
                )

            # 状态内部预转向活动状态探测
            preturn_active = getattr(fsm.current_state, "preturn_active", False)
            preturn_slam = getattr(fsm.current_state, "preturn_slam_anchor", None)
            preturn_odom = getattr(fsm.current_state, "preturn_odom_anchor", None)

            # 融合位姿推算
            cur_pose, plan_pose, use_odom, nav_source = context.update_pose_estimation(
                raw_pose=raw_pose,
                kf_pose=kf_pose,
                mode=slam_mode,
                odom_now=odom_now,
                preturn_active=preturn_active,
                preturn_slam_anchor=preturn_slam,
                preturn_odom_anchor=preturn_odom,
            )

            # 组装不可变单帧快照
            snapshot = FrameSnapshot(
                step=step,
                now=now,
                dt=dt,
                cur_pose=cur_pose,
                raw_pose=raw_pose,
                kf_pose=kf_pose,
                slam_mode=slam_mode,
                odom_pose=odom_now,
                nav_y=nav_y,
                obstacle_snapshot=obstacle_snapshot,
                obs_current=obs_current,
                n_map_points=n_map,
                halted=mock.get_halt(),
                img=img,
            )

            # ---- 2. 全局高优先级拦截：SLAM 丢失进入 RELOC 恢复 ----
            if (
                slam_mode is not None
                and getattr(slam_mode, "name", "") == "RELOC"
                and not isinstance(fsm.current_state, RelocState)
                and not context.reloc_giveup
            ):
                context.reloc_prev_state_cls = type(fsm.current_state)
                fsm.change_state(RelocState, snapshot)

            # ---- 3. 全局目标检测 (视觉截胡) ----
            handle_global_auto_detection(context, snapshot, fsm)

            # ---- 4. 状态机单步演进 ----
            decision = fsm.step(snapshot)
            context.last_cmd = decision.command_desc

            # ---- 5. 底层执行与运动控制 ----
            if snapshot.halted or decision.reset_motion:
                services.motion_thread.stop()
            else:
                services.motion_thread.set_velocity(decision.linear, decision.angular)
                services.motion_thread.start()

            # ---- 6. 渲染 Web 调试视图 (Overlay 与 Top-Down 栅格地图) ----
            if DEBUG_FEATURES.get("web_ui", True):
                if img is not None:
                    overlay_lines = [
                        (f"[{fsm.current_state.name}] {decision.command_desc}", (0, 255, 0)),
                        (f"nav[{nav_source}] x={cur_pose[0]:.2f} z={cur_pose[1]:.2f} yaw={np.degrees(cur_pose[2]):.0f}°", (0, 255, 255)) if cur_pose else ("pose lost", (0, 0, 255)),
                        f"fps={fps:.1f}  n_kf={services.slam.num_keyframes()}",
                    ]
                    overlay_frame = draw_debug_overlay(
                        img,
                        {
                            "lines": overlay_lines,
                            "rects": [],
                            "robot_pose": cur_pose,
                            "using_odom": use_odom,
                            "vlm_dets": context.vlm_latest[0],
                            "vlm_masks": context.vlm_latest[1],
                        },
                    )
                    mock.set_frame(overlay_frame)

                frontier_viz = query_frontier_for_viz(
                    services.slam.memory,
                    pose_xz=(cur_pose[0], cur_pose[1]) if cur_pose else None,
                    nav_y=nav_y,
                    obstacle_points=obs_current,
                    robot_radius=ROBOT_RADIUS,
                    yaw=float(cur_pose[2]) if cur_pose else None,
                )

                memory_gaussians, mem_shape, mem_kappa, n_memory_walked = None, None, None, 0
                if services.slam.memory and services.slam.memory.enable:
                    memory_gaussians = services.slam.memory.get_all_for_viz()
                    mem_shape = getattr(services.slam.memory, "_default_shape", None)
                    mem_kappa = getattr(services.slam.memory, "_default_kappa", None)
                    n_memory_walked = services.slam.memory.count()

                map_frame = draw_map_view({
                    "obstacle_points": obs_current,
                    "memory_gaussians": memory_gaussians,
                    "memory_shape": mem_shape,
                    "memory_kappa": mem_kappa,
                    "memory_alpha": 0.30,
                    "n_memory_walked": n_memory_walked,
                    "frontier_xz": frontier_viz.get("frontier_xz"),
                    "frontier_dist_m": frontier_viz.get("frontier_dist_m"),
                    "frontier_ms": frontier_viz.get("frontier_ms"),
                    "grid_occ_inf": frontier_viz.get("grid_occ_inf"),
                    "grid_walked": frontier_viz.get("grid_walked"),
                    "grid_n": frontier_viz.get("grid_n"),
                    "grid_origin_x": frontier_viz.get("grid_origin_x"),
                    "grid_origin_z": frontier_viz.get("grid_origin_z"),
                    "grid_res": frontier_viz.get("grid_res"),
                    "robot_pose": cur_pose,
                    "using_odom": use_odom,
                    "patrol_target": context.patrol_target,
                    "patrol_nav": context.patrol_goal_nav,
                    "final_target": context.final_target,
                    "final_nav": context.final_goal_nav,
                    "lookahead": context.lookahead_target,
                    "path": context.path,
                    "path_idx": context.path_idx,
                    "d_tgt": context.d_tgt,
                    "d_robot": context.d_robot,
                    "state": fsm.current_state.name,
                    "nav_y": nav_y,
                    "n_map_points": n_map,
                    "goal_source": context.goal_source,
                })
                mock.set_map(map_frame)

            # 任务完成正常退出
            if isinstance(fsm.current_state, DoneState):
                break

            services.rate.sleep()

    finally:
        services.stop_all()


if __name__ == "__main__":
    main()
