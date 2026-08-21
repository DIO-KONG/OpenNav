#!/usr/bin/env python
"""nav_2d.py - 状态机导航 + Web Debug (localhost:5001)。
状态机 WAITING→SCANNING→VLM_INQUIRY→PATROL/FINAL PLAN/FOLLOW→DONE。
键盘: j/k/l=patrol 目标, o=VLM 检测触发(final 目标), m=手动, WASD=驱动, Space=急停。
坐标系: SLAM X=右 Y=下 Z=前; 导航平面 (X,Z), Y=高度。
"""

import os
import sys
import time
import signal
import threading
import rospy
import numpy as np
from nav_msgs.msg import Odometry

from mast3r_slam_wrapper import Mast3rSlamWrapper

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tracer_ros", "tracer_http_interface", "scripts"))
from tracer_http_interface.scripts.rw_api import TracerRobot

from nav_control import (MockState, NavState, mock, MotionThread, KeyboardThread,
                         MOTION_HZ, SCAN_ANGULAR, RELOC_STOP_TIME, RELOC_BACK_TIME,
                         MANUAL_HOLD_TIMEOUT, _angle_diff,
                         PointCloudCache, PC_HZ, PC_INTERVAL_DT)
import nav_path
from nav_path import (ROBOT_RADIUS, PLAN_BOX_PAD, PATROL_ARRIVE_EPS,
                      FOLLOW_LIN_MAX, FOLLOW_YAW_EMA, FOLLOW_OUT_EMA, FOLLOW_DEADBAND,
                      FOLLOW_STRAIGHT_ALPHA, FOLLOW_SLEW,
                      crop_to_box, plan_path, follow_path_step,
                      find_escape_dir, escape_velocity,
                      check_nav_point, check_nav_path,
                      GOAL_RETRIEVE_MAX)

# ============================================================
import nav_page
from nav_page import (YawSmoother, OutputSmoother,
                     draw_debug_overlay,                      draw_map_view)

# ============================================================
from nav_helpers import (
    key_to_bbox,
    run_vlm_inquiry,
    get_keyframe_pose,
    OdomHolder,
    _odom_nav_pose,
    extract_target_3d,
    filter_filament_obstacles,
    get_obstacle_points,
    query_frontier_for_viz,
    _goal_walked,
    build_obstacle_snapshot,
)
from nav_vlm import VlmDetector, qwen_bbox_to_pixel
from nav_constants import (VLM_DETECT_TARGET_DEFAULT,
                         VLM_URL_DEFAULT,
                         VLM_H_PROMPT,
                         MOBILE_SAM_CHECKPOINT_PATH,
                         BOOT_SCAN_ENABLED,
                         REPLAN_PERIOD,
                         RRT_OBSTACLE_INFLATE)

# ============================================================
# 障碍判定统一使用 nav_path 的权威接口 (与 nav_auto 一致):
#   build_obstacle_snapshot (nav_helpers) 构造 obstacle_snapshot,
#   check_nav_point / check_nav_path (nav_path) 判点/路径安全。
# 不再使用旧的 build_obstacle_tree/tree_nearest_dist + 已删除的 is_path_blocked/shift_goal_to_free。
# ============================================================

_last_action_key = None   # 模块级: 动作日志去重

# ============================================================
#  调试打印统一开关
# ============================================================
DEBUG_PRINT = False

PLAN_FAIL_MAX = 10

def _dbg(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)


_last_state_print_t = 0.0
_printed_state = None

def _prn_state(s):
    global _last_state_print_t, _printed_state
    _now = time.time()
    if s != _printed_state or _now - _last_state_print_t >= 1.0:
        print(f"[nav_2d][STATE] {s.name}", flush=True)
        _printed_state = s
        _last_state_print_t = _now


# ============================================================
DEBUG_PROBES = {
    "navstep": False,
    "pcstat":  False,
    "reloc":   False,
}

def _probe(key, msg):
    """性能探针打印: 仅当 DEBUG_PROBES[key] 为 True 时才输出。"""
    if DEBUG_PROBES.get(key):
        print(msg)


# ============================================================
_cmd_stats = None
_cmd_window_t = None
_cmd_prev_pose = None

def _rec_cmd(v, omega, pose):
    """每秒统计发给底盘的 (linear, omega) 数值范围 + cur_pose 帧间跳变幅度。
    诊断用, 默认随 DEBUG_PRINT 关闭; 需要排查抖动时把 DEBUG_PRINT 置 True。"""
    if not DEBUG_PRINT:
        return
    global _cmd_stats, _cmd_window_t, _cmd_prev_pose
    now = time.time()
    if _cmd_window_t is None:
        _cmd_window_t = now
        _cmd_stats = {"n": 0, "vmin": 9e9, "vmax": -9e9,
                      "wmin": 9e9, "wmax": -9e9, "djmax": 0.0, "dyawmax": 0.0}
    s = _cmd_stats
    if _cmd_prev_pose is not None and pose is not None:
        dj = ((pose[0] - _cmd_prev_pose[0]) ** 2 +
              (pose[1] - _cmd_prev_pose[1]) ** 2) ** 0.5
        dyaw = abs((pose[2] - _cmd_prev_pose[2] + np.pi) % (2 * np.pi) - np.pi)
        s["djmax"] = max(s["djmax"], dj)
        s["dyawmax"] = max(s["dyawmax"], dyaw)
    if pose is not None:
        _cmd_prev_pose = pose
    s["n"] += 1
    s["vmin"] = min(s["vmin"], v); s["vmax"] = max(s["vmax"], v)
    s["wmin"] = min(s["wmin"], omega); s["wmax"] = max(s["wmax"], omega)
    if now - _cmd_window_t >= 1.0:
        print(f"[CMDSTAT] n={s['n']} v=[{s['vmin']:.3f},{s['vmax']:.3f}] "
              f"omega=[{s['wmin']:+.3f},{s['wmax']:+.3f}] "
              f"posejump_max={s['djmax']*100:.1f}cm "
              f"dyaw_max={np.degrees(s['dyawmax']):.1f}deg")
        _cmd_window_t = None
        _cmd_stats = None


def log_action(tag, method, detail=""):
    """动作名 + 移动方式 仅在 (动作, 移动方式) 切换时打印一次, 方便核对
    是否用了正确的移动方式 (连续 cmd_vel 线程 vs 旧 timed_move)。

    tag    : 动作名, 如 SCAN / FOLLOW / ESTOP
    method : 移动方式, 如 MotionThread(continuous cmd_vel)
    detail : 附加信息 (速度/路点等)
    """
    global _last_action_key
    key = (tag, method)
    if key != _last_action_key:
        line = f"[nav_2d] >>> ACTION={tag}  METHOD={method}"
        if detail:
            line += f"  {detail}"
        print(line)
        _last_action_key = key


#  Debug Overlay
# ============================================================

# ============================================================
#  顶视地图可视化
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="nav_2d 状态机导航")
    ap.add_argument("--vlm-url", default=VLM_URL_DEFAULT,
                   help="Qwen VLM 服务 (OpenAI 兼容), 默认 keyboard_qwen 的默认地址")
    args = ap.parse_args()

    rospy.init_node("nav_2d", anonymous=True)

    slam = Mast3rSlamWrapper(rgb_topic="/camera_f/color/image_raw")
    slam.start()

    odom_holder = OdomHolder()
    rospy.Subscriber("/odom", Odometry, odom_holder.cb, queue_size=1)
    print("[nav_2d] Subscribed to /odom (wheel odometry fallback)")

    bot = TracerRobot(base_url="http://localhost:8080")

    # --- VLM 检测器 (o 键, 默认启用; 默认地址同 keyboard_qwen) ---
    # 本地检测 prompt 副本 (不改动 nav_constants): 必须用字面量 "{}" 占位,
    # 否则 nav_vlm.detect 的 prompt.replace("{}", target) 找不到占位符,
    # 网页填的 detect 目标词会被静默丢弃 (VLM 永远只检测默认目标)。
    VLM_DETECTION_PROMPT = "Detect {}" +"""and identify their reference designators (reference numbers), and output the results in the following JSON format:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "type_of_component", "sub_label": "Reference_designator"},
  ...
]
```"""
    vlm_detector = VlmDetector(
        vlm_url=args.vlm_url,
        det_prompt=VLM_DETECTION_PROMPT,
        sam_ckpt=MOBILE_SAM_CHECKPOINT_PATH)
    print(f"[nav_2d] VLM 检测已启用: {args.vlm_url} model=(default)")

    nav_page.mock = mock
    nav_page.start_flask(port=5001)

    keyboard = KeyboardThread(manual_lin=0.10, manual_ang=0.30)
    keyboard.start()

    _nav_stop = threading.Event()
    __sigint_timer = None

    def _on_sigint(signum, frame):
        nonlocal __sigint_timer
        print("\n[nav_2d] 收到 Ctrl+C, 正在退出...")
        _nav_stop.set()
        keyboard._running = False
        keyboard._restore_term()   # 立即恢复终端, 避免退出后无回显
        try:
            if keyboard._fd is not None:
                os.close(keyboard._fd)
        except Exception:
            pass
        try:
            motion_thread.stop()
        except Exception:
            pass
        try:
            bot.stop()
        except Exception:
            pass
        try:
            rospy.signal_shutdown("Ctrl+C")
        except Exception:
            pass
        def _force_exit():
            try:
                bot.stop()
            except Exception:
                pass
            os._exit(0)
        __sigint_timer = threading.Timer(0.5, _force_exit)
        __sigint_timer.daemon = True
        __sigint_timer.start()

    signal.signal(signal.SIGINT, _on_sigint)

    state = NavState.WAITING
    _boot_scan_done = False   # 仅开局/reloc 恢复后才允许 WAITING->SCANNING; 运行时放弃目标回 WAITING 不重扫描
    rate = rospy.Rate(10)

    # 导航变量
    path = None
    path_idx = 0
    patrol_target = None
    final_target = None
    patrol_goal_nav = None
    patrol_goal_nav_target = None
    final_goal_nav = None
    d_tgt = None
    d_robot = None
    nav_y = None
    _patrol_plan_fail_cnt = 0
    _final_plan_fail_cnt = 0
    scan_accumulated = 0.0
    scan_prev_yaw = None

    # debug 变量
    step = 0
    prev_t = time.time()
    fps = 0.0

    _nav_step_t = time.time()   # 上一次打印节流时钟
    _nav_grab_ms = 0.0
    _nav_draw_ms = 0.0
    _nav_work_ms = 0.0
    last_cmd = "idle"
    vlm_latest = (None, None)  # (dets, masks) 最新 VLM 检测结果, 供绘制
    prev_halt = False         # 上一帧的急停状态 (检测上升沿)
    prev_manual_on = False    # 上一帧的手动模式状态 (检测上升沿)
    motion_thread = MotionThread(bot, hz=MOTION_HZ)
    keyboard.motion_thread = motion_thread

    _reloc_phase = None
    _reloc_t = 0.0
    _reloc_giveup = False
    RELOC_MAX_TIME = 120.0 # 秒

    # ===== 兜底清理 (atexit) =====
    import atexit
    def _at_exit():
        try:
            motion_thread.stop()
        except Exception:
            pass
        try:
            bot.stop()
        except Exception:
            pass
        try:
            keyboard._restore_term()
        except Exception:
            pass
        try:
            slam.shutdown()
        except Exception:
            pass
        try:
            rospy.signal_shutdown("atexit cleanup")
        except Exception:
            pass
    atexit.register(_at_exit)
    yaw_smoother = YawSmoother(alpha=FOLLOW_YAW_EMA)
    out_smoother = OutputSmoother(alpha=FOLLOW_OUT_EMA,
                                 deadband=FOLLOW_DEADBAND,
                                 slew=FOLLOW_SLEW)
    _last_action_key = None
    _vlog_t = 0.0             # FOLLOW 速度日志节流时间戳
    _reloc_start_t = 0.0
    _reloc_probe_t = 0.0      # [RELOC] 探针节流时钟
    _anchor_kf = None
    _anchor_cur = None
    _kf_prev_pos = None
    _nav_smooth = None
    _odom_anchor = None
    _slam_anchor = None
    _plan_slam_anchor = None
    _plan_odom_anchor = None
    _plan_pose = None
    _nav_source = "slam"
    _escaping = False
    _escape_dir = None   # 位姿入侵脱困方向(远离最近障碍, 2D 单位向量)
    _yaw_sign = -1.0
    NAV_SMOOTH_POS = 0.30   # 位置 (x,z) 收敛系数
    NAV_SMOOTH_YAW = 0.30   # 偏航 (yaw) 收敛系数
    # 障碍快照 (与 nav_auto 一致): 取代旧的 bo_tree / tree_nearest_dist
    obstacle_snapshot = build_obstacle_snapshot(None, fixed_y=0.0, revision=0)
    _obstacle_points_id = None
    _obstacle_revision = 0

    print("[nav_2d] 启动状态机导航, 等待 SLAM 就绪...")
    print("[nav_2d] 打开浏览器查看 debug "
          "(本机 http://localhost:5001, 其他设备用本机 IP:5001)")

    pc_cache = PointCloudCache(slam, PC_INTERVAL_DT, _nav_stop,
                               get_obstacle_points, _probe)
    pc_cache.start()
    print(f"[nav_2d] 点云后台刷新已启动 (PC_HZ={PC_HZ:.1f}, 控制 10Hz)")
    _mem = getattr(slam, "memory", None)

    while (not rospy.is_shutdown()) and \
            (not _nav_stop.is_set()) and \
            (state != NavState.DONE):

        # ===== 每步: 取帧 + 取位姿 =====
        _tg0 = time.time()
        img = slam.get_img()
        raw_pose = slam.get_pose()
        kf_pose  = get_keyframe_pose(slam)
        mode = slam.get_mode()
        _nav_grab_ms = (time.time() - _tg0) * 1000.0   # [NAVSTEP] 取帧+位姿耗时
        cur_pose = raw_pose
        _kf_updated = False
        if raw_pose is not None and kf_pose is not None:
            _kf_shift = (((kf_pose[0] - (_kf_prev_pos[0] if _kf_prev_pos else kf_pose[0])) ** 2 +
                          (kf_pose[1] - (_kf_prev_pos[1] if _kf_prev_pos else kf_pose[1])) ** 2) ** 0.5)
            if _kf_prev_pos is None or _kf_shift > 0.02:
                _anchor_kf = kf_pose          # 关键帧更新 -> 重新锚定
                _anchor_cur = raw_pose
                _kf_prev_pos = (kf_pose[0], kf_pose[1])
                _kf_updated = True
            elif _anchor_kf is None:
                _anchor_kf = kf_pose
                _anchor_cur = raw_pose
                _kf_updated = True
            fx = _anchor_kf[0] + (raw_pose[0] - _anchor_cur[0])
            fz = _anchor_kf[1] + (raw_pose[1] - _anchor_cur[1])
            fyaw = _anchor_kf[2] + _angle_diff(raw_pose[2], _anchor_cur[2])
            if _nav_smooth is None:
                _nav_smooth = (fx, fz, fyaw)
            else:
                sx = _nav_smooth[0] + NAV_SMOOTH_POS * (fx - _nav_smooth[0])
                sz = _nav_smooth[1] + NAV_SMOOTH_POS * (fz - _nav_smooth[1])
                syaw = _nav_smooth[2] + NAV_SMOOTH_YAW * _angle_diff(fyaw, _nav_smooth[2])
                _nav_smooth = (sx, sz, syaw)
            cur_pose = _nav_smooth      # 软收敛后的导航位姿
        _nav_source = "slam"
        _use_odom = False
        odom_now = None
        if cur_pose is not None:
            odom_now = odom_holder.get()
            _use_odom = ((mode is not None and mode.name == "RELOC") or _escaping)
            if _use_odom and odom_now is not None:
                if _odom_anchor is None or _slam_anchor is None:
                    _slam_anchor = cur_pose
                    _odom_anchor = odom_now
                cur_pose = _odom_nav_pose(_slam_anchor, _odom_anchor, odom_now,
                                         yaw_sign=_yaw_sign)
                _nav_source = "odom"
                _nav_smooth = None
            else:
                _slam_anchor = cur_pose
                _odom_anchor = odom_now

        if _use_odom:
            _plan_pose = cur_pose
        elif cur_pose is not None and odom_now is not None:
            _plan_yaw_diff = (abs(_angle_diff(cur_pose[2], _plan_slam_anchor[2]))
                              if _plan_slam_anchor is not None else float('inf'))
            if _kf_updated or _plan_yaw_diff > 0.17:
                _plan_slam_anchor = cur_pose
                _plan_odom_anchor = odom_now
            if _plan_slam_anchor is not None and _plan_odom_anchor is not None:
                _plan_pose = _odom_nav_pose(_plan_slam_anchor, _plan_odom_anchor,
                                            odom_now, yaw_sign=_yaw_sign)
            else:
                _plan_pose = cur_pose
        else:
            _plan_pose = cur_pose
        _escaping = False
        lookahead_target = None
        d_tgt = None
        d_robot = None

        halted = mock.get_halt()
        if halted and not prev_halt:
            bot.stop()              # 上升沿立即停止
            motion_thread.stop()    # 同时停掉移动线程 (扫描/跟随)
            log_action("ESTOP", "stop", "急停上升沿")
            prev_halt = True
        elif not halted and prev_halt:
            prev_halt = False
        if halted:
            last_cmd = "ESTOP (halt)"

        manual_on = mock.get_manual_mode()

        # ===== RELOC 主动恢复 (仅旁路正常状态机, 不改动 SCANNING 本身) =====
        if (not manual_on) and (not _reloc_giveup) and \
                mode is not None and mode.name == "RELOC":
            if _reloc_phase is None:
                _reloc_phase = "stop"
                _reloc_t = time.time()
                _reloc_start_t = time.time()
                motion_thread.stop()
                print("[RELOC] 进入 RELOC 自动恢复: 停车")
                last_cmd = "RELOC (停车)"
            elif _reloc_phase == "stop" and time.time() - _reloc_t > RELOC_STOP_TIME:
                _reloc_phase = "back"          # 停稳后进入后退
                _reloc_t = time.time()
                print("[RELOC] 阶段 stop -> back (直线后退离墙)")
            elif _reloc_phase == "back":       # L1: 慢速直线后退离墙
                if not halted:
                    motion_thread.set_velocity(-FOLLOW_LIN_MAX, 0.0)
                    motion_thread.start()
                last_cmd = "RELOC (后退离墙)"
                if time.time() - _reloc_t > RELOC_BACK_TIME:
                    _reloc_phase = "spin"
                    motion_thread.stop()
                    print("[RELOC] 阶段 back -> spin (原地绕圈找 tracking)")
            elif _reloc_phase == "spin":
                if not halted:
                    motion_thread.set_velocity(0.0, SCAN_ANGULAR)
                    motion_thread.start()
                last_cmd = "RELOC (绕圈找 tracking)"
                if _reloc_start_t > 0 and (time.time() - _reloc_start_t) > RELOC_MAX_TIME:
                    _reloc_giveup = True
                    motion_thread.stop()
                    _reloc_phase = None
                    print(f"[RELOC] 警告: 持续 {(time.time()-_reloc_start_t):.0f}s "
                          f"未恢复, 放弃自动 RELOC 接管, 释放回正常流程")

            # ===== RELOC 期间也要刷新 RGB + 地图, 否则画面冻结无法 debug =====
            _reloc_elapsed = (time.time() - _reloc_start_t) if _reloc_start_t > 0 else 0.0
            _reloc_lines = [
                (f"RELOC [{_reloc_phase}] {_reloc_elapsed:.1f}s", (0, 0, 255)),
                (f"nav[{_nav_source}] x={cur_pose[0]:.2f} z={cur_pose[1]:.2f} "
                 f"yaw={np.degrees(cur_pose[2]):.0f}°", (0, 255, 255)),
            ]
            if raw_pose is not None:
                _reloc_lines.append(
                    (f"raw x={raw_pose[0]:.2f} z={raw_pose[1]:.2f} "
                     f"yaw={np.degrees(raw_pose[2]):.0f}°", (200, 200, 200)))
            _reloc_lines.append(f"n_kf={slam.num_keyframes()}")
            if img is not None:
                reloc_frame = draw_debug_overlay(
                    img, {"lines": _reloc_lines, "rects": [],
                          "robot_pose": cur_pose,
                          "using_odom": _use_odom,
                          "vlm_dets": vlm_latest[0],
                          "vlm_masks": vlm_latest[1]})
                mock.set_frame(reloc_frame)
            obs_current, _, n_map = pc_cache.snapshot()
            map_frame = draw_map_view({
                "obstacle_points": obs_current,
                "robot_pose": cur_pose,
                "using_odom": _use_odom,
                "patrol_target": patrol_target,
                "patrol_nav": patrol_goal_nav,
                "final_target": final_target,
                "final_nav": final_goal_nav,
                "lookahead": lookahead_target,
                "path": path,
                "path_idx": path_idx,
                "d_tgt": d_tgt,
                "d_robot": d_robot,
                "state": "RELOC",
                "nav_y": nav_y,
                "n_map_points": n_map,
            })
            mock.set_map(map_frame)
            if DEBUG_PROBES.get("reloc") and time.time() - _reloc_probe_t >= 1.0:
                if raw_pose is not None:
                    print(f"[RELOC] phase={_reloc_phase} t={_reloc_elapsed:.1f}s "
                          f"nav[{_nav_source}] x={cur_pose[0]:.3f} z={cur_pose[1]:.3f} "
                          f"yaw={np.degrees(cur_pose[2]):.1f}° | "
                          f"raw x={raw_pose[0]:.3f} z={raw_pose[1]:.3f} "
                          f"yaw={np.degrees(raw_pose[2]):.1f}°")
                else:
                    print(f"[RELOC] phase={_reloc_phase} t={_reloc_elapsed:.1f}s raw=N/A")
                _reloc_probe_t = time.time()
            time.sleep(0.066)
            continue
        else:
            # 正常流程用正确位姿接管重规划。
            if _reloc_phase is not None:
                motion_thread.stop()
                _reloc_phase = None
                if BOOT_SCAN_ENABLED:
                    _boot_scan_done = False   # reloc 恢复后允许重新扫描一圈建图
                    state = NavState.WAITING  # 下一帧 _boot_scan_done=False 重新 SCANNING
                    print("[RELOC] 恢复 TRACKING: 回到正常状态机重规划 (允许重新扫描)")
                else:
                    _boot_scan_done = True
                    state = NavState.VLM_INQUIRY  # 跳过重新扫描, 直接询问
                    print("[RELOC] 恢复 TRACKING: 跳过重新扫描, 直接 VLM_INQUIRY")
                last_cmd = "RELOC 恢复 (回到正常流程)"
            # 继续放行手动驱动。
            if _reloc_giveup and (mode is None or mode.name != "RELOC"):
                _reloc_giveup = False
                patrol_goal_nav = None
                final_goal_nav = None
                path = None
                path_idx = 0
                if state == NavState.PATROL_FOLLOW:
                    state = NavState.PATROL_PLAN
                elif state == NavState.FINAL_FOLLOW:
                    state = NavState.FINAL_PLAN

        # 初始化导航高度
        
        if cur_pose is not None:
            pose_full = slam.get_pose_full()
            if pose_full is not None:
                new_y = float(pose_full[1])
                if nav_y is None:
                    nav_y = new_y
                    _dbg(f"[nav_2d] 导航高度 Y={nav_y:.3f} (初始)")
                else:
                    nav_y = new_y

        pc_cache.set_nav_y(nav_y)

        # ===== 点云 + 丝状滤波: 由后台线程低频刷新并缓存, 主循环只读 =====
        _map_pts, obs_current, n_map = pc_cache.snapshot()

        # 障碍快照: 仅在点云对象变化时才重建 (与 nav_auto 一致), 取代旧的 bo_tree
        if id(obs_current) != _obstacle_points_id:
            _obstacle_points_id = id(obs_current)
            _obstacle_revision += 1
            obstacle_snapshot = build_obstacle_snapshot(
                obs_current,
                fixed_y=(nav_y if nav_y is not None else 0.0),
                revision=_obstacle_revision,
            )

        # ===== 手动 frontier 请求 (网页 f 键): 仅触发一次最近 frontier 查询并设为 patrol 目标 =====
        if mock.peek_frontier_request():
            mock.take_frontier_request()
            if cur_pose is None:
                print("[nav_2d][FRONTIER] 收到 f 请求, 但 pose 未就绪, 忽略")
            else:
                _fr_mem = getattr(slam, "memory", None)
                if _fr_mem is None:
                    print("[nav_2d][FRONTIER] 收到 f 请求, 但 MemorySystem 未就绪, 忽略")
                else:
                    _fr = query_frontier_for_viz(
                        _fr_mem,
                        pose_xz=(cur_pose[0], cur_pose[1]),
                        nav_y=nav_y if nav_y is not None else 0.0,
                        obstacle_points=obs_current,
                        robot_radius=ROBOT_RADIUS,
                        yaw=float(cur_pose[2]),
                    )
                    _fxz = _fr.get("frontier_xz")
                    if _fxz is not None:
                        _fx, _fz = float(_fxz[0]), float(_fxz[1])
                        # 注: f 键 frontier 不再做 walked 记忆门控, 直接作为 patrol 目标
                        patrol_target = (_fx, _fz)
                        _patrol_plan_fail_cnt = 0
                        patrol_goal_nav = None
                        patrol_goal_nav_target = None
                        path = None
                        path_idx = 0
                        print(f"[nav_2d][FRONTIER] 最近 frontier "
                              f"({patrol_target[0]:.2f},{patrol_target[1]:.2f}) "
                              f"dist={_fr.get('frontier_dist_m')}m -> PATROL_PLAN")
                        state = NavState.PATROL_PLAN
                    else:
                        print("[nav_2d][FRONTIER] 未找到可用 frontier, 忽略")

        # ===== VLM 目标检测 (o 键触发, 异步) =====
        # 消费一次性 o 键; VLM 默认启用 (默认地址同 keyboard_qwen)。
        # ===== VLM 目标检测 (o 键触发, 同步) =====
        # 单 client, 直接同步 detect(): 按下 o 即阻塞跑到出结果, 无需 busy 门控。
        _det_key = mock.take_det_key()
        if _det_key == "o" and img is not None:
            _det_tgt = (mock.get_detect_target() or VLM_DETECT_TARGET_DEFAULT)
            print(f"[nav_2d][DET] 触发 VLM 检测: '{_det_tgt}'")
            _vres = vlm_detector.detect(img, _det_tgt)
            _vdets, _vmasks, _vtgt = _vres
            if _vdets == "error":
                print(f"\033[91m[nav_2d][DET] VLM 错误: "
                      f"{_vmasks}\033[0m")
                vlm_latest = (None, None)
            elif _vdets:
                _vb = _vdets[0].get("bbox_2d")
                if _vb and len(_vb) == 4 and cur_pose is not None:
                    h, w = img.shape[:2]
                    _x1, _y1, _x2, _y2 = qwen_bbox_to_pixel(
                        _vb, w, h)
                    _t3d = extract_target_3d(
                        slam, (_x1, _y1, _x2, _y2), nav_y=nav_y)
                    if _t3d is not None:
                        final_target = (_t3d[0], _t3d[2])
                        _final_plan_fail_cnt = 0
                        print(f"[nav_2d][DET] VLM 目标 "
                              f"({_t3d[0]:.2f},{_t3d[2]:.2f}) "
                              f"label={_vdets[0].get('label','?')} "
                              f"-> FINAL_PLAN")
                        if (not manual_on) and state in (
                                NavState.SCANNING,
                                NavState.VLM_INQUIRY,
                                NavState.PATROL_PLAN,
                                NavState.PATROL_FOLLOW):
                            path = None
                            path_idx = 0
                            state = NavState.FINAL_PLAN
                        vlm_latest = (_vdets, _vmasks)
                    else:
                        print("[nav_2d][DET] bbox->3D 失败, 等待")
                        vlm_latest = (None, None)
                else:
                    print("[nav_2d][DET] VLM 未返回有效 bbox")
                    vlm_latest = (None, None)
            else:
                print("[nav_2d][DET] VLM 未检测到目标 "
                      f"(target='{_vtgt}')")
                vlm_latest = (None, None)

        # ===== H 键自由问答 (网页/终端 h, 仅显示, 不移动) =====
        _h_key = mock.take_h_key()
        if _h_key == "h" and img is not None:
            print(f"[nav_2d][H] 触发 VLM 问答: prompt='{VLM_H_PROMPT}'")
            _ans = vlm_detector.ask(img, VLM_H_PROMPT)
            mock.set_vlm_h_answer(_ans)
            print(f"[nav_2d][H] VLM 回答: {_ans}")

        # ===== 状态分发 =====

        if state == NavState.WAITING:
            _prn_state(state)
            last_cmd = "idle (waiting)"
            if not _boot_scan_done and cur_pose is not None:
                if mode is not None and mode.name == "TRACKING":
                    if BOOT_SCAN_ENABLED:
                        scan_accumulated = 0.0
                        scan_prev_yaw = None
                        _dbg("[nav_2d] WAITING -> SCANNING "
                             "(开局/reloc 恢复先扫描一圈建图)")
                        state = NavState.SCANNING
                        _boot_scan_done = True
                    else:
                        # flag=False: 跳过开局扫描, 直接进 VLM 询问态
                        _dbg("[nav_2d] WAITING -> VLM_INQUIRY "
                             "(boot_scan=false, 跳过开局扫描)")
                        state = NavState.VLM_INQUIRY
                        _boot_scan_done = True

        elif state == NavState.SCANNING:
            _prn_state(state)
            if cur_pose is None:
                motion_thread.stop()      # 停掉移动线程
                last_cmd = "stop (pose lost)"
                _dbg("[nav_2d] SCANNING: 当前帧 pose 丢失, 停车等待")
            else:
                if not halted:
                    motion_thread.set_velocity(0.0, SCAN_ANGULAR)
                    motion_thread.start()
                    log_action("SCAN", "MotionThread(continuous cmd_vel)",
                               f"angular={SCAN_ANGULAR:.2f}")
                # 累计旋转角度
                if scan_prev_yaw is not None:
                    d = cur_pose[2] - scan_prev_yaw
                    d = (d + np.pi) % (2 * np.pi) - np.pi
                    scan_accumulated += abs(d)
                scan_prev_yaw = cur_pose[2]

                if scan_accumulated >= 2 * np.pi:
                    motion_thread.stop()
                    _dbg(f"[nav_2d] SCANNING 完成 "
                          f"(累计 {np.degrees(scan_accumulated):.0f}°)")
                    if patrol_target is not None:
                        # reloc 恢复/已有巡逻任务: 直接恢复原巡逻 (目标已保留),
                        # 不再掉进 VLM_INQUIRY 傻等按键 -> 原 patrol 目标丢失
                        patrol_goal_nav = None
                        patrol_goal_nav_target = None
                        path = None
                        path_idx = 0
                        _patrol_plan_fail_cnt = 0
                        state = NavState.PATROL_PLAN
                        _dbg("[nav_2d] SCANNING 完成 -> 恢复 PATROL_PLAN "
                             "(保留原 patrol 目标)")
                    elif final_target is not None:
                        final_goal_nav = None
                        path = None
                        path_idx = 0
                        _final_plan_fail_cnt = 0
                        state = NavState.FINAL_PLAN
                        _dbg("[nav_2d] SCANNING 完成 -> 恢复 FINAL_PLAN "
                             "(保留原 final 目标)")
                    else:
                        state = NavState.VLM_INQUIRY
                        _dbg("[nav_2d] SCANNING 完成 -> VLM_INQUIRY "
                             "(无已有目标, 等待询问)")
                else:
                    last_cmd = (f"scan "
                                f"({np.degrees(scan_accumulated):.0f}"
                                f"/360°)")

        elif state == NavState.VLM_INQUIRY:
            _prn_state(state)
            last_cmd = "idle (vlm wait)"
            vlm_key = mock.peek_vlm_key()
            if img is None or cur_pose is None:
                if vlm_key is not None:
                    print(f"[nav_2d] VLM_INQUIRY: 方向键 '{vlm_key}' 已收到，"
                          f"但 image/pose 未就绪，等待")
                    last_cmd = f"idle (vlm wait: no image/pose)"
                pass
            elif vlm_key is not None:
                print(f"[nav_2d] VLM_INQUIRY: 收到方向键 '{vlm_key}', 开始检测目标...")
                bbox = run_vlm_inquiry(img, cur_pose)
                if bbox is None:
                    if vlm_key is not None:
                        print(f"[nav_2d] VLM_INQUIRY: 方向键 '{vlm_key}' 未找到目标 "
                              f"(未在图像中框出有效区域)")
                        last_cmd = f"idle (vlm: no target for '{vlm_key}')"
                else:
                    t3d = extract_target_3d(slam, bbox, nav_y=nav_y)
                    if t3d is not None:
                        _tx, _tz = float(t3d[0]), float(t3d[2])
                        if _goal_walked(getattr(slam, "memory", None), _tx, _tz, nav_y):
                            print(f"[nav_2d] VLM 目标 ({_tx:.2f},{_tz:.2f}) "
                                  f"已在 walked 记忆内(走过), 跳过")
                        else:
                            patrol_target = (_tx, _tz)
                            _patrol_plan_fail_cnt = 0  # 新目标, 清零失败计数
                            if (patrol_goal_nav is not None and
                                    patrol_target != patrol_goal_nav_target):
                                _dbg(f"[nav_2d] patrol 目标已更新, "
                                     f"旧 sub-opt-goal {patrol_goal_nav} 失效 "
                                     f"-> PATROL_PLAN (重新按新 patrol 计算)")
                                patrol_goal_nav = None
                                patrol_goal_nav_target = None
                                state = NavState.PATROL_PLAN
                            else:
                                _nav = (patrol_goal_nav if patrol_goal_nav is not None
                                        else patrol_target)
                                d_tgt = ((_nav[0] - cur_pose[0]) ** 2 +
                                         (_nav[1] - cur_pose[1]) ** 2) ** 0.5
                                if d_tgt < PATROL_ARRIVE_EPS:
                                    last_cmd = "idle (at patrol)"
                                    _dbg(f"[nav_2d] VLM 已在导航终点附近 "
                                         f"({d_tgt:.2f}m < {PATROL_ARRIVE_EPS:.2f}m) "
                                         f"-> 停在 patrol 等待检测 (不重规划)")
                                else:
                                    patrol_goal_nav = None   # 已远离, 重置导航终点
                                    patrol_goal_nav_target = None
                                    _dbg(f"[nav_2d] VLM 目标 "
                                         f"3D=({t3d[0]:.2f},{t3d[1]:.2f},"
                                         f"{t3d[2]:.2f}) -> 2D=({patrol_target[0]:.2f},"
                                         f"{patrol_target[1]:.2f})")
                                    _dbg("[nav_2d] VLM_INQUIRY -> PATROL_PLAN")
                                    state = NavState.PATROL_PLAN
                    else:
                        if vlm_key is not None:
                            print(f"[nav_2d] VLM_INQUIRY: 方向键 '{vlm_key}' 未找到目标 "
                                  f"(框内无法投影到 3D: 无点云/深度)")
                            last_cmd = f"idle (vlm: no 3D for '{vlm_key}')"
                        _dbg("[nav_2d] VLM bbox -> 3D 失败, 等待")

        elif state == NavState.PATROL_PLAN:
            _prn_state(state)
            last_cmd = "idle (planning)"
            if cur_pose is not None and patrol_target is not None:
                obs = obs_current
                path, patrol_goal_nav = plan_path(
                    (cur_pose[0], cur_pose[1]),
                    patrol_target, obs, fixed_y=nav_y,
                    plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                   if _plan_pose is not None else None))
                patrol_goal_nav_target = patrol_target
                if path is not None and len(path) > 0:
                    _patrol_plan_fail_cnt = 0  # 成功规划, 清零失败计数
                    path_idx = 0
                    _dbg(f"[nav_2d] PATROL_PLAN -> PATROL_FOLLOW "
                          f"({len(path)} waypoints, "
                          f"nav_goal=({patrol_goal_nav[0]:.2f},"
                          f"{patrol_goal_nav[1]:.2f})) "
                          f"[DBG] path_idx 重置为 0")
                    for _di, _dp in enumerate(path):
                        _dbg(f"[nav_2d][DBG-FOLLOW-ENTER]   wp[{_di}]="
                              f"({_dp[0]:.2f},{_dp[1]:.2f})")
                    yaw_smoother.reset()
                    out_smoother.reset()
                    state = NavState.PATROL_FOLLOW
                else:
                    _patrol_plan_fail_cnt += 1
                    if _patrol_plan_fail_cnt >= PLAN_FAIL_MAX:
                        print(f"[nav_2d][ERROR] PATROL 规划不成功 (已连续 "
                              f"{_patrol_plan_fail_cnt} 次 RRT 规划失败, "
                              f"目标不可达/被完全封锁) -> 放弃目标, 回到 WAITING")
                        _patrol_plan_fail_cnt = 0
                        patrol_target = None
                        patrol_goal_nav = None
                        patrol_goal_nav_target = None
                        state = NavState.WAITING
                    elif time.time() - nav_path._last_plan_fail_t >= 2.0:
                        _dbg(f"[nav_2d] 路径规划失败 (第 {_patrol_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试 (最新地图)")
                        nav_path._last_plan_fail_t = time.time()

        elif state == NavState.PATROL_FOLLOW:
            _prn_state(state)
            if cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_2d] pose 丢失, 停车等待")
            elif path is not None:
                _nav = (patrol_goal_nav if patrol_goal_nav is not None
                        else patrol_target)
                d_tgt = ((_nav[0] - cur_pose[0]) ** 2 +
                         (_nav[1] - cur_pose[1]) ** 2) ** 0.5
                if path is not None and len(path) > 0:
                    _de = ((path[-1][0] - cur_pose[0]) ** 2 +
                           (path[-1][1] - cur_pose[1]) ** 2) ** 0.5
                else:
                    _de = float('inf')
                # ===== 障碍判定统一用 obstacle_snapshot (与 nav_auto 一致) =====
                _robot_check = check_nav_point(
                    (cur_pose[0], cur_pose[1]), obstacle_snapshot,
                    clearance=ROBOT_RADIUS)
                d_robot = _robot_check.clearance
                # 目标再校验(方案A, 与 nav_auto 一致): 目标/终点已嵌入障碍 -> 直接重规划, 不平移
                _goal_unsafe = False
                if (patrol_target is not None and
                        not check_nav_point(patrol_target, obstacle_snapshot).safe):
                    _goal_unsafe = True
                if (patrol_goal_nav is not None and
                        not check_nav_point(patrol_goal_nav, obstacle_snapshot).safe):
                    patrol_goal_nav = None
                    _goal_unsafe = True
                if _goal_unsafe:
                    _dbg("[nav_2d][GOAL-UPDATE] 目标/终点已嵌入障碍 "
                         "-> PATROL_PLAN 重算可站立终点")
                    patrol_goal_nav_target = None
                    state = NavState.PATROL_PLAN
                    continue
                if (d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS or
                        path_idx >= len(path)):
                    motion_thread.stop()
                    _dbg(f"[nav_2d] PATROL_FOLLOW -> VLM_INQUIRY "
                          f"(到达巡逻导航站位, 距 nav={d_tgt:.2f}m / "
                          f"距路径终点={_de:.2f}m "
                          f"<= {PATROL_ARRIVE_EPS:.2f}m)")
                    state = NavState.VLM_INQUIRY
                else:
                    # 已自由 -> 复位脱困方向 (每帧重算, 无需缓存旧点)
                    if _robot_check.safe:
                        _escape_dir = None
                    now_t = time.time()
                    if now_t - _vlog_t >= 0.5:
                        _dbg(f"[nav_2d]   [DEBUG] 未到达 "
                             f"nav={d_tgt:.2f}m path_end={_de:.2f}m "
                             f"eps={PATROL_ARRIVE_EPS:.2f}m "
                             f"path_idx={path_idx}/{len(path)} "
                             f"cur_pose=({cur_pose[0]:.2f},{cur_pose[1]:.2f}) "
                             f"sub_opt={patrol_goal_nav} "
                             f"d_robot={d_robot:.2f}m")
                    _vlog_t = now_t
                    # 周期重规划已移除(与 nav_auto 一致: 仅路径被阻挡时才重规划)
                    if not _robot_check.safe:
                        # ---- 位姿入侵脱困: 沿最近障碍反方向稳定后退 (不跳变) ----
                        _escape_dir = nav_path.find_escape_dir(
                            (cur_pose[0], cur_pose[1]),
                            obstacle_snapshot.tree,
                            fixed_y=obstacle_snapshot.fixed_y)
                        if time.time() - nav_path._last_intrusion_warn_t >= 2.0:
                            if _escape_dir is not None:
                                _dbg(f"[nav_2d] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"沿障碍反方向后退脱困")
                            else:
                                _dbg(f"[nav_2d] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"无最近障碍点, 兜底直线后退")
                            nav_path._last_intrusion_warn_t = time.time()
                        if _escape_dir is not None:
                            _el, _ea = nav_path.escape_velocity(cur_pose, _escape_dir)
                            if not halted:
                                motion_thread.set_velocity(_el, _ea)
                                motion_thread.start()
                            last_cmd = f"escape->back (near obs {d_robot:.2f}m)"
                        else:
                            if not halted:
                                motion_thread.set_velocity(
                                    -FOLLOW_LIN_MAX * 0.5, 0.0)
                                motion_thread.start()
                            last_cmd = f"escape blind (near obs {d_robot:.2f}m)"
                        _escaping = True
                    elif not check_nav_path(
                            path, obstacle_snapshot, start_idx=path_idx,
                            current_pose=(cur_pose[0], cur_pose[1]),
                            clearance=RRT_OBSTACLE_INFLATE).safe:
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _dbg("[nav_2d] 路径被阻挡 -> PATROL_PLAN (重规划)")
                            nav_path._last_replan_t = now_t
                            state = NavState.PATROL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                        sm_yaw = yaw_smoother.update(cur_pose[2])
                        sm_pose = (cur_pose[0], cur_pose[1], sm_yaw)
                        cmd, path_idx, lookahead_target, alpha = follow_path_step(path, path_idx, sm_pose)
                        _db = 0.0 if abs(alpha) > FOLLOW_STRAIGHT_ALPHA else FOLLOW_DEADBAND
                        cmd = (cmd[0], out_smoother.update(cmd[1], deadband=_db))
                        _rec_cmd(cmd[0], cmd[1], cur_pose)
                        if halted:
                            motion_thread.stop()
                            last_cmd = "ESTOP (halt)"
                        else:
                            motion_thread.set_velocity(cmd[0], cmd[1])
                            motion_thread.start()
                            log_action("FOLLOW", "PurePursuit+smooth",
                                       f"lin={cmd[0]:.2f} ang={cmd[1]:.2f} "
                                       f"wp={path_idx}/{len(path)}")
                            now_t = time.time()
                            if now_t - _vlog_t >= 0.5:
                                _wp_now = (path[path_idx]
                                           if path_idx < len(path)
                                           else path[-1])
                                _d_wp = ((_wp_now[0] - cur_pose[0]) ** 2 +
                                         (_wp_now[1] - cur_pose[1]) ** 2) ** 0.5
                                _dbg(f"[nav_2d]   FOLLOW cmd "
                                      f"lin={cmd[0]:.2f} ang={cmd[1]:.2f} "
                                      f"wp={path_idx}/{len(path)} (continuous) "
                                      f"cur_pose=({cur_pose[0]:.2f},"
                                      f"{cur_pose[1]:.2f}) "
                                      f"wp_now=({_wp_now[0]:.2f},"
                                      f"{_wp_now[1]:.2f}) "
                                      f"dist_wp={_d_wp:.2f}m "
                                      f"d_tgt={d_tgt:.2f}m")
                                _vlog_t = now_t
                            last_cmd = _cmd_str(cmd[0], cmd[1])
                        if path_idx >= len(path):
                            motion_thread.stop()
                            if d_tgt <= PATROL_ARRIVE_EPS:
                                _dbg("[nav_2d] PATROL_FOLLOW -> VLM_INQUIRY "
                                      "(到达导航终点, 完成寻路)")
                                state = NavState.VLM_INQUIRY
                            else:
                                _dbg(f"[nav_2d] 路径已走完但未到达 nav 终点 "
                                      f"(距 {d_tgt:.2f}m) -> PATROL_PLAN "
                                      f"(重规划)")
                                state = NavState.PATROL_PLAN
            else:
                pass

        elif state == NavState.FINAL_PLAN:
            _prn_state(state)
            last_cmd = "idle (final plan)"
            if cur_pose is not None and final_target is not None:
                obs = obs_current
                path, final_goal_nav = plan_path(
                    (cur_pose[0], cur_pose[1]),
                    final_target, obs, fixed_y=nav_y,
                    plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                   if _plan_pose is not None else None))
                if path is not None and len(path) > 0:
                    _final_plan_fail_cnt = 0  # 成功规划, 清零失败计数
                    path_idx = 0
                    _dbg(f"[nav_2d] FINAL_PLAN -> FINAL_FOLLOW "
                          f"({len(path)} waypoints, "
                          f"nav_goal=({final_goal_nav[0]:.2f},"
                          f"{final_goal_nav[1]:.2f}))")
                    yaw_smoother.reset()
                    out_smoother.reset()
                    state = NavState.FINAL_FOLLOW
                else:
                    _final_plan_fail_cnt += 1
                    if _final_plan_fail_cnt >= PLAN_FAIL_MAX:
                        print(f"[nav_2d][ERROR] FINAL 规划不成功 (已连续 "
                              f"{_final_plan_fail_cnt} 次 RRT 规划失败, "
                              f"目标不可达/被完全封锁) -> 放弃目标, 回到 WAITING")
                        _final_plan_fail_cnt = 0
                        final_target = None
                        final_goal_nav = None
                        state = NavState.WAITING
                    elif time.time() - nav_path._last_plan_fail_t >= 2.0:
                        _dbg(f"[nav_2d] 最终路径规划失败 (第 {_final_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试 (最新地图)")
                        nav_path._last_plan_fail_t = time.time()

        elif state == NavState.FINAL_FOLLOW:
            _prn_state(state)
            if cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_2d] pose 丢失, 停车等待")
            elif path is not None:
                _nav = (final_goal_nav if final_goal_nav is not None
                        else final_target)
                d_tgt = ((_nav[0] - cur_pose[0]) ** 2 +
                         (_nav[1] - cur_pose[1]) ** 2) ** 0.5
                if path is not None and len(path) > 0:
                    _de = ((path[-1][0] - cur_pose[0]) ** 2 +
                           (path[-1][1] - cur_pose[1]) ** 2) ** 0.5
                else:
                    _de = float('inf')
                # ===== 障碍判定统一用 obstacle_snapshot (与 nav_auto 一致) =====
                _robot_check = check_nav_point(
                    (cur_pose[0], cur_pose[1]), obstacle_snapshot,
                    clearance=ROBOT_RADIUS)
                d_robot = _robot_check.clearance
                # 目标再校验(方案A, 与 nav_auto 一致): 目标/终点已嵌入障碍 -> 直接重规划, 不平移
                _goal_unsafe = False
                if (final_target is not None and
                        not check_nav_point(final_target, obstacle_snapshot).safe):
                    _goal_unsafe = True
                if (final_goal_nav is not None and
                        not check_nav_point(final_goal_nav, obstacle_snapshot).safe):
                    final_goal_nav = None
                    _goal_unsafe = True
                if _goal_unsafe:
                    _dbg("[nav_2d][GOAL-UPDATE] 目标/终点已嵌入障碍 "
                         "-> FINAL_PLAN 重算可站立终点")
                    state = NavState.FINAL_PLAN
                    continue
                if (d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS or
                        path_idx >= len(path)):
                    motion_thread.stop()
                    _dbg(f"[nav_2d] FINAL_FOLLOW -> DONE "
                          f"(到达检测目标, 距 nav={d_tgt:.2f}m / "
                          f"距路径终点={_de:.2f}m "
                          f"<= {PATROL_ARRIVE_EPS:.2f}m)")
                    state = NavState.DONE
                else:
                    # 已自由 -> 复位脱困方向 (每帧重算, 无需缓存旧点)
                    if _robot_check.safe:
                        _escape_dir = None
                    now_t = time.time()
                    if now_t - _vlog_t >= 0.5:
                        _dbg(f"[nav_2d]   [DEBUG] 未到达 "
                             f"nav={d_tgt:.2f}m path_end={_de:.2f}m "
                             f"eps={PATROL_ARRIVE_EPS:.2f}m "
                             f"path_idx={path_idx}/{len(path)} "
                             f"cur_pose=({cur_pose[0]:.2f},{cur_pose[1]:.2f}) "
                             f"sub_opt={final_goal_nav} "
                             f"d_robot={d_robot:.2f}m")
                    _vlog_t = now_t
                    # 周期重规划已移除(与 nav_auto 一致: 仅路径被阻挡时才重规划)
                    if not _robot_check.safe:
                        # ---- 位姿入侵脱困: 沿最近障碍反方向稳定后退 (不跳变) ----
                        _escape_dir = nav_path.find_escape_dir(
                            (cur_pose[0], cur_pose[1]),
                            obstacle_snapshot.tree,
                            fixed_y=obstacle_snapshot.fixed_y)
                        if time.time() - nav_path._last_intrusion_warn_t >= 2.0:
                            if _escape_dir is not None:
                                _dbg(f"[nav_2d] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"沿障碍反方向后退脱困")
                            else:
                                _dbg(f"[nav_2d] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"无最近障碍点, 兜底直线后退")
                            nav_path._last_intrusion_warn_t = time.time()
                        if _escape_dir is not None:
                            _el, _ea = nav_path.escape_velocity(cur_pose, _escape_dir)
                            if not halted:
                                motion_thread.set_velocity(_el, _ea)
                                motion_thread.start()
                            last_cmd = f"escape->back (near obs {d_robot:.2f}m)"
                        else:
                            if not halted:
                                motion_thread.set_velocity(
                                    -FOLLOW_LIN_MAX * 0.5, 0.0)
                                motion_thread.start()
                            last_cmd = f"escape blind (near obs {d_robot:.2f}m)"
                        _escaping = True
                    elif not check_nav_path(
                            path, obstacle_snapshot, start_idx=path_idx,
                            current_pose=(cur_pose[0], cur_pose[1]),
                            clearance=RRT_OBSTACLE_INFLATE).safe:
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _dbg("[nav_2d] 最终路径被阻挡 -> FINAL_PLAN (重规划)")
                            nav_path._last_replan_t = now_t
                            state = NavState.FINAL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                        sm_yaw = yaw_smoother.update(cur_pose[2])
                        sm_pose = (cur_pose[0], cur_pose[1], sm_yaw)
                        cmd, path_idx, lookahead_target, alpha = follow_path_step(path, path_idx, sm_pose)
                        _db = 0.0 if abs(alpha) > FOLLOW_STRAIGHT_ALPHA else FOLLOW_DEADBAND
                        cmd = (cmd[0], out_smoother.update(cmd[1], deadband=_db))
                        _rec_cmd(cmd[0], cmd[1], cur_pose)
                        if halted:
                            motion_thread.stop()
                            last_cmd = "ESTOP (halt)"
                        else:
                            motion_thread.set_velocity(cmd[0], cmd[1])
                            motion_thread.start()
                            log_action("FOLLOW", "PurePursuit+smooth",
                                       f"lin={cmd[0]:.2f} ang={cmd[1]:.2f} "
                                       f"wp={path_idx}/{len(path)}")
                            now_t = time.time()
                            if now_t - _vlog_t >= 0.5:
                                _wp_now = (path[path_idx]
                                           if path_idx < len(path)
                                           else path[-1])
                                _d_wp = ((_wp_now[0] - cur_pose[0]) ** 2 +
                                         (_wp_now[1] - cur_pose[1]) ** 2) ** 0.5
                                _dbg(f"[nav_2d]   FOLLOW cmd "
                                      f"lin={cmd[0]:.2f} ang={cmd[1]:.2f} "
                                      f"wp={path_idx}/{len(path)} (continuous) "
                                      f"cur_pose=({cur_pose[0]:.2f},"
                                      f"{cur_pose[1]:.2f}) "
                                      f"wp_now=({_wp_now[0]:.2f},"
                                      f"{_wp_now[1]:.2f}) "
                                      f"dist_wp={_d_wp:.2f}m "
                                      f"d_tgt={d_tgt:.2f}m")
                                _vlog_t = now_t
                            last_cmd = _cmd_str(cmd[0], cmd[1])
                        if path_idx >= len(path):
                            motion_thread.stop()
                            if d_tgt <= PATROL_ARRIVE_EPS:
                                _dbg("[nav_2d] FINAL_FOLLOW -> DONE "
                                      "(到达导航终点, 完成寻路)")
                                state = NavState.DONE
                            else:
                                _dbg(f"[nav_2d] 路径已走完但未到达 nav 终点 "
                                      f"(距 {d_tgt:.2f}m) -> FINAL_PLAN "
                                      f"(重规划)")
                                state = NavState.FINAL_PLAN

        _moving_states = (NavState.SCANNING,
                          NavState.PATROL_FOLLOW,
                          NavState.FINAL_FOLLOW)
        if (not manual_on) and \
                ((state not in _moving_states or halted)
                 and motion_thread.is_running()):
            motion_thread.stop()

        # ===== 键盘 WASD 手动覆盖 (在状态机之后, 抢占 motion_thread) =====
        if manual_on and not prev_manual_on:
            log_action("MANUAL_ON", "WASD (web/term)",
                       "进入手动遥控, 状态机运动被覆盖")
            prev_manual_on = True
        elif not manual_on and prev_manual_on:
            motion_thread.stop()
            log_action("MANUAL_OFF", "WASD (web/term)", "退出手动, 恢复状态机")
            prev_manual_on = False

        if manual_on:
            if halted:
                motion_thread.stop()
                last_cmd = "MANUAL estop (halt)"
            elif mock.get_term_driving():
                last_cmd = "MANUAL (terminal WASD)"
            else:
                lin, ang = mock.get_manual_velocity(timeout=MANUAL_HOLD_TIMEOUT)
                motion_thread.set_velocity(lin, ang)
                motion_thread.start()
                if abs(lin) < 1e-3 and abs(ang) < 1e-3:
                    last_cmd = "MANUAL idle (无按键)"
                else:
                    log_action("MANUAL",
                               "MotionThread(continuous cmd_vel)",
                               f"lin={lin:.2f} ang={ang:.2f}")
                    last_cmd = f"MANUAL lin={lin:.2f} ang={ang:.2f}"

        if halted:
            last_cmd = "ESTOP (halt)"

        # ===== FPS =====
        now = time.time()
        dt = now - prev_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_t = now
        step += 1

        # ===== Debug overlay =====
        # 计算 debug 信息
        has_img = img is not None
        has_pose = raw_pose is not None
        mode_str = mode.name if mode is not None else "N/A"

        if raw_pose is not None:
            pose_str = (f"raw x={raw_pose[0]:.2f} "
                        f"z={raw_pose[1]:.2f} "
                        f"yaw={np.degrees(raw_pose[2]):.0f}°")
            if cur_pose is not None and cur_pose is not raw_pose:
                pose_str += (f" | nav[{_nav_source}] x={cur_pose[0]:.2f} "
                             f"z={cur_pose[1]:.2f} "
                             f"yaw={np.degrees(cur_pose[2]):.0f}°"
                             f" yawsign={_yaw_sign:+.0f}")
        else:
            pose_str = "N/A"

        nav_y_str = f"{nav_y:.2f}" if nav_y is not None else "N/A"

        path_str = (f"{path_idx}/{len(path)}" if path else "none")
        blocked_str = "no"
        if path and not check_nav_path(
                path, obstacle_snapshot, start_idx=path_idx,
                current_pose=(cur_pose[0], cur_pose[1]) if cur_pose is not None
                else None,
                clearance=RRT_OBSTACLE_INFLATE).safe:
            blocked_str = "YES"

        patrol_str = (f"({patrol_target[0]:.2f},{patrol_target[1]:.2f})"
                       if patrol_target else "none")
        final_str = (f"({final_target[0]:.2f},{final_target[1]:.2f})"
                     if final_target else "none")
        patrol_nav_str = (f"({patrol_goal_nav[0]:.2f},{patrol_goal_nav[1]:.2f})"
                          if patrol_goal_nav else "=tgt")
        final_nav_str = (f"({final_goal_nav[0]:.2f},{final_goal_nav[1]:.2f})"
                         if final_goal_nav else "=tgt")

        dist_str = "N/A"
        if cur_pose is not None:
            tgt = (final_goal_nav if state in (NavState.FINAL_PLAN,
                                               NavState.FINAL_FOLLOW)
                   else patrol_goal_nav)
            if tgt is None:
                tgt = (final_target if state in (NavState.FINAL_PLAN,
                                                 NavState.FINAL_FOLLOW)
                       else patrol_target)
            if tgt is not None:
                dx = tgt[0] - cur_pose[0]
                dz = tgt[1] - cur_pose[1]
                dist_str = f"{(dx*dx+dz*dz)**0.5:.2f}m"

        vlm_pending = mock.peek_vlm_key()
        det_pending = mock.peek_det_key()

        lines = [
            (f"[SLAM]  mode={mode_str}  img={'Y' if has_img else 'N'}  "
             f"pose={'Y' if has_pose else 'N'}  mapY={nav_y_str}",
             (0, 255, 255) if mode_str == "RELOC" else (0, 255, 0)),
            f"[POSE]  {pose_str}",
            f"[NAV]   state={state.name}  step={step}  fps={fps:.1f}",
            f"[PATH]  {path_str}  blocked={blocked_str}",
            f"[TGT]   patrol={patrol_str}  final={final_str}",
            f"[NAV]   patrol_nav={patrol_nav_str}  final_nav={final_nav_str}  "
            f"dist={dist_str}",
            f"[CMD]   {last_cmd}",
            f"[MOCK]  vlm={'none' if not vlm_pending else vlm_pending}  "
            f"det={'none' if not det_pending else det_pending}",
        ]
        if halted:
            lines.append("[ESTOP] HALTED (space to resume)")

        if state == NavState.SCANNING:
            lines.append(
                f"[SCAN]  {np.degrees(scan_accumulated):.0f}/360°")

        # pending mock 区域框
        rects = []
        if img is not None and vlm_pending:
            hh, ww = img.shape[:2]
            bbox = key_to_bbox(vlm_pending, hh, ww)
            if bbox:
                rects.append({"bbox": bbox, "color": (0, 255, 0),
                              "label": f"VLM:{vlm_pending}"})
        if img is not None and det_pending:
            hh, ww = img.shape[:2]
            bbox = key_to_bbox(det_pending, hh, ww)
            if bbox:
                rects.append({"bbox": bbox, "color": (0, 0, 255),
                              "label": "DET:o"})

        _td0 = time.time()
        overlay_frame = draw_debug_overlay(img, {"lines": lines,
                                                   "rects": rects,
                                                   "robot_pose": cur_pose,
                                                   "using_odom": _use_odom,
                                                   "vlm_dets": vlm_latest[0],
                                                   "vlm_masks": vlm_latest[1]})
        mock.set_frame(overlay_frame)

        # ===== 顶视地图 =====
        obs_pts = obs_current
        _mem = getattr(slam, "memory", None)
        memory_gaussians, n_memory_walked, t_export_ms = [], 0, 0.0
        mem_shape, mem_kappa = "disk", 1.0

        if _mem is not None:
            _t0 = time.perf_counter()
            try:
                from mast3r_slam.memory.types import SID_WALKED
                memory_gaussians = [
                    (float(r["mu_W"][0]), float(r["mu_W"][2]),
                     float(r.get("radius_W", 0.4)))
                    for r in _mem.export_centers_W()
                    if int(r.get("sid", 0)) == SID_WALKED
                ]
            except Exception:
                memory_gaussians = []
            n_memory_walked = len(memory_gaussians)
            t_export_ms = (time.perf_counter() - _t0) * 1000.0
            fcfg = _mem.cfg.get("frontier_grid", {}) if hasattr(_mem, "cfg") else {}
            mem_shape = fcfg.get("shape", "disk")
            mem_kappa = float(fcfg.get("kappa", 1.0))

            frontier_viz = query_frontier_for_viz(
                _mem,
                pose_xz=((cur_pose[0], cur_pose[1])
                         if cur_pose is not None else None),
                nav_y=nav_y if nav_y is not None else 0.0,
                obstacle_points=obs_pts,
                robot_radius=ROBOT_RADIUS,
                yaw=(float(cur_pose[2]) if cur_pose is not None else None),
            )
        frontier_xz=frontier_viz["frontier_xz"] 


        map_frame = draw_map_view({
            "obstacle_points": obs_pts,
            "memory_gaussians": memory_gaussians,
            "memory_shape": mem_shape,
            "memory_kappa": mem_kappa,
            "memory_alpha": 0.30,
            "n_memory_walked": n_memory_walked,
            "frontier_xz": frontier_viz.get("frontier_xz"),
            "frontier_dist_m": frontier_viz.get("frontier_dist_m"),
            "frontier_ms": frontier_viz.get("frontier_ms"),
            # Grid debug overlay
            "grid_occ_inf": frontier_viz.get("grid_occ_inf"),
            "grid_walked": frontier_viz.get("grid_walked"),
            "grid_n": frontier_viz.get("grid_n"),
            "grid_origin_x": frontier_viz.get("grid_origin_x"),
            "grid_origin_z": frontier_viz.get("grid_origin_z"),
            "grid_res": frontier_viz.get("grid_res"),
            "robot_pose": cur_pose,
            "using_odom": _use_odom,
            "patrol_target": patrol_target,
            "patrol_nav": patrol_goal_nav,
            "final_target": final_target,
            "final_nav": final_goal_nav,
            "lookahead": lookahead_target,
            "path": path,
            "path_idx": path_idx,
            "d_tgt": d_tgt,
            "d_robot": d_robot,
            "state": state,
            "nav_y": nav_y,
            "n_map_points": n_map,
        })
        mock.set_map(map_frame)
        _nav_draw_ms = (time.time() - _td0) * 1000.0   # [NAVSTEP] 绘制耗时

        _nav_now = time.time()
        if _nav_now - _nav_step_t >= 2.0:
            _nav_cyc_ms = (time.time() - prev_t) * 1000.0
            _nav_work_ms = max(0.0, _nav_cyc_ms - _nav_grab_ms - _nav_draw_ms)
            _probe("navstep", f"[NAVSTEP] cyc={_nav_cyc_ms:.0f}ms  grab={_nav_grab_ms:.1f}ms  "
                  f"work={_nav_work_ms:.1f}ms  draw={_nav_draw_ms:.1f}ms  "
                  f"fps={fps:.1f}")
            _nav_step_t = _nav_now


        rate.sleep()

    # 清理
    if __sigint_timer is not None:
        __sigint_timer.cancel()
    motion_thread.stop()
    keyboard.stop()      # 恢复终端原始模式
    bot.stop()
    _nav_stop.set()     # 唤醒并退出点云后台刷新线程
    slam.stop()
    try:
        rospy.signal_shutdown("main loop exit")
    except Exception:
        pass
    print("[nav_2d] 导航结束")


def _cmd_str(linear, angular):
    """速度 → 可读动作描述。"""
    if abs(linear) < 0.01 and abs(angular) < 0.01:
        return "stop"
    if abs(angular) < 0.05:
        return f"forward({linear:.2f})"
    if abs(linear) < 0.05:
        d = "L" if angular > 0 else "R"
        return f"turn_{d}({abs(angular):.2f})"
    d = "L" if angular > 0 else "R"
    return f"arc(v={linear:.2f},{d}{abs(angular):.2f})"


if __name__ == "__main__":
    main()
