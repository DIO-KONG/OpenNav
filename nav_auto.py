#!/usr/bin/env python
"""nav_auto.py - 自动版状态机导航 + Web Debug (localhost:5001)。
状态机 WAITING→SCANNING→VLM_INQUIRY→PATROL/FINAL PLAN/FOLLOW→DONE。
坐标系: SLAM X=右 Y=下 Z=前; 导航平面 (X,Z), Y=高度。
(紧急时仍可网页/终端按 m 进手动, Space 急停。)
"""

import os
import re
import sys
import time
import signal
import threading
import rospy
import numpy as np

from nav_msgs.msg import Odometry

from mast3r_slam_wrapper import Mast3rSlamWrapper

# 模块已在仓库根目录: 单层 dirname（与 nav_mock / NavDebugger 一致）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.insert(0, os.path.join(
    _PROJECT_ROOT,
    "tracer_ros", "tracer_http_interface", "scripts"))
from tracer_http_interface.scripts.rw_api import TracerRobot

from nav_control import (NavState, mock, MotionThread, KeyboardThread,
                         MOTION_HZ, SCAN_ANGULAR, RELOC_STOP_TIME, RELOC_BACK_TIME,
                         MANUAL_HOLD_TIMEOUT, _angle_diff,
                         PointCloudCache, PC_HZ, PC_INTERVAL_DT)
import nav_path
from nav_path import (ROBOT_RADIUS, PLAN_BOX_PAD, PATROL_ARRIVE_EPS,
                      FRONTIER_ARRIVE_EPS,
                      FOLLOW_LIN_MAX, FOLLOW_YAW_EMA, FOLLOW_OUT_EMA, FOLLOW_DEADBAND,
                      FOLLOW_STRAIGHT_ALPHA, FOLLOW_SLEW,
                      crop_to_box, plan_path, follow_path_step, is_path_blocked,
                      GOAL_SHIFT_COOLDOWN,
                      RRT_OBSTACLE_INFLATE,
                      shift_goal_to_free,
                      )

# ============================================================
import nav_page
from nav_page import (YawSmoother, OutputSmoother,
                     draw_debug_overlay,                      draw_map_view)

# ============================================================
from nav_helpers import (
    _cmd_str,
    key_to_bbox,
    get_keyframe_pose,
    OdomHolder,
    _odom_nav_pose,
    extract_target_3d,
    filter_filament_obstacles,
    get_obstacle_points,
    query_frontier_for_viz,
    _goal_walked,
    auto_ask_direction,
    auto_trigger_patrol
)
from nav_vlm import VlmDetector, qwen_bbox_to_pixel, vlm_detect_with_presence
from nav_constants import (VLM_DETECT_TARGET_DEFAULT,
                         VLM_URL_DEFAULT,
                         VLM_H_PROMPT,
                         MOBILE_SAM_CHECKPOINT_PATH,
                         BOOT_SCAN_ENABLED,
                         OBJ_DET_PERIOD,
                         FINAL_ADJUST_TIME,
                         FINAL_ADJUST_MIN_FRAMES,
                         FINAL_ADJUST_DET_PERIOD,
                         FINAL_ADJUST_SMOOTH_ALPHA,
                         FINAL_ADJUST_REJECT_DIST,
                         FINAL_ADJUST_LOCK_DIST,
                         FINAL_ADJUST_FORCE_TIME,
                         POST_ESCAPE_HOLD,
                         POST_ESCAPE_CONVERGE_DIST,
                         POST_ESCAPE_CONVERGE_YAW,
                         AUTO_VLM_RETRY_PERIOD,
                         DEBUG_FEATURES)
from nav_debug import NavDebugger

_last_action_key = None   # 模块级: 动作日志去重

# ============================================================
#  调试打印统一开关
# ============================================================
DEBUG_PRINT = DEBUG_FEATURES.get("console_debug", False)

PLAN_FAIL_MAX = 10


def _dbg(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)


_ACTIVE_DEBUGGER = None



_last_state_print_t = 0.0
_printed_state = None

def _prn_state(s):
    global _last_state_print_t, _printed_state
    _now = time.time()
    if s != _printed_state or _now - _last_state_print_t >= 1.0:
        if DEBUG_PRINT:
            print(f"[nav_2d][STATE] {s.name}", flush=True)
        _printed_state = s
        _last_state_print_t = _now
    if _ACTIVE_DEBUGGER is not None:
        _ACTIVE_DEBUGGER.record_state(s)


# ============================================================
DEBUG_PROBES = {
    "navstep": DEBUG_FEATURES.get("performance_probe", False),
    "pcstat":  DEBUG_FEATURES.get("performance_probe", False),
    "reloc":   DEBUG_FEATURES.get("performance_probe", False),
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
        # motion hz 统计已关闭 (避免 log 刷屏); 如需排查抖动可重新打开下面的 print
        # print(f"[CMDSTAT] n={s['n']} v=[{s['vmin']:.3f},{s['vmax']:.3f}] "
        #       f"omega=[{s['wmin']:+.3f},{s['wmax']:+.3f}] "
        #       f"posejump_max={s['djmax']*100:.1f}cm "
        #       f"dyaw_max={np.degrees(s['dyawmax']):.1f}deg")
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
        if _ACTIVE_DEBUGGER is not None:
            _ACTIVE_DEBUGGER.record_action(tag, method, detail)
        _last_action_key = key


#  Debug Overlay
# ============================================================

def _nav_point_reached(cur_pose, pt, mem, nav_y):
    """判断某导航点是否已被当前位姿到达: 距离 <= PATROL_ARRIVE_EPS 或已落入 walked 记忆。

    用于 sub-opt goal 已到达判定: 当 RRT 退到的自由点(或真实目标本身)就在机器人当前位姿
    附近/已走过时, 朝它走已无意义, 应改走 frontier。"""
    if cur_pose is None or pt is None:
        return False
    _d = ((pt[0] - cur_pose[0]) ** 2 + (pt[1] - cur_pose[1]) ** 2) ** 0.5
    if _d <= PATROL_ARRIVE_EPS:
        return True
    return _goal_walked(mem, float(pt[0]), float(pt[1]), nav_y)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="nav_2d 状态机导航")
    ap.add_argument("--vlm-url", default=VLM_URL_DEFAULT,
                   help="Qwen VLM 服务 (OpenAI 兼容), 默认 keyboard_qwen 的默认地址")
    args = ap.parse_args()

    # 创建本次运行的 episode 文件夹并初始化结构化调试记录器。
    global _ACTIVE_DEBUGGER
    debugger = NavDebugger(project_root=_PROJECT_ROOT)
    _ACTIVE_DEBUGGER = debugger
    debugger.setup_episode(mode="nav_auto")

    rospy.init_node("nav_2d", anonymous=True)

    slam = Mast3rSlamWrapper(rgb_topic="/camera_f/color/image_raw")
    slam.start()

    odom_holder = OdomHolder()
    rospy.Subscriber("/odom", Odometry, odom_holder.cb, queue_size=1)
    print("[nav_2d] Subscribed to /odom (wheel odometry fallback)")

    bot = TracerRobot(base_url="http://localhost:8080")

    # --- VLM 检测器 (o 键, 默认启用; 默认地址同 keyboard_qwen) ---
    vlm_detector = VlmDetector(
        vlm_url=args.vlm_url,
        sam_ckpt=MOBILE_SAM_CHECKPOINT_PATH)
    print(f"[nav_2d] VLM 检测已启用: {args.vlm_url} model=(default)")

    nav_page.mock = mock
    if DEBUG_FEATURES.get("web_ui", True):
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
    patrol_source = None   # patrol_target 来源: "frontier" / "vlm_dir"
    goal_source = None     # 当前活跃目标来源: "vlm_det" / "frontier" / "vlm_dir"
    _last_goal_src = None      # 上一次写入保存日志的 goal_source (离散事件去重)
    _last_patrol_src = None    # 上一次写入保存日志的 patrol_source (离散事件去重)
    _last_goal_tgt_key = None  # 上一次写入保存日志的活跃目标坐标(取整), 用于变化检测
    final_adjust_t = 0.0      # FINAL_ADJUST 进入时刻 (计时锁定用)
    final_adjust_frames = 0   # FINAL_ADJUST 累积有效检测帧数 (EMA 平滑样本数)
    _final_from_reloc = False  # reloc 来源进入 FINAL_ADJUST: 禁用 EMA 平滑, 每帧直接用检测值
    _last_goal_shift_t = 0.0   # 目标再校验(方案A)节流时间戳
    d_tgt = None
    d_robot = None
    nav_y = None
    _patrol_plan_fail_cnt = 0
    _patrol_retry_t = 0.0          # nav_auto 自有节流 (PATROL 重规划等待日志), 不复用 nav_path._last_plan_fail_t
    _patrol_frontier_t = 0.0       # nav_auto 自有节流 (无 frontier 时重试查询), 不复用 nav_path._last_plan_fail_t
    _patrol_blocked_diverted = False   # 已因 RRT 连续失败改用 frontier, 防无限 frontier 循环
    _final_plan_fail_cnt = 0
    _final_adj_plan_fail_cnt = 0   # FINAL_ADJUST/FINAL_FOLLOW 的 path=None 重规划失败计数
    _final_adj_replan_pending = False  # FINAL_ADJUST 边走边规划: blocked 时标记一次重规划, 不停车
    _final_adj_blocked = False  # FINAL_ADJUST: 已检测到阻挡且截断到障碍点, 持久到"重规划出新路"才清除
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
    _reloc_prev_state = None   # [RELOC] 进入 reloc 前的导航状态, 恢复时据此保留高层目标
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
        try:
            debugger.close()
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
    _prev_escaping = False   # 上一帧 _escaping(用于检测脱困结束瞬间)
    _post_escape = False     # 脱困后 odom 保持窗口: 继续用 odom 直到 SLAM 追上/超时
    _post_escape_t = 0.0     # 保持窗口起始时间
    _escape_dir = None   # 位姿入侵脱困方向(远离最近障碍, 2D 单位向量)
    _enclosed_escape = False  # 当前是否处于"被包围 -> 直线倒退"脱困模式
    _yaw_sign = -1.0
    NAV_SMOOTH_POS = 0.30   # 位置 (x,z) 收敛系数
    NAV_SMOOTH_YAW = 0.30   # 偏航 (yaw) 收敛系数

    # ===== 自动决策状态 (nav_auto: 取代人工 j/k/l/f/o 按键) =====
    _auto_vlm_asked = False   # 本次 VLM_INQUIRY 是否已问过 VLM 选方向
    _auto_vlm_ask_t = 0.0     # 上次 VLM 方向问询时刻 (失败重试节流)
    _auto_det_t = 0.0         # 上次自动检测时刻 (OBJ_DET_PERIOD 节流)
    _presence_cache = [False]  # 周期自动检测: presence 一旦问询为 yes 即缓存, 后续不再重复问询 (list 容器, 供 vlm_detect_with_presence 传参控制)
    # 到达 VLM 方向点后"再次询问"的标记:
    #   False -> 本次检测到方向点已到达/走过, 不应直接去 frontier, 而是重新问 VLM;
    #            同时置 True, 表示"已为重问过一轮"。
    #   True  -> 这是重问后得到的新方向点, 若其 sub-opt goal 也已达/走过, 才是 frontier 点。
    _vlm_reask_pending = False

    def _set_patrol_goal(ft, src="frontier"):
        """设置 F 点目标并切到 PATROL_PLAN (F = frontier 巡逻目标)。
        ft: frontier (x,z)。重置巡逻导航状态。最终目标由 object detection 另行设置。
        src: 目标来源标记, "frontier" 或 "vlm_dir" 等, 用于 HUD/log 说明。
        """
        nonlocal patrol_target, patrol_goal_nav, patrol_goal_nav_target
        nonlocal path, path_idx, _patrol_plan_fail_cnt, _patrol_blocked_diverted
        nonlocal state, _auto_vlm_asked, patrol_source, goal_source
        _auto_vlm_asked = False
        patrol_target = ft
        patrol_source = src
        goal_source = src
        _patrol_plan_fail_cnt = 0
        _patrol_blocked_diverted = False   # 新目标 = 新"受阻"回合, 允许再次尝试 frontier 兜底
        patrol_goal_nav = None
        patrol_goal_nav_target = None
        path = None
        path_idx = 0
        state = NavState.PATROL_PLAN

    def _try_patrol_frontier():
        """连续路径失败 (RRT 无解 或 路径被 is_path_blocked 判阻挡) 满 PLAN_FAIL_MAX
        后, 尝试切换到最近 frontier。两种"路径失败"复用同一计数, 视为一回事。
        成功 -> _set_patrol_goal (内部清零计数并留 PATROL_PLAN 重规划);
        无 frontier -> 2s 节流打印, 留原状态持续重试 (不进 WAITING)。"""
        nonlocal _patrol_plan_fail_cnt, _patrol_frontier_t
        _ft, _nst = auto_trigger_patrol(slam, cur_pose, nav_y, obs_current)
        if _ft is not None:
            print(f"[nav_2d][WARN] PATROL 路径连续失败 (已 {_patrol_plan_fail_cnt} 次"
                  f" RRT失败/路径被阻挡) -> 改用 frontier 探索 (不进 WAITING)")
            _set_patrol_goal(_ft)   # 内部清零 _patrol_plan_fail_cnt, 留 PATROL_PLAN
        elif time.time() - _patrol_frontier_t >= 2.0:
            _patrol_frontier_t = time.time()
            print(f"[nav_2d][WARN] PATROL 路径连续失败 (已 {_patrol_plan_fail_cnt} 次)"
                  f" 且无可用 frontier -> 继续等待地图更新, 每 2s 重试 (不进 WAITING)")

    def _run_escape(cur_pose, d_robot, bo_tree, nav_y, halted, motion_thread):
        """位姿入侵脱困控制。优先用法向量方向安全后退; 若判定被障碍包围则
        直线倒退(沿 odom 朝向原路返回, 不信任会指错方向的法向量)。
        返回 (last_cmd, enclosed); 调用方负责置 _escaping=True。"""
        enclosed = nav_path.is_enclosed((cur_pose[0], cur_pose[1]), bo_tree, fixed_y=nav_y)
        _dir = (nav_path.find_escape_dir((cur_pose[0], cur_pose[1]), bo_tree, fixed_y=nav_y)
                if not enclosed else None)
        if time.time() - nav_path._last_intrusion_warn_t >= 2.0:
            if enclosed:
                _dbg(f"[nav_2d][WARN] 位姿侵入且被障碍包围 (距最近障碍 {d_robot:.2f}m), "
                     f"直线倒退原路返回脱困")
            elif _dir is not None:
                _dbg(f"[nav_2d][WARN] 位姿侵入障碍 (距最近障碍 {d_robot:.2f}m < "
                     f"robot_radius {ROBOT_RADIUS:.2f}m), 沿障碍法向后退脱困")
            else:
                _dbg(f"[nav_2d][WARN] 位姿侵入障碍 (距最近障碍 {d_robot:.2f}m < "
                     f"robot_radius {ROBOT_RADIUS:.2f}m), 无最近障碍点, 兜底直线后退")
            nav_path._last_intrusion_warn_t = time.time()
        if enclosed:
            # 被包围: 不信任法向量, 直接沿 odom 朝向直线倒退(原路返回)
            if not halted:
                motion_thread.set_velocity(-FOLLOW_LIN_MAX * 0.5, 0.0)
                motion_thread.start()
            return (f"escape->backtrack (enclosed, reverse, near obs {d_robot:.2f}m)", True)
        if _dir is not None:
            _el, _ea = nav_path.escape_velocity(cur_pose, _dir)
            if not halted:
                motion_thread.set_velocity(_el, _ea)
                motion_thread.start()
            return (f"escape->back (near obs {d_robot:.2f}m)", False)
        if not halted:
            motion_thread.set_velocity(-FOLLOW_LIN_MAX * 0.5, 0.0)
            motion_thread.start()
        return (f"escape blind (near obs {d_robot:.2f}m)", False)

    print("[nav_2d] 启动状态机导航, 等待 SLAM 就绪...")
    print("[nav_2d] 打开浏览器查看 debug "
          "(本机 http://localhost:5001, 其他设备用本机 IP:5001)")

    pc_cache = PointCloudCache(slam, PC_INTERVAL_DT, _nav_stop,
                               get_obstacle_points, _probe)
    pc_cache.start()
    print(f"[nav_2d] 点云后台刷新已启动 (PC_HZ={PC_HZ:.1f}, 控制 10Hz)")
    _mem = slam.memory

    while (not rospy.is_shutdown()) and \
            (not _nav_stop.is_set()) and \
            (state != NavState.DONE):

        # ===== 时序主键: 循环开头自增并绑定 debugger =====
        # 同轮内 VLM / pose / frame / telemetry / console 共享此 step。
        step += 1
        debugger.set_step(step)

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
        # 脱困结束瞬间(上一帧脱困、这一帧不再脱困) -> 进入 odom 保持窗口:
        # 继续用 odom 位姿(锚点冻结 + 实时 odom 增量), 等 SLAM 追上 odom 修正位姿
        # (距离+航向都够近) 或超时, 才切回 SLAM。否则 stale SLAM 会把 odom 在脱困
        # 期间挣脱出来的进度又拉回障碍方向 (snap-back)。
        if _prev_escaping and not _escaping:
            _post_escape = True
            _post_escape_t = time.time()
            # 脱困结束: 作废旧路径, 让 FOLLOW 状态下一帧用最新地图重新 RRT 规划。
            # 否则会沿原路径(穿回刚退出的障碍)再次入困 (死循环)。
            path = None
            print("[ODOM] 脱困结束 -> 进入 odom 保持窗口 (等 SLAM 追上); 作废旧路径待重规划")
        if _escaping:
            _post_escape = False   # 重新进入脱困, 不需要保持窗口
        _prev_escaping = _escaping
        if cur_pose is not None:
            odom_now = odom_holder.get()
            _use_odom = ((mode is not None and mode.name == "RELOC")
                         or _escaping or _post_escape)
            if _use_odom and odom_now is not None:
                _slam_pose_fresh = cur_pose   # 当前 SLAM 位姿(回退判定用, 回退前不覆盖)
                if _odom_anchor is None or _slam_anchor is None:
                    _slam_anchor = _slam_pose_fresh
                    _odom_anchor = odom_now
                _odom_pose = _odom_nav_pose(_slam_anchor, _odom_anchor, odom_now,
                                            yaw_sign=_yaw_sign)
                # 脱困后保持窗口: 检查 SLAM 是否已追上 odom 修正位姿
                if _post_escape:
                    _dx = _slam_pose_fresh[0] - _odom_pose[0]
                    _dz = _slam_pose_fresh[1] - _odom_pose[1]
                    _dpos = (_dx * _dx + _dz * _dz) ** 0.5
                    _dyaw = abs(_angle_diff(_slam_pose_fresh[2], _odom_pose[2]))
                    if _dpos < POST_ESCAPE_CONVERGE_DIST and _dyaw < POST_ESCAPE_CONVERGE_YAW:
                        _post_escape = False
                        print("[ODOM] SLAM 已追上 odom, 解除保持 -> 回退 SLAM")
                    elif time.time() - _post_escape_t > POST_ESCAPE_HOLD:
                        _post_escape = False
                        print("[ODOM] odom 保持超时, 强制回退 SLAM")
                cur_pose = _odom_pose
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

        manual_on = False  # 手动 WASD 驱动已禁用 (网页/终端), 仅保留空格急停

        # ===== RELOC 主动恢复 (仅旁路正常状态机, 不改动 SCANNING 本身) =====
        if (not manual_on) and (not _reloc_giveup) and \
                mode is not None and mode.name == "RELOC":
            # RELOC 自动恢复阶段机: stop(停车) -> back(直线后退离墙) -> spin(原地绕圈找 tracking)。
            # 视觉检测挑出逻辑不再放在阶段机之前 (否则一进 reloc 就被挑出, 永远转不了圈),
            # 而是移到下方 spin 分支内: reloc 先完整 stop->back->spin 转圈尝试恢复 tracking,
            # 只有 spin(绕圈) 期间视觉看到可信目标才挑出转 FINAL_ADJUST 追 —— 满足"reloc 该转转"。
            if _reloc_phase is None:
                _reloc_phase = "stop"
                _reloc_t = time.time()
                _reloc_start_t = time.time()
                _reloc_prev_state = state   # 记录进入 reloc 前的导航状态
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
                # ---- spin 阶段周期视觉检测: 看到可信目标则提前挑出 reloc 去追 ----
                # 仅在此阶段允许挑出: reloc 已先 stop->back->spin 完整转圈尝试恢复 tracking,
                # 绕圈期间若视觉看到目标 (用 odom 投影, 因 SLAM 仍在 RELOC 位姿不可信)
                # 才转 FINAL_ADJUST 追, 不压制 reloc 转圈找 tracking 的主路径。
                _det_now = time.time()
                if _det_now - _auto_det_t >= OBJ_DET_PERIOD:
                    _auto_det_t = _det_now
                    if img is not None and cur_pose is not None:
                        _det_tgt = VLM_DETECT_TARGET_DEFAULT  # 网页 o 键显式目标已忽略 (仅保留空格急停)
                        # RELOC 转圈期间: 先问 present 确认画面含目标, 再 detect+投影
                        # (presence_cache=None -> 始终重新问, 不缓存)
                        if _use_odom and cur_pose is not None:
                            _cam_pos_3d = (cur_pose[0],
                                           nav_y if nav_y is not None else 0.0,
                                           cur_pose[1])
                        else:
                            _cam_pos_3d = None
                        _present, _vdets, _vmasks, _bbox_px = vlm_detect_with_presence(
                            vlm_detector, img, _det_tgt,
                            require_presence=True, presence_cache=None)
                        if not _present:
                            # 画面无目标 (present=no): 继续转圈
                            vlm_latest = (None, None)
                        elif _vdets == "error":
                            print(f"\033[91m[RELOC][DET] VLM 错误: {_vmasks}\033[0m")
                            vlm_latest = (None, None)
                        elif _bbox_px is not None:
                            _t3d = extract_target_3d(
                                slam, _bbox_px, nav_y=nav_y,
                                allow_map_fallback=True,
                                camera_pos_3d=_cam_pos_3d, debugger=debugger)
                            if _t3d is not None:
                                # 提前挑出 reloc: 放弃自动恢复, 用 odom+视觉追目标
                                motion_thread.stop()
                                _reloc_phase = None
                                _reloc_giveup = True
                                final_target = (_t3d[0], _t3d[2])
                                _final_from_reloc = True
                                goal_source = "vlm_det"
                                state = NavState.FINAL_ADJUST
                                final_adjust_t = time.time()
                                final_adjust_frames = 1
                                path = None
                                path_idx = 0
                                vlm_latest = (_vdets, _vmasks)
                                print(f"[RELOC] 检测到 '{_det_tgt}' -> 挑出 RELOC "
                                      f"转 FINAL_ADJUST 追目标 "
                                      f"({final_target[0]:.2f},{final_target[1]:.2f})")
                                debugger.record_vlm({
                                    "stage": "reloc_detect_exit",
                                    "bbox_2d": list(_vdets[0].get("bbox_2d")),
                                    "target_xz": final_target,
                                })
                                continue
                            else:
                                print("[RELOC][DET] bbox->3D 失败, 继续 reloc")
                                vlm_latest = (None, None)
                        else:
                            # detect 有结果但无有效 bbox / 空 dets: 静默继续转圈
                            vlm_latest = (None, None)
                if _reloc_start_t > 0 and (time.time() - _reloc_start_t) > RELOC_MAX_TIME:
                    _reloc_giveup = True
                    motion_thread.stop()
                    _reloc_phase = None
                    print(f"[RELOC] 警告: 持续 {(time.time()-_reloc_start_t):.0f}s "
                          f"未恢复, 放弃自动 RELOC 接管, 释放回正常流程")

            # ===== RELOC 期间也要刷新 RGB + 地图, 否则画面冻结无法 debug =====
            # step 已在循环开头递增并 set_step, 本分支 continue 前仍写 pose/telemetry/frame。
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
            reloc_frame = None
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
                "goal_source": goal_source,
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
            debugger.record_pose(cur_pose, nav_y=nav_y, source=_nav_source)
            if debugger.due("telemetry", commit=False):
                if debugger.emit("telemetry", {
                    "state": "RELOC",
                    "mode": mode.name if mode is not None else None,
                    "reloc_phase": _reloc_phase,
                    "reloc_elapsed": _reloc_elapsed,
                    "path_index": path_idx,
                    "path_size": len(path) if path is not None else 0,
                    "patrol_target": patrol_target,
                    "patrol_source": patrol_source,
                    "final_target": final_target,
                    "goal_source": goal_source,
                    "distance_to_target": d_tgt,
                    "distance_to_obstacle": d_robot,
                    "last_command": last_cmd,
                    "fps": fps,
                }, feature="telemetry"):
                    debugger.mark_due("telemetry")
            debugger.save_debug_pair(reloc_frame, map_frame)
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
                # reloc 回到 TRACKING 后: 保留进入 reloc 前的导航状态与高层目标
                # (final_target / patrol 目标), 不再强制重新扫描一圈 (SCANNING 转完) ——
                # SLAM 重定位后坐标系已自愈, 无需再来一圈建图。
                _boot_scan_done = True
                _ps = _reloc_prev_state
                if _ps in (NavState.FINAL_PLAN, NavState.FINAL_FOLLOW, NavState.FINAL_ADJUST):
                    # FINAL 流程: 保留 final_target/final_goal_nav, 直接回 FINAL_FOLLOW 续走。
                    # 位姿可能因 reloc 跳变, 清空 path 让其重新规划到【当前 final_target】,
                    # 不再丢 final 去 patrol 找 nav goal。
                    state = NavState.FINAL_FOLLOW
                    path = None
                    path_idx = 0
                    print("[RELOC] 恢复 TRACKING: 保留 final 目标, 直接 FINAL_FOLLOW 续走 (不重扫描)")
                elif _ps in (NavState.PATROL_PLAN, NavState.PATROL_FOLLOW):
                    # PATROL 流程: 保留 patrol 目标, 直接 PATROL_PLAN 重规划续走
                    state = NavState.PATROL_PLAN
                    path = None
                    path_idx = 0
                    print("[RELOC] 恢复 TRACKING: 保留 patrol 目标, 直接 PATROL_PLAN 续走 (不重扫描)")
                elif _ps == NavState.SCANNING:
                    # 扫描中被 reloc: 不继续转完, 按已有目标直接续 (有目标->PLAN, 否则询问)
                    if final_target is not None:
                        state = NavState.FINAL_PLAN
                        path = None; path_idx = 0
                    elif patrol_target is not None:
                        state = NavState.PATROL_PLAN
                        path = None; path_idx = 0
                    else:
                        state = NavState.VLM_INQUIRY
                        _auto_vlm_asked = False
                        _vlm_reask_pending = False
                    print("[RELOC] 恢复 TRACKING: 扫描中 reloc, 跳过扫描续走")
                else:
                    # WAITING / VLM_INQUIRY / ESCAPE 等早期状态: 直接续回原状态 (不重扫描)
                    state = _ps if _ps is not None else NavState.WAITING
                    print(f"[RELOC] 恢复 TRACKING: 续回 {state.name} (不重扫描)")
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
                elif state in (NavState.FINAL_FOLLOW, NavState.FINAL_ADJUST):
                    state = NavState.FINAL_PLAN
                    _final_from_reloc = False

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

        # (网页 f 键 frontier 请求逻辑已移除: 防止误触打断导航状态机, 仅保留空格急停)

        # ===== AUTO: 周期目标检测 (o 键逻辑) =====
        _det_now = time.time()
        _det_period = (FINAL_ADJUST_DET_PERIOD
                       if state == NavState.FINAL_ADJUST else OBJ_DET_PERIOD)
        if _det_now - _auto_det_t >= _det_period:
            _auto_det_t = _det_now
            if (img is not None and cur_pose is not None and
                    ((final_target is None
                      and state not in (NavState.FINAL_PLAN, NavState.FINAL_FOLLOW))
                     or state == NavState.FINAL_ADJUST)):
                _det_tgt = VLM_DETECT_TARGET_DEFAULT      # 网页 o 键显式目标已忽略 (仅保留空格急停)
                _trigger = "auto"
                # presence(可选+缓存) -> detect -> extract -> coord (纯 VLM 采集已抽 vlm_detect_with_presence)
                if _use_odom and cur_pose is not None:
                    _cam_pos_3d = (cur_pose[0],
                                   nav_y if nav_y is not None else 0.0,
                                   cur_pose[1])
                else:
                    _cam_pos_3d = None
                # FINAL_ADJUST 下每帧重问 presence (不缓存); 其余状态用缓存避免重复问询
                _pres_cache = None if state == NavState.FINAL_ADJUST else _presence_cache
                _present, _vdets, _vmasks, _bbox_px = vlm_detect_with_presence(
                    vlm_detector, img, _det_tgt,
                    require_presence=True, presence_cache=_pres_cache)
                if not _present:
                    # 画面无目标 (presence=no / 空回答): 跳过检测
                    print(f"[nav_auto][DET] 问询: 画面无 '{_det_tgt}' "
                          f"-> 跳过检测")
                    vlm_latest = (None, None)
                elif _vdets == "error":
                    print(f"\033[91m[nav_auto][DET] VLM 错误: {_vmasks}\033[0m")
                    vlm_latest = (None, None)
                    debugger.record_vlm({
                        "stage": "detect_error",
                        "detail": str(_vmasks),
                        "trigger": _trigger,
                    })
                elif _bbox_px is not None:
                    _t3d = extract_target_3d(
                        slam, _bbox_px, nav_y=nav_y,
                        allow_map_fallback=(state == NavState.FINAL_ADJUST),
                        camera_pos_3d=_cam_pos_3d, debugger=debugger)
                    if _t3d is not None:
                        _vb = _vdets[0].get("bbox_2d")
                        if state == NavState.FINAL_ADJUST:
                            if _final_from_reloc:
                                # reloc 来源进入终调: 检测 100% 准 + odom 位姿短时不漂,
                                # 每帧直接用当前检测值覆盖, 不做 EMA 平滑、也不做离群拒收。
                                final_target = (_t3d[0], _t3d[2])
                                final_adjust_frames += 1
                                path = None
                            else:
                                # 终调阶段: EMA 平滑多次检测, 不覆盖首帧原始值
                                _dx = _t3d[0] - final_target[0]
                                _dz = _t3d[2] - final_target[1]
                                _jump = (_dx * _dx + _dz * _dz) ** 0.5
                                if _jump <= FINAL_ADJUST_REJECT_DIST:
                                    final_target = (
                                        FINAL_ADJUST_SMOOTH_ALPHA * _t3d[0]
                                        + (1.0 - FINAL_ADJUST_SMOOTH_ALPHA) * final_target[0],
                                        FINAL_ADJUST_SMOOTH_ALPHA * _t3d[2]
                                        + (1.0 - FINAL_ADJUST_SMOOTH_ALPHA) * final_target[1])
                                    final_adjust_frames += 1
                                    path = None
                                else:
                                    print(f"[nav_auto][DET] EMA 拒收离群测量 "
                                          f"(跳变 {_jump:.2f}m > "
                                          f"{FINAL_ADJUST_REJECT_DIST:.2f}m, "
                                          f"保持旧终点 ({final_target[0]:.2f},"
                                          f"{final_target[1]:.2f}))")
                        else:
                            final_target = (_t3d[0], _t3d[2])
                            goal_source = "vlm_det"
                        _final_plan_fail_cnt = 0
                        print(f"[nav_auto][DET] VLM 目标 "
                              f"({final_target[0]:.2f},{final_target[1]:.2f}) "
                              f"label={_vdets[0].get('label','?')} "
                              f"-> FINAL_ADJUST"
                              f"{' (EMA平滑)' if state == NavState.FINAL_ADJUST else ''}")
                        if (not manual_on) and state in (
                                NavState.SCANNING,
                                NavState.VLM_INQUIRY,
                                NavState.PATROL_PLAN,
                                NavState.PATROL_FOLLOW):
                            path = None
                            path_idx = 0
                            final_adjust_t = time.time()
                            final_adjust_frames = 1  # 首帧检测 = 样本 #1
                            state = NavState.FINAL_ADJUST
                        vlm_latest = (_vdets, _vmasks)
                        debugger.record_vlm({
                            "stage": "detect",
                            "bbox_2d": list(_vb),
                            "bbox_2d_px": _bbox_px,
                            "target_xz": final_target,
                            "manual": False,
                            "trigger": _trigger,
                        })
                    else:
                        # bbox->3D 投影失败
                        print("[nav_auto][DET] bbox->3D 失败")
                        vlm_latest = (None, None)
                        debugger.record_vlm({
                            "stage": "bbox_3d_fail",
                            "bbox_2d": list(_vdets[0].get("bbox_2d")),
                            "bbox_2d_px": _bbox_px,
                            "trigger": _trigger,
                        })
                else:
                    # detect 有结果但无有效 bbox / 空 dets: 未检测到目标
                    print(f"[nav_auto][DET] 未检测到目标 "
                          f"(target='{_det_tgt}') -> 继续 follow path")
                    vlm_latest = (None, None)
                    debugger.record_vlm({
                        "stage": "no_detection",
                        "detail": f"empty_dets target='{_det_tgt}'",
                        "trigger": _trigger,
                    })
        # (网页 h 键 VLM 问答逻辑已移除, 仅保留空格急停)

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
                        _auto_vlm_asked = False
                        _vlm_reask_pending = False
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
                        _auto_vlm_asked = False
                        _vlm_reask_pending = False
                        _dbg("[nav_2d] SCANNING 完成 -> VLM_INQUIRY "
                             "(无已有目标, 等待询问)")
                else:
                    last_cmd = (f"scan "
                                f"({np.degrees(scan_accumulated):.0f}"
                                f"/360°)")

        elif state == NavState.VLM_INQUIRY:
            _prn_state(state)
            last_cmd = "idle (vlm wait)"
            vlm_key = None   # 自动模式下由下方 auto_ask_direction 赋值, 先初始化防未绑定

            # ===== AUTO: 自动问 VLM 选方向 (取代人工 j/k/l) =====
            # 本次 VLM_INQUIRY 尚未问过 -> 问一次; 失败/无明确答案则节流重试。
            if not _auto_vlm_asked:
                if time.time() - _auto_vlm_ask_t >= AUTO_VLM_RETRY_PERIOD:
                    _auto_vlm_ask_t = time.time()
                    _auto_vlm_asked = True
                    _dir = auto_ask_direction(vlm_detector, img)
                    print(f"[nav_auto] VLM 方向问询 -> '{_dir}'")
                    debugger.record_vlm({
                        "stage": "direction",
                        "dir": _dir,
                        "answer": _dir,
                    })
                    if _dir == "f":
                        # 非 left/right/front -> F (frontier 巡逻目标)
                        _ft, _nst = auto_trigger_patrol(
                            slam, cur_pose, nav_y, obs_current)
                        if _ft is not None:
                            _set_patrol_goal(_ft)
                            _dbg(f"[nav_auto] VLM=F -> {_nst} (frontier patrol {_ft})")
                        else:
                            print("[nav_auto] VLM=F 但无可用 frontier, 留 VLM_INQUIRY 重试")
                            _auto_vlm_asked = False   # 留在 VLM_INQUIRY, 冷却后重问
                    elif _dir in ("j", "k", "l"):
                        vlm_key = _dir   # 2D 左/中/右, 继续下方检测逻辑
                        print(f"[nav_auto] VLM 方向 '{vlm_key}' -> 检测目标")
                    else:
                        # 'none': VLM 出错, 允许下次重试 (不置 done)
                        _auto_vlm_asked = False
                        last_cmd = "idle (vlm ask failed, retry)"
                else:
                    last_cmd = "idle (vlm ask cooldown)"

            if img is None or cur_pose is None:
                pass
            elif vlm_key is not None:
                print(f"[nav_2d] VLM_INQUIRY: 方向键 '{vlm_key}', 开始检测目标...")
                # 直接按方向取图像三分之一区域 (同 key_to_bbox), 不再等人工按键
                bbox = key_to_bbox(vlm_key, img.shape[0], img.shape[1])
                if bbox is None:
                    last_cmd = f"idle (vlm: no target for '{vlm_key}')"
                else:
                    t3d = extract_target_3d(slam, bbox, nav_y=nav_y, debugger=debugger)
                    if t3d is not None:
                        _tx, _tz = float(t3d[0]), float(t3d[2])
                        patrol_target = (_tx, _tz)
                        patrol_source = "vlm_dir"
                        goal_source = "vlm_dir"
                        _patrol_plan_fail_cnt = 0  # 新目标, 清零失败计数
                        _patrol_blocked_diverted = False   # 新目标 = 新受阻回合
                        if (patrol_goal_nav is not None and
                                patrol_target != patrol_goal_nav_target):
                            _dbg(f"[nav_2d] patrol 目标已更新, "
                                 f"旧 sub-opt-goal {patrol_goal_nav} 失效 "
                                 f"-> 重新计算")
                            patrol_goal_nav = None
                            patrol_goal_nav_target = None
                        _nav = (patrol_goal_nav if patrol_goal_nav is not None
                                else patrol_target)
                        d_tgt = ((_nav[0] - cur_pose[0]) ** 2 +
                                 (_nav[1] - cur_pose[1]) ** 2) ** 0.5
                        _walked = _goal_walked(
                            slam.memory, _tx, _tz, nav_y)
                        if _walked or d_tgt < PATROL_ARRIVE_EPS:
                            # ===== 方向点已到达/已走过: 不直接去 frontier, 而是再次询问 VLM =====
                            if _vlm_reask_pending:
                                # 这是"重问"后得到的新方向, 其 sub-opt goal 也已达/走过
                                # -> 才是真正的 frontier 点
                                print(f"[nav_auto] 重问后方向点仍已达/走过 "
                                      f"({d_tgt:.2f}m, walked={_walked}) "
                                      f"-> F(patrol)")
                                _ft, _nst = auto_trigger_patrol(
                                    slam, cur_pose, nav_y, obs_current)
                                if _ft is not None:
                                    _set_patrol_goal(_ft)
                                else:
                                    print("[nav_auto] 重问后仍无 frontier, "
                                          f"留 VLM_INQUIRY 重试")
                                    _auto_vlm_asked = False
                                _vlm_reask_pending = False
                            else:
                                # 第一次到达/走过 -> 重新询问 VLM 取下一个方向
                                print(f"[nav_auto] 已到达/走过 VLM 方向点 "
                                      f"({d_tgt:.2f}m, walked={_walked}), "
                                      f"再次询问 VLM")
                                _auto_vlm_asked = False  # 触发下一轮 VLM 问询
                                _vlm_reask_pending = True
                                vlm_key = None          # 清掉, 让上方重新问 VLM
                                # 留在 VLM_INQUIRY, 下一帧重问
                        else:
                            # 真正的远方向点 -> 去巡逻
                            _vlm_reask_pending = False
                            patrol_goal_nav = None   # 已远离, 重置导航终点
                            patrol_goal_nav_target = None
                            _dbg(f"[nav_2d] VLM 目标 "
                                 f"3D=({t3d[0]:.2f},{t3d[1]:.2f},"
                                 f"{t3d[2]:.2f}) -> 2D=({patrol_target[0]:.2f},"
                                 f"{patrol_target[1]:.2f})")
                            _dbg("[nav_2d] VLM_INQUIRY -> PATROL_PLAN")
                            state = NavState.PATROL_PLAN
                    else:
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
                # 例: VLM 方向目标被障碍挡住, RRT 退到最近自由点(sub-opt), 但机器人
                # 当前位姿已在该自由点(或已走过) -> 朝它走已无意义, 改用 frontier 探索。
                _divert_to_frontier = False
                _mem = slam.memory
                if (path is not None and len(path) > 0 and
                        patrol_goal_nav is not None and
                        _nav_point_reached(cur_pose, patrol_goal_nav, _mem, nav_y) and
                        not _nav_point_reached(cur_pose, patrol_target, _mem, nav_y)):
                    _ft, _nst = auto_trigger_patrol(
                        slam, cur_pose, nav_y, obs_current)
                    if _ft is not None:
                        print(f"[nav_auto] sub-opt-goal {patrol_goal_nav} 已到达 "
                              f"但真实目标 {patrol_target} 未达 -> 改用 frontier {_ft}")
                        _set_patrol_goal(_ft)
                        path = None  # 本次不进入 FOLLOW, 下一循环朝 frontier 重规划
                        _divert_to_frontier = True
                    # 无可用 frontier: 降级为正常进入 FOLLOW(到达后回 VLM_INQUIRY)
                if _divert_to_frontier:
                    # 仍留在 PATROL_PLAN, 下一循环朝 frontier 重新规划
                    pass
                elif path is not None and len(path) > 0:
                    # 注意: 成功规划不再清零 _patrol_plan_fail_cnt —— 清零改到
                    # PATROL_FOLLOW 正常跟随帧 (见下), 避免擦边路径 (RRT 成功但
                    # is_path_blocked 反复判阻挡) 被每轮清零, 永远到不了 frontier。
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
                    # RRT 无解: 累计一次"路径失败" (与 FOLLOW 中 is_path_blocked 合并同一计数)
                    _patrol_plan_fail_cnt += 1
                    if _patrol_plan_fail_cnt >= PLAN_FAIL_MAX:
                        _try_patrol_frontier()
                    elif time.time() - _patrol_retry_t >= 2.0:
                        _patrol_retry_t = time.time()
                        _dbg(f"[nav_2d] 路径规划失败 (第 {_patrol_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试 (最新地图)")

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
                obs_global = obs_current
                from nav_rrt import build_obstacle_tree, tree_nearest_dist
                bo_tree = build_obstacle_tree(
                    crop_to_box(obs_global, (cur_pose[0], cur_pose[1]),
                                patrol_target, PLAN_BOX_PAD), nav_y)
                d_robot = tree_nearest_dist(bo_tree, nav_y,
                                            cur_pose[0], cur_pose[1])
                # ---- 目标再校验 (方案A): 当前地图下目标/终点已嵌入/紧贴障碍 -> 平移到可站立自由视点 ----
                _now_gs = time.time()
                if _now_gs - _last_goal_shift_t >= GOAL_SHIFT_COOLDOWN:
                    _nav_ep = (patrol_goal_nav if patrol_goal_nav is not None
                               else patrol_target)
                    _dt = (tree_nearest_dist(bo_tree, nav_y,
                            float(patrol_target[0]), float(patrol_target[1]))
                           if patrol_target is not None else float('inf'))
                    _de = (tree_nearest_dist(bo_tree, nav_y,
                             float(_nav_ep[0]), float(_nav_ep[1]))
                            if _nav_ep is not None else float('inf'))
                    if patrol_target is not None and _dt < RRT_OBSTACLE_INFLATE:
                        # 语义目标本身嵌入/紧贴障碍(距障<膨胀半径) -> 平移到可站立自由视点 (goal 更新)
                        # free_margin 与 is_path_blocked 阻挡门槛一致, 避免退到 (GOAL_STAND_DIST,0.55) 区间
                        # 仍被路径阻挡判据误杀 -> 重规划死锁。
                        _shifted = shift_goal_to_free(
                            patrol_target, (cur_pose[0], cur_pose[1]),
                            obs_current, fixed_y=nav_y,
                            free_margin=RRT_OBSTACLE_INFLATE)
                        _last_goal_shift_t = _now_gs
                        if _shifted is not None:
                            _dbg(f"[nav_2d][GOAL-UPDATE] patrol_target 已嵌入障碍 "
                                 f"(距障 {_dt:.2f}m), 平移到可站立自由视点 {_shifted}")
                            patrol_target = _shifted
                            patrol_goal_nav = None
                            patrol_goal_nav_target = None
                            state = NavState.PATROL_PLAN
                            continue
                        # shift 失败: 保留原目标, 靠下方 endpoint 重规划兜底
                    if _de < RRT_OBSTACLE_INFLATE:
                        # 当前导航终点(可能地图更新后嵌死)不可站立 -> 转 PATROL_PLAN 用最新地图重算
                        _last_goal_shift_t = _now_gs
                        _dbg(f"[nav_2d][GOAL-UPDATE] 巡逻终点嵌死 (距障 {_de:.2f}m), "
                             f"转 PATROL_PLAN 用最新地图重算可站立终点")
                        state = NavState.PATROL_PLAN
                        continue
                # ---- frontier 目标失效: 该点已被走过(memory walked margin) -> 放弃旧目标, 重启 VLM_INQUIRY ----
                if (patrol_source == "frontier" and patrol_target is not None
                        and _goal_walked(slam.memory, float(patrol_target[0]),
                                         float(patrol_target[1]), nav_y)):
                    _dbg(f"[nav_2d][FRONTIER-STALE] 巡逻 frontier 点 {patrol_target} "
                         f"已被走过(不在 memory margin 条件), 放弃旧目标 -> VLM_INQUIRY 重问")
                    patrol_target = None
                    patrol_source = None
                    patrol_goal_nav = None
                    patrol_goal_nav_target = None
                    motion_thread.stop()
                    state = NavState.VLM_INQUIRY
                    _auto_vlm_asked = False
                    _vlm_reask_pending = False
                    continue

                # 到达判达: 边界 goal 会让 d_robot<R(贴墙), 故"侵入时也算到达"(脱困中若已到目标即停)
                _arrive_eps = (FRONTIER_ARRIVE_EPS if patrol_source == "frontier"
                               else PATROL_ARRIVE_EPS)
                if ((d_robot >= ROBOT_RADIUS and
                     (d_tgt <= _arrive_eps or _de <= _arrive_eps
                      or path_idx >= len(path)))
                    or (d_robot < ROBOT_RADIUS and
                        (d_tgt <= _arrive_eps or _de <= _arrive_eps))):
                    _mem = slam.memory
                    _sub_reached = _nav_point_reached(
                        cur_pose, patrol_goal_nav, _mem, nav_y)
                    _tgt_reached = _nav_point_reached(
                        cur_pose, patrol_target, _mem, nav_y)
                    if _sub_reached and not _tgt_reached:
                        # 到达 sub-opt fallback 但真实目标仍受阻 -> 改用 frontier
                        _ft, _nst = auto_trigger_patrol(
                            slam, cur_pose, nav_y, obs_current)
                        if _ft is not None:
                            print(f"[nav_auto] FOLLOW 到达 sub-opt-goal "
                                  f"{patrol_goal_nav} 但真实目标未达 -> "
                                  f"改用 frontier {_ft}")
                            _set_patrol_goal(_ft)
                            state = NavState.PATROL_PLAN
                        else:
                            motion_thread.stop()
                            _dbg(f"[nav_2d] PATROL_FOLLOW -> VLM_INQUIRY "
                                  f"(sub-opt 已到达但无可用 frontier, "
                                  f"距 nav={d_tgt:.2f}m, 回退重问)")
                            state = NavState.VLM_INQUIRY
                            _auto_vlm_asked = False
                            _vlm_reask_pending = False
                    else:
                        motion_thread.stop()
                        _dbg(f"[nav_2d] PATROL_FOLLOW -> VLM_INQUIRY "
                              f"(到达巡逻点, 距 nav={d_tgt:.2f}m / "
                              f"距路径终点={_de:.2f}m "
                              f"<= {PATROL_ARRIVE_EPS:.2f}m, 早停防撞)")
                        state = NavState.VLM_INQUIRY
                        _auto_vlm_asked = False
                        _vlm_reask_pending = False
                else:
                    # 已自由 -> 复位脱困方向 (每帧重算, 无需缓存旧点)
                    if d_robot >= ROBOT_RADIUS:
                        _escape_dir = None
                    now_t = time.time()
                    if now_t - _vlog_t >= 0.5:
                        _dbg(f"[nav_2d]   [DEBUG] 未到达 "
                             f"nav={d_tgt:.2f}m path_end={_de:.2f}m "
                             f"eps={_arrive_eps:.2f}m "
                             f"path_idx={path_idx}/{len(path)} "
                             f"cur_pose=({cur_pose[0]:.2f},{cur_pose[1]:.2f}) "
                             f"sub_opt={patrol_goal_nav} "
                             f"d_robot={d_robot:.2f}m")
                        _vlog_t = now_t

                    # ---- 位姿入侵脱困: 法向量方向后退; 被包围则直线倒退(原路返回) ----
                    if d_robot < ROBOT_RADIUS:
                        last_cmd, _enclosed_now = _run_escape(
                            cur_pose, d_robot, bo_tree, nav_y, halted, motion_thread)
                        _enclosed_escape = _enclosed_now
                        _escaping = True
                    elif is_path_blocked(path, path_idx, obs_global,
                                         fixed_y=nav_y, tree=bo_tree):
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            # 路径被阻挡也算一次"路径失败", 与 RRT 无解合并同一计数
                            _patrol_plan_fail_cnt += 1
                            nav_path._last_replan_t = now_t
                            if _patrol_plan_fail_cnt >= PLAN_FAIL_MAX:
                                _try_patrol_frontier()   # 满 N 次 -> frontier (不转 PLAN)
                            else:
                                _dbg("[nav_2d] 路径被阻挡 -> PATROL_PLAN (重规划)")
                                state = NavState.PATROL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                    else:
                        # 路径通畅、正常跟随: 清零"路径失败"计数 (只有连续卡住才会累积到 N)
                        _patrol_plan_fail_cnt = 0
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
                            if d_tgt <= _arrive_eps:
                                _dbg("[nav_2d] PATROL_FOLLOW -> VLM_INQUIRY "
                                      "(到达导航终点, 完成寻路)")
                                state = NavState.VLM_INQUIRY
                                _auto_vlm_asked = False
                                _vlm_reask_pending = False
                            else:
                                _dbg(f"[nav_2d] 路径已走完但未到达 nav 终点 "
                                      f"(距 {d_tgt:.2f}m) -> PATROL_PLAN "
                                      f"(重规划)")
                                state = NavState.PATROL_PLAN
            else:
                # 脱困后旧路径已作废 (path=None) -> 用最新地图重新 RRT 规划,
                # 避免沿原路径再次进入同一障碍 (再次入困)。
                state = NavState.PATROL_PLAN
                continue

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
                        # FINAL 终点不可达: 复用 shift_goal_to_free 把终点重定位到
                        # 最近可达自由点 (不进 frontier, 不丢目标 -> 留 FINAL_PLAN 重规划)
                        _reloc = nav_path.relocate_goal_reachable(
                            final_target, (cur_pose[0], cur_pose[1]),
                            obs_current, fixed_y=nav_y,
                            free_margin=RRT_OBSTACLE_INFLATE, max_radius=6.0)
                        _final_plan_fail_cnt = 0
                        if _reloc is not None and _reloc != tuple(final_target):
                            print(f"[nav_2d][GOAL-RELOC] FINAL 终点不可达, 重定位到最近可达视点 "
                                  f"{_reloc} (距原目标 "
                                  f"{((_reloc[0]-final_target[0])**2 + (_reloc[1]-final_target[1])**2)**0.5:.2f}m), "
                                  f"继续 FINAL_PLAN 重规划")
                            final_target = _reloc
                            final_goal_nav = None
                            path = None
                            path_idx = 0
                            # 留 FINAL_PLAN, 下一帧用新终点重规划
                        else:
                            print(f"[nav_2d][WARN] FINAL 终点不可达且就近重定位失败 "
                                  f"(目标区域似被封死) -> 保留原终点持续重试 "
                                  f"(不进 frontier/WAITING)")
                            # 不放弃目标, 不进 frontier; 留 FINAL_PLAN 周期重试, 等待地图补全
                    elif time.time() - nav_path._last_plan_fail_t >= 2.0:
                        _dbg(f"[nav_2d] 最终路径规划失败 (第 {_final_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试 (最新地图)")
                        nav_path._last_plan_fail_t = time.time()

        elif state in (NavState.FINAL_FOLLOW, NavState.FINAL_ADJUST):
            _prn_state(state)
            _is_adjust = (state == NavState.FINAL_ADJUST)
            if cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_2d] pose 丢失, 停车等待")
            elif path is None:
                # FINAL_ADJUST 首帧 (检测命中时 path 被清空) / FINAL_FOLLOW 路径被清空:
                # 立即规划一次到当前 final_target, 之后走下面的跟随逻辑。
                if final_target is not None:
                    obs = obs_current
                    path, final_goal_nav = plan_path(
                        (cur_pose[0], cur_pose[1]),
                        final_target, obs, fixed_y=nav_y,
                        plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                       if _plan_pose is not None else None))
                    if path is not None and len(path) > 0:
                        _final_adj_plan_fail_cnt = 0
                        path_idx = 0
                        yaw_smoother.reset()
                        out_smoother.reset()
                        # 新规划出的路径若已不阻挡(FINAL_ADJUST 边走边规划期间), 清除持久阻挡标志
                        if _final_adj_blocked and not is_path_blocked(
                                path, 0, obs, fixed_y=nav_y):
                            _final_adj_blocked = False
                        _dbg(f"[nav_2d] {('FINAL_ADJUST' if _is_adjust else 'FINAL_FOLLOW')} "
                             f"首规划 ({len(path)} wp, nav=({final_goal_nav[0]:.2f},"
                             f"{final_goal_nav[1]:.2f}))")
                    else:
                        # RRT 无解: 累计失败, 满阈值转 FINAL_PLAN 走'重定位到最近可达'逻辑
                        _final_adj_plan_fail_cnt += 1
                        if _final_adj_plan_fail_cnt >= PLAN_FAIL_MAX:
                            _final_adj_plan_fail_cnt = 0
                            print(f"[nav_2d][WARN] FINAL_ADJUST/FOLLOW 原地重规划连续 "
                                  f"{PLAN_FAIL_MAX} 次失败, 终点不可达 -> 转 FINAL_PLAN 重定位")
                            state = NavState.FINAL_PLAN
                last_cmd = "plan pending (adjust/follow)"
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
                obs_global = obs_current
                from nav_rrt import build_obstacle_tree, tree_nearest_dist
                bo_tree = build_obstacle_tree(
                    crop_to_box(obs_global, (cur_pose[0], cur_pose[1]),
                                final_target, PLAN_BOX_PAD), nav_y)
                d_robot = tree_nearest_dist(bo_tree, nav_y,
                                            cur_pose[0], cur_pose[1])
                # ---- 目标再校验 (方案A): 当前地图下目标/终点已嵌入/紧贴障碍 -> 平移到可站立自由视点 ----
                # 防止"接近目标时露出障碍 -> d_robot<ROBOT_RADIUS 触发脱困 -> 永远到不了"死循环。
                _now_gs = time.time()
                if _now_gs - _last_goal_shift_t >= GOAL_SHIFT_COOLDOWN:
                    _nav_ep = (final_goal_nav if final_goal_nav is not None
                               else final_target)
                    _dt = (tree_nearest_dist(bo_tree, nav_y,
                            float(final_target[0]), float(final_target[1]))
                           if final_target is not None else float('inf'))
                    _de = (tree_nearest_dist(bo_tree, nav_y,
                             float(_nav_ep[0]), float(_nav_ep[1]))
                            if _nav_ep is not None else float('inf'))
                    if final_target is not None and _dt < RRT_OBSTACLE_INFLATE:
                        # 语义目标本身嵌入/紧贴障碍(距障<膨胀半径) -> 平移到可站立自由视点 (goal 更新)
                        # free_margin 与 is_path_blocked 阻挡门槛一致: FINAL 目标常投影到门框/墙角(距障 0.35~0.55),
                        # 若只退到 GOAL_STAND_DIST(0.35) 仍落 (0.35,0.55) 区间 -> actual_goal 附近航点被
                        # is_path_blocked(0.55) 误判阻挡 -> 重规划死锁。这里把 final_target 本身也退到自由点,
                        # 使终调投影/到达判定与导航终点一致。
                        _shifted = shift_goal_to_free(
                            final_target, (cur_pose[0], cur_pose[1]),
                            obs_current, fixed_y=nav_y,
                            free_margin=RRT_OBSTACLE_INFLATE)
                        _last_goal_shift_t = _now_gs
                        if _shifted is not None:
                            _dbg(f"[nav_2d][GOAL-UPDATE] final_target 已嵌入障碍 "
                                 f"(距障 {_dt:.2f}m), 平移到可站立自由视点 "
                                 f"{_shifted} (距原目标 "
                                 f"{((_shifted[0]-final_target[0])**2 + (_shifted[1]-final_target[1])**2)**0.5:.2f}m)")
                            final_target = _shifted
                            final_goal_nav = None
                            path = None
                            path_idx = 0
                            last_cmd = "goal shifted (embedded)"
                            continue
                        # shift 失败: 保留原目标, 靠下方 endpoint 重规划兜底
                    if _de < RRT_OBSTACLE_INFLATE and path is not None:
                        # 当前导航终点(可能地图更新后嵌死)不可站立 -> 强制重规划用最新地图重算可站立终点
                        _last_goal_shift_t = _now_gs
                        _dbg(f"[nav_2d][GOAL-UPDATE] 导航终点嵌死 (距障 {_de:.2f}m), "
                             f"强制重规划用最新地图")
                        path = None
                        path_idx = 0
                        last_cmd = "goal endpoint replan (embedded)"
                        continue
                if ((d_robot >= ROBOT_RADIUS and
                     (d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS))
                    or (d_robot < ROBOT_RADIUS and
                        (d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS))):
                    motion_thread.stop()
                    if _is_adjust:
                        # 终调阶段已到当前平滑终点: 原地保持, 继续微调检测, 不提前 DONE
                        last_cmd = "hold (adjust, reached smoothed target)"
                        _dbg(f"[nav_2d] FINAL_ADJUST 已到当前平滑终点 "
                              f"(距 nav={d_tgt:.2f}m), 原地保持继续微调")
                    else:
                        _dbg(f"[nav_2d] FINAL_FOLLOW -> DONE "
                              f"(到达检测目标, 距 nav={d_tgt:.2f}m / "
                              f"距路径终点={_de:.2f}m "
                              f"<= {PATROL_ARRIVE_EPS:.2f}m, 早停防撞)")
                        state = NavState.DONE
                else:
                    # 已自由 -> 复位脱困方向 (每帧重算, 无需缓存旧点)
                    if d_robot >= ROBOT_RADIUS:
                        _escape_dir = None
                    # ---- 位姿入侵脱困: 法向量方向后退; 被包围则直线倒退(原路返回) ----
                    _blocked = is_path_blocked(path, path_idx, obs_global,
                                               fixed_y=nav_y, tree=bo_tree)
                    if d_robot < ROBOT_RADIUS:
                        last_cmd, _enclosed_now = _run_escape(
                            cur_pose, d_robot, bo_tree, nav_y, halted, motion_thread)
                        _enclosed_escape = _enclosed_now
                        _escaping = True
                    elif _blocked and not _is_adjust:
                        # FINAL_FOLLOW(非终调): 阻挡 -> 急停 + 转 FINAL_PLAN 重规划
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _dbg("[nav_2d] 最终路径被阻挡 -> FINAL_PLAN (重规划)")
                            nav_path._last_replan_t = now_t
                            state = NavState.FINAL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                    else:
                        # 未阻挡, 或 FINAL_ADJUST 且阻挡: 边走边规划(不停车)
                        if _is_adjust and (_blocked or _final_adj_blocked):
                            # FINAL_ADJUST: 命中 blocked -> 截断 path 到障碍点 blk, 只走到障碍点为止;
                            # 持久标志 _final_adj_blocked 保证: 截断后 is_path_blocked 因末点被跳过返
                            # False 时, 仍每 1s 持续触发重规划, 直到重规划出新路(在 1470 分支清除标志)。
                            if _blocked:
                                blk, seg_start = nav_path.first_blocked_point(
                                    path, path_idx, obs_global,
                                    fixed_y=nav_y, tree=bo_tree)
                                if blk is not None:
                                    _final_adj_blocked = True
                                    # 截断: 保留 seg_start 之前(含)航点 + 障碍点本身作末航点,
                                    # 使 follow_path_step 在 blk 停下(不穿过障碍, 不开到路尾)。
                                    path = path[:seg_start + 1] + [blk]
                                    _dbg(f"[nav_2d] FINAL_ADJUST 路径被阻挡 -> "
                                         f"边走边规划(截断到障碍点 {blk}, 不停车)")
                            # 每 1s 节流触发一次重规划(命中或持久标志期间均持续)
                            now_t = time.time()
                            if now_t - nav_path._last_replan_t >= 1.0:
                                _final_adj_replan_pending = True
                                nav_path._last_replan_t = now_t
                            if _final_adj_replan_pending:
                                # 消费重规划标志: 清空 path 让下一循环重连新路
                                # (1470 分支重规划不停车, 机器人保持上一帧速度继续走)
                                _final_adj_replan_pending = False
                                path = None
                                path_idx = 0
                                continue

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
                        if _is_adjust:
                            # 终调阶段路径走完但未到目标: 由周期重规划重新连线, 不切状态
                            last_cmd = "adjust: path done, replan pending"
                            _dbg(f"[nav_2d] FINAL_ADJUST 路径走完未达目标 "
                                  f"(距 {d_tgt:.2f}m), 等周期重规划")
                        elif d_tgt <= PATROL_ARRIVE_EPS:
                            _dbg("[nav_2d] FINAL_FOLLOW -> DONE "
                                  "(到达导航终点, 完成寻路)")
                            state = NavState.DONE
                        else:
                            _dbg(f"[nav_2d] 路径已走完但未到达 nav 终点 "
                                  f"(距 {d_tgt:.2f}m) -> FINAL_PLAN "
                                  f"(重规划)")
                            state = NavState.FINAL_PLAN

                    # FINAL_ADJUST 锁定逻辑: 目标远则持续终调(每帧重投影), 直到足够近才固定 final goal。
                    #  - 近距锁定: 距目标 <= FINAL_ADJUST_LOCK_DIST 且累积帧数达标 -> 进 FINAL_FOLLOW 固定终点。
                    #  - 超时兜底: 进入终调后很久(FINAL_ADJUST_FORCE_TIME)仍远 -> 强制锁定最后可信终点, 防永久终调。
                    if _is_adjust:
                        _adj_elapsed = time.time() - final_adjust_t
                        _reached_min = final_adjust_frames >= FINAL_ADJUST_MIN_FRAMES
                        _close = d_tgt <= FINAL_ADJUST_LOCK_DIST
                        _force_lock = (_adj_elapsed >= FINAL_ADJUST_FORCE_TIME)
                        if (_adj_elapsed >= FINAL_ADJUST_TIME
                                and _reached_min and _close) or _force_lock:
                            if _force_lock and not _close:
                                print(f"\033[93m[nav_2d] FINAL_ADJUST 超时强制锁定 "
                                      f"(目标仍远 {d_tgt:.2f}m, 帧数 "
                                      f"{final_adjust_frames}, 终点 "
                                      f"({final_target[0]:.2f},{final_target[1]:.2f}))\033[0m")
                            else:
                                _dbg(f"[nav_2d] FINAL_ADJUST -> FINAL_FOLLOW "
                                      f"(平滑 {final_adjust_frames} 帧 / "
                                      f"{_adj_elapsed:.1f}s, 目标距 {d_tgt:.2f}m "
                                      f"<= {FINAL_ADJUST_LOCK_DIST:.2f}m, 锁定终点 "
                                      f"({final_target[0]:.2f},{final_target[1]:.2f}))")
                            state = NavState.FINAL_FOLLOW
                            _final_from_reloc = False

        _moving_states = (NavState.SCANNING,
                          NavState.PATROL_FOLLOW,
                          NavState.FINAL_FOLLOW,
                          NavState.FINAL_ADJUST)
        if (not manual_on) and \
                ((state not in _moving_states or halted)
                 and motion_thread.is_running()):
            motion_thread.stop()

        # (键盘 WASD 手动覆盖逻辑已移除: 仅保留空格急停, 手动模式恒为 False)

        if halted:
            last_cmd = "ESTOP (halt)"

        # ===== FPS =====
        now = time.time()
        dt = now - prev_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_t = now
        # step 已在循环开头自增 + set_step, 此处不再 step+=1

        # ===== Debug overlay / 结构化遥测 (同 step) =====
        debugger.record_pose(cur_pose, nav_y=nav_y, source=_nav_source)
        debugger.record_state(state)
        if debugger.due("telemetry", commit=False):
            if debugger.emit("telemetry", {
                "state": state.name,
                "mode": mode.name if mode is not None else None,
                "path_index": path_idx,
                "path_size": len(path) if path is not None else 0,
                "patrol_target": patrol_target,
                "patrol_source": patrol_source,
                "final_target": final_target,
                "goal_source": goal_source,
                "distance_to_target": d_tgt,
                "distance_to_obstacle": d_robot,
                "last_command": last_cmd,
                "fps": fps,
            }, feature="telemetry"):
                debugger.mark_due("telemetry")
        # ===== 目标来源离散事件(保存日志) =====
        # 仅当 goal_source 或活跃目标坐标变化时记录一次, 避免埋在 0.5s 遥测流里难以回看。
        # 同时 print 到 nav_auto.log (NavDebugger 已 tee 落盘), 便于直接 grep 文本日志。
        _active_goal_tgt = (final_target
                            if state in (NavState.FINAL_PLAN, NavState.FINAL_FOLLOW,
                                         NavState.FINAL_ADJUST)
                            else patrol_target)
        _tgt_key = (None if _active_goal_tgt is None
                    else (round(float(_active_goal_tgt[0]), 3),
                          round(float(_active_goal_tgt[1]), 3)))
        # 来源变化一定记; 目标坐标变化通常也记(新 frontier / 新 final)。
        _goal_src_changed = (goal_source != _last_goal_src)
        _patrol_src_changed = (patrol_source != _last_patrol_src)
        _goal_tgt_changed = (_tgt_key != _last_goal_tgt_key)
        if _goal_src_changed or _patrol_src_changed or (_goal_tgt_changed and
                                 state != NavState.FINAL_ADJUST):
            _last_goal_src = goal_source
            _last_patrol_src = patrol_source
            _last_goal_tgt_key = _tgt_key
            debugger.emit("goal", {
                "source": goal_source,
                "patrol_source": patrol_source,
                "active_target": _active_goal_tgt,
                "patrol_nav": patrol_goal_nav,
                "final_nav": final_goal_nav,
                "state": state.name,
            }, feature="state_events")
            _gs = goal_source if goal_source is not None else "none"
            _ps = patrol_source if patrol_source is not None else "none"
            _gt = (f"({_active_goal_tgt[0]:.2f},{_active_goal_tgt[1]:.2f})"
                   if _active_goal_tgt is not None else "none")
            print(f"[nav_2d] >>> GOAL source={_gs} patrol={_ps} target={_gt} "
                  f"state={state.name}")
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
        if path and is_path_blocked(path, path_idx,
                obs_current):
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
                                               NavState.FINAL_FOLLOW,
                                               NavState.FINAL_ADJUST)
                   else patrol_goal_nav)
            if tgt is None:
                tgt = (final_target if state in (NavState.FINAL_PLAN,
                                                 NavState.FINAL_FOLLOW,
                                                 NavState.FINAL_ADJUST)
                       else patrol_target)
            if tgt is not None:
                dx = tgt[0] - cur_pose[0]
                dz = tgt[1] - cur_pose[1]
                dist_str = f"{(dx*dx+dz*dz)**0.5:.2f}m"

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
        ]
        if halted:
            lines.append("[ESTOP] HALTED (space to resume)")

        if state == NavState.SCANNING:
            lines.append(
                f"[SCAN]  {np.degrees(scan_accumulated):.0f}/360°")

        # (网页 vlm/det 按键区域框显示已移除)
        rects = []

        _td0 = time.time()
        # 快照门控: 同一次 due("snapshot") 内成对生成 overlay+map, 避免单侧 due 丢帧
        _want_snap = debugger.due("snapshot", commit=False)
        _want_rgb = debugger.due("rgb_overlay") or _want_snap
        _want_map = debugger.due("map_view") or _want_snap

        overlay_frame = None
        if _want_rgb:
            overlay_frame = draw_debug_overlay(img, {"lines": lines,
                                                       "rects": rects,
                                                       "robot_pose": cur_pose,
                                                       "using_odom": _use_odom,
                                                       "vlm_dets": vlm_latest[0],
                                                       "vlm_masks": vlm_latest[1]})
            mock.set_frame(overlay_frame)

        # ===== 顶视地图 =====
        obs_pts = obs_current
        _mem = slam.memory
        frontier_viz = {"frontier_xz": None, "frontier_dist_m": None,
                        "frontier_ms": 0.0}
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

            frontier_viz = debugger.query_frontier_viz_cached(
                _mem,
                cur_pose,
                nav_y,
                obs_pts,
            )
        frontier_xz = frontier_viz["frontier_xz"]

        map_frame = None
        if _want_map:
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
                "goal_source": goal_source,
            })
            mock.set_map(map_frame)

        # 成对落盘: save_debug_pair 内部再 due("snapshot") 并 mark; 允许单侧 None
        if _want_snap and (overlay_frame is not None or map_frame is not None):
            debugger.save_debug_pair(overlay_frame, map_frame)
        _nav_draw_ms = (time.time() - _td0) * 1000.0   # [NAVSTEP] 绘制耗时

        _nav_now = time.time()
        if _nav_now - _nav_step_t >= 2.0:
            _nav_cyc_ms = (time.time() - prev_t) * 1000.0
            _nav_work_ms = max(0.0, _nav_cyc_ms - _nav_grab_ms - _nav_draw_ms)
            _probe("navstep", f"[NAVSTEP] cyc={_nav_cyc_ms:.0f}ms  grab={_nav_grab_ms:.1f}ms  "
                  f"work={_nav_work_ms:.1f}ms  draw={_nav_draw_ms:.1f}ms  "
                  f"fps={fps:.1f}")
            _nav_step_t = _nav_now

        debugger.record_performance({
            "cycle_ms": (time.time() - prev_t) * 1000.0,
            "grab_ms": _nav_grab_ms,
            "draw_ms": _nav_draw_ms,
            "fps": fps,
        })

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
    debugger.close()      # flush 并关闭 episode 日志文件


if __name__ == "__main__":
    main()
