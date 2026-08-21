#!/usr/bin/env python
"""nav_frontier.py - 纯 Frontier 探索状态机导航 + Web Debug (localhost:5001)。
状态机 WAITING→SCANNING→FRONTIER_SELECT→PATROL/FINAL PLAN/FOLLOW→DONE。
内部复用 NavState.VLM_INQUIRY 表示 FRONTIER_SELECT，但不执行 VLM 方向问询。
坐标系: SLAM X=右 Y=下 Z=前; 导航平面 (X,Z), Y=高度。
(紧急时可通过网页或终端 Space 急停。)
"""

import os
import re
import select
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

from nav_control import (NavState, mock, MotionThread,
                         MOTION_HZ, SCAN_ANGULAR, RELOC_STOP_TIME, RELOC_BACK_TIME,
                         _angle_diff,
                         PointCloudCache, PC_HZ, PC_INTERVAL_DT)
import nav_path
from nav_path import (PATROL_ARRIVE_EPS,
                      FRONTIER_ARRIVE_EPS,
                       FOLLOW_LIN_MAX, FOLLOW_YAW_EMA, FOLLOW_OUT_EMA, FOLLOW_DEADBAND,
                       FOLLOW_STRAIGHT_ALPHA, FOLLOW_SLEW,
                       FOLLOW_WAYPOINT_THRESHOLD,
                      plan_path, follow_path_step,
                      check_nav_point, check_nav_segment,
                      resolve_nav_goal,
                      plan_goal_resolution,
                      make_escape_path_start_line,
                      escape_align_to_yaw_velocity,
                      escape_to_path_start_velocity,
                      ESCAPE_PATH0_ARRIVE_EPS,
                      )

# ============================================================
import nav_page
from nav_page import (YawSmoother, OutputSmoother,
                     draw_debug_overlay,                      draw_map_view)

# ============================================================
from nav_helpers import (
    _cmd_str,
    get_keyframe_pose,
    OdomHolder,
    _odom_nav_pose,
    extract_target_3d_from_snapshot,
    _get_accumulated_map,
    filter_filament_obstacles,
    get_obstacle_points,
    query_frontier_for_viz,
    _goal_walked,
    _presence_is_yes,
    auto_trigger_patrol,
    build_obstacle_snapshot,
    load_camera_calibration,
    project_locked_target_bbox,
)
from nav_vlm import (AsyncVlmWorker, VlmDetector, VlmJob,
                     qwen_bbox_to_pixel)
from nav_constants import (VLM_DETECT_TARGET_DEFAULT,
                         ROBOT_RADIUS,
                         VLM_URL_DEFAULT,
                         VLM_H_PROMPT,
                         VLM_PRESENCE_PROMPT,
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
                         PRETURN_MIN_TURN_DEG,
                         PRETURN_RETREAT_DISTANCE,
                         PRETURN_RETREAT_SPEED,
                         PRETURN_RETREAT_TIMEOUT,
                         DEBUG_FEATURES)
from nav_debug import NavDebugger
from nav_memory.frontier_grid import detect_front_wall
from nav_memory.types import SID_WALKED
from nav_runtime import NavigationRuntime

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


class EstopKeyboardThread:
    """仅监听终端 Space 急停，兼容仍使用旧 KeyboardThread 的 nav_control。"""

    def __init__(self):
        self._running = False
        self._thread = None
        self.available = False
        self._fd = None
        self._old_settings = None

    def start(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Space 急停要求交互式终端")
        import termios
        import tty
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        # cbreak 保留 Ctrl+C 信号，同时让 Space 无需回车即可生效。
        tty.setcbreak(self._fd)
        self.available = True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[nav_2d] 终端 Space 急停已启用")

    def stop(self):
        self._running = False
        self._restore_term()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _restore_term(self):
        if self._fd is not None and self._old_settings is not None:
            import termios
            termios.tcsetattr(
                self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _run(self):
        try:
            while self._running:
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if self._fd not in readable:
                    continue
                data = os.read(self._fd, 1)
                if not data:
                    raise EOFError("终端输入已关闭")
                if data.decode("utf-8", "replace") == " ":
                    new_halt = not mock.get_halt()
                    mock.set_halt(new_halt)
                    print(f"[nav_2d] >>> ESTOP toggle = {new_halt} "
                          "(键盘 Space)")
        finally:
            self._restore_term()



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
    debugger.setup_episode(mode="nav_frontier")

    rospy.init_node("nav_2d", anonymous=True)

    slam = Mast3rSlamWrapper(rgb_topic="/camera_f/color/image_raw")
    slam.start()

    # 仅用于 FINAL_FOLLOW 的 Debug bbox 解析投影；不传给 SLAM，避免改变
    # 当前后端的无标定运行模式。
    _bbox_calibration = None
    try:
        _bbox_calibration = load_camera_calibration(
            os.path.join(_PROJECT_ROOT, "masterslam", "config",
                         "intrinsics.yaml"))
    except Exception as exc:
        print(f"[nav_2d][WARN] bbox 相机标定加载失败: {exc}")

    odom_holder = OdomHolder()
    rospy.Subscriber("/odom", Odometry, odom_holder.cb, queue_size=1)
    print("[nav_2d] Subscribed to /odom (wheel odometry fallback)")

    bot = TracerRobot(base_url="http://localhost:8080")

    # --- VLM 检测器 (o 键, 默认启用; 默认地址同 keyboard_qwen) ---
    vlm_detector = VlmDetector(
        vlm_url=args.vlm_url,
        sam_ckpt=MOBILE_SAM_CHECKPOINT_PATH)
    vlm_worker = AsyncVlmWorker(vlm_detector)
    vlm_worker.start()
    print(f"[nav_2d] VLM 检测已启用: {args.vlm_url} model=(default)")

    nav_page.mock = mock
    if DEBUG_FEATURES.get("web_ui", True):
        nav_page.start_flask(port=5001)

    estop_keyboard = EstopKeyboardThread()
    estop_keyboard.start()

    _nav_stop = threading.Event()
    __sigint_timer = None

    def _on_sigint(signum, frame):
        nonlocal __sigint_timer
        print("\n[nav_2d] 收到 Ctrl+C, 正在退出...")
        _nav_stop.set()
        estop_keyboard._running = False
        estop_keyboard._restore_term()   # 立即恢复终端, 避免退出后无回显
        try:
            if estop_keyboard._fd is not None:
                os.close(estop_keyboard._fd)
        except Exception:
            pass
        try:
            motion_thread.stop()
        except Exception:
            pass
        try:
            vlm_worker.stop(timeout=1.0)
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
    runtime = NavigationRuntime(state)
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
    _nav_invalid_replan_pending = False
    _nav_invalid_replan_reason = None
    final_adjust_t = 0.0      # FINAL_ADJUST 进入时刻 (计时锁定用)
    final_adjust_frames = 0   # FINAL_ADJUST 累积有效检测帧数 (EMA 平滑样本数)
    _final_from_reloc = False  # reloc 来源进入 FINAL_ADJUST: 禁用 EMA 平滑, 每帧直接用检测值
    _last_goal_shift_t = 0.0   # 目标再校验(方案A)节流时间戳
    d_tgt = None
    d_robot = None
    nav_y = None
    obstacle_snapshot = build_obstacle_snapshot(None, fixed_y=0.0, revision=0)
    _obstacle_points_id = None
    _obstacle_revision = 0
    _patrol_plan_fail_cnt = 0
    _patrol_retry_t = 0.0          # nav_auto 自有节流 (PATROL 重规划等待日志), 不复用 nav_path._last_plan_fail_t
    _patrol_frontier_t = 0.0       # nav_auto 自有节流 (无 frontier 时重试查询), 不复用 nav_path._last_plan_fail_t
    _patrol_blocked_diverted = False   # 已因 RRT 连续失败改用 frontier, 防无限 frontier 循环
    _final_plan_fail_cnt = 0
    _final_plan_retry_t = 0.0       # FINAL_PLAN 实际规划节流, 等待地图更新
    _final_adj_plan_fail_cnt = 0   # FINAL_ADJUST/FINAL_FOLLOW 的 path=None 重规划失败计数
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
    _final_bbox_anchor = None  # 锁定目标的世界点、参考 bbox/位姿及标签
    prev_halt = False         # 上一帧的急停状态 (检测上升沿)
    motion_thread = MotionThread(bot, hz=MOTION_HZ)

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
            vlm_worker.stop(timeout=1.0)
        except Exception:
            pass
        try:
            bot.stop()
        except Exception:
            pass
        try:
            estop_keyboard._restore_term()
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
    _escape_via_path0 = False  # 本轮脱困是否以 path[0] 为锚(结束时保留旧 path)
    _escape_path0_drive_mode = None  # 本轮 path[0] 脱困锁定 forward/reverse
    _escape_path0_anchor = None      # 锚点改变时重新选择行驶方向
    _escape_path0_path = None        # 路径对象替换时也解除方向锁定
    _escape_path0_line = None        # 冻结的起点/直线方向/车头航向
    _escape_path0_aligned = False    # 是否已完成一次性原地对准
    _escape_path1_yaw = None         # 到达 path[0] 后对准 path[1] 的冻结航向
    _escape_phase = None             # ESCAPE: plan / path0 / path1_align / fallback_retreat
    _escape_resume_state = None      # 完成后恢复的 FOLLOW / ADJUST 状态
    _escape_target = None            # 从 path_old 冻结的实际导航终点
    _escape_semantic_target = None   # FINAL 入侵时冻结的 VLM 语义目标
    _escape_replan_from_semantic = False
    _escape_plan_fail_cnt = 0
    # 规划成功后的近墙预转向回退子阶段；状态仍保持 PATROL/FINAL FOLLOW。
    _preturn_active = False
    _preturn_start_pose = None
    _preturn_slam_anchor = None
    _preturn_odom_anchor = None
    _preturn_active_time = 0.0
    _preturn_last_tick = 0.0
    _preturn_resume_plan_state = None
    _yaw_sign = -1.0
    NAV_SMOOTH_POS = 0.30   # 位置 (x,z) 收敛系数
    NAV_SMOOTH_YAW = 0.30   # 偏航 (yaw) 收敛系数

    # ===== 自动决策状态（nav_frontier：探索方向始终由 Frontier 提供） =====
    _auto_vlm_asked = False   # 本次 VLM_INQUIRY 是否已问过 VLM 选方向
    _auto_vlm_ask_t = 0.0     # 兼容异步 direction 结果字段；本入口不提交该任务
    _frontier_query_t = 0.0   # FRONTIER_SELECT 查询节流
    _auto_det_t = 0.0         # 上次自动检测时刻 (OBJ_DET_PERIOD 节流)
    _presence_confirmed = False  # 周期自动检测: presence 一旦问询为 yes 即缓存, 后续不再重复问询
    _vlm_epoch = 0
    _vlm_request_id = 0
    # 到达 VLM 方向点后"再次询问"的标记:
    #   False -> 本次检测到方向点已到达/走过, 不应直接去 frontier, 而是重新问 VLM;
    #            同时置 True, 表示"已为重问过一轮"。
    #   True  -> 这是重问后得到的新方向点, 若其 sub-opt goal 也已达/走过, 才是 frontier 点。
    _vlm_reask_pending = False

    def _invalidate_vlm(reason):
        """使状态变化前已提交、但无法取消的 HTTP 结果失效。"""
        nonlocal _vlm_epoch
        _vlm_epoch += 1
        _dbg(f"[VLM-ASYNC] invalidate epoch={_vlm_epoch} reason={reason}")

    def _vlm_snapshot(frame, pose, fixed_y, use_odom,
                      allow_map_fallback, **extra):
        """在主线程冻结图像、逐像素世界点云和射线原点。"""
        if frame is None or pose is None:
            return None, None
        pointcloud = slam.get_pointcloud_2d()
        if pointcloud is None:
            return None, None
        if use_odom:
            _half_yaw = 0.5 * float(pose[2])
            camera_pose = (
                float(pose[0]),
                float(fixed_y if fixed_y is not None else 0.0),
                float(pose[1]),
                0.0, float(np.sin(_half_yaw)), 0.0,
                float(np.cos(_half_yaw)),
            )
        else:
            pose_full = slam.get_pose_full()
            if pose_full is None:
                return None, None
            camera_pose = tuple(float(v) for v in pose_full[:7])
        camera_pos = camera_pose[:3]
        context = {
            "pointcloud_2d": np.array(pointcloud, copy=True),
            "image_shape": tuple(int(v) for v in frame.shape[:2]),
            "camera_pos_3d": camera_pos,
            "camera_pose_3d": camera_pose,
            "nav_y": (None if fixed_y is None else float(fixed_y)),
            "allow_map_fallback": bool(allow_map_fallback),
            "use_odom": bool(use_odom),
            "submit_pose": tuple(float(v) for v in pose),
        }
        context.update(extra)
        return np.array(frame, copy=True), context

    def _submit_vlm(kind, frame, target, prompt, pose, fixed_y,
                    use_odom, allow_map_fallback=False, **context_extra):
        """非阻塞提交；worker 忙或快照不可用时保持主循环继续。"""
        nonlocal _vlm_request_id
        if vlm_worker.busy():
            return False
        image_snapshot, context = _vlm_snapshot(
            frame, pose, fixed_y, use_odom, allow_map_fallback,
            **context_extra)
        if image_snapshot is None or context is None:
            return False
        request_id = _vlm_request_id + 1
        job = VlmJob(
            request_id=request_id,
            kind=kind,
            image=image_snapshot,
            target=target,
            prompt=prompt,
            submit_step=step,
            submit_state=(state.name if hasattr(state, "name") else str(state)),
            epoch=_vlm_epoch,
            context=context,
        )
        if not vlm_worker.submit(job):
            return False
        _vlm_request_id = request_id
        print(f"[VLM-ASYNC] submit id={request_id} kind={kind} "
              f"step={step} state={job.submit_state} epoch={_vlm_epoch}")
        debugger.record_vlm({
            "stage": "async_submit",
            "request_id": request_id,
            "kind": kind,
            "submit_step": step,
            "submit_state": job.submit_state,
            "epoch": _vlm_epoch,
        }, step=step)
        return True

    def _finish_vlm_result(result, action, reason):
        """统一记录一次异步结果的应用/丢弃决定。"""
        latency_ms = max(
            0.0, (float(result.finished_at) - float(result.started_at)) * 1000.0)
        print(f"[VLM-ASYNC] result id={result.request_id} kind={result.kind} "
              f"submit={result.submit_step} result={step} latency={latency_ms:.0f}ms "
              f"action={action} reason={reason}")
        debugger.record_vlm({
            "stage": "async_result",
            "request_id": result.request_id,
            "kind": result.kind,
            "submit_step": result.submit_step,
            "result_step": step,
            "submit_state": result.submit_state,
            "epoch": result.epoch,
            "latency_ms": latency_ms,
            "action": action,
            "reason": reason,
            "error": result.error,
        }, step=step)

    def _reset_intrusion_escape(clear_path0=True):
        """清理 ESCAPE 上下文；完成 path0 时保留一帧 path0 标记供 odom 接管。"""
        nonlocal _escape_phase, _escape_resume_state, _escape_target
        nonlocal _escape_semantic_target, _escape_replan_from_semantic
        nonlocal _escape_plan_fail_cnt, _escape_dir
        nonlocal _escape_via_path0, _escape_path0_drive_mode
        nonlocal _escape_path0_anchor, _escape_path0_path
        nonlocal _escape_path0_line, _escape_path0_aligned
        nonlocal _escape_path1_yaw
        _escape_phase = None
        _escape_resume_state = None
        _escape_target = None
        _escape_semantic_target = None
        _escape_replan_from_semantic = False
        _escape_plan_fail_cnt = 0
        _escape_dir = None
        if clear_path0:
            _escape_via_path0 = False
            _escape_path0_drive_mode = None
            _escape_path0_anchor = None
            _escape_path0_path = None
            _escape_path0_line = None
            _escape_path0_aligned = False
            _escape_path1_yaw = None

    def _begin_intrusion_escape(resume_state, target, clearance):
        """冻结 path_old 的实际目标，废弃旧路径并进入同目标 ESCAPE 重规划。"""
        nonlocal state, path, path_idx
        nonlocal _escape_phase, _escape_resume_state, _escape_target
        nonlocal _escape_semantic_target, _escape_replan_from_semantic
        if target is None:
            return False
        _invalidate_vlm("intrusion_escape")
        _reset_intrusion_escape(clear_path0=True)
        _escape_phase = "plan"
        _escape_resume_state = resume_state
        _escape_target = (float(target[0]), float(target[1]))
        if (resume_state in (NavState.FINAL_FOLLOW, NavState.FINAL_ADJUST,
                             NavState.FINAL_PLAN) and final_target is not None):
            _escape_semantic_target = (
                float(final_target[0]), float(final_target[1]))
        elif (resume_state in (NavState.PATROL_FOLLOW, NavState.PATROL_PLAN)
              and patrol_target is not None):
            _escape_semantic_target = (
                float(patrol_target[0]), float(patrol_target[1]))
        else:
            _escape_semantic_target = None
        _escape_replan_from_semantic = False
        path = None
        path_idx = 0
        motion_thread.stop()
        state = NavState.ESCAPE
        print(f"[INTRUSION] phase=start target=({_escape_target[0]:.3f},"
              f"{_escape_target[1]:.3f}) clearance={float(clearance):.3f}m "
              f"resume={resume_state.name} action=replan_same_target")
        return True

    def _mark_invalid_nav_replan(reason):
        """让下一条 goal 事件标记为由非法 NAV/路径触发的重算。"""
        nonlocal _nav_invalid_replan_pending, _nav_invalid_replan_reason
        _nav_invalid_replan_pending = True
        _nav_invalid_replan_reason = str(reason)

    def _final_plan_goal(cur_pose, obstacle_points, fixed_y):
        """解析 FINAL 安全站位并同时选择最近且 RRT 可达的候选。"""
        nonlocal final_goal_nav
        if final_target is None or cur_pose is None:
            return None, None
        if (final_goal_nav is not None and
                check_nav_point(final_goal_nav, obstacle_snapshot).safe):
            _path, _actual = plan_path(
                (cur_pose[0], cur_pose[1]), final_goal_nav,
                obstacle_points, fixed_y=fixed_y,
                plan_start_2d=((_plan_pose[0], _plan_pose[1])
                               if _plan_pose is not None else None),
                obstacle_snapshot=obstacle_snapshot)
            return _path, _actual
        if final_goal_nav is not None:
            _mark_invalid_nav_replan("final_nav_invalid")
            final_goal_nav = None
        resolution = resolve_nav_goal(
            final_target, (cur_pose[0], cur_pose[1]), obstacle_snapshot,
        )
        _path, _actual, _selected = plan_goal_resolution(
            (cur_pose[0], cur_pose[1]), resolution, obstacle_points,
            fixed_y=fixed_y,
            plan_start_2d=((_plan_pose[0], _plan_pose[1])
                           if _plan_pose is not None else None),
            obstacle_snapshot=obstacle_snapshot)
        final_goal_nav = _selected.goal
        if final_goal_nav is None or _path is None:
            print(f"[nav_2d][GOAL] FINAL 站位解析失败 "
                  f"reason={_selected.reason} clearance={_selected.clearance:.3f}m")
            return None, None
        if _selected.reason != "direct":
            print(f"[nav_2d][GOAL] FINAL semantic=({final_target[0]:.3f},"
                  f"{final_target[1]:.3f}) nav=({final_goal_nav[0]:.3f},"
                  f"{final_goal_nav[1]:.3f}) reason={_selected.reason} "
                  f"shift={_selected.shifted_distance:.3f}m")
        return _path, _actual

    def _lock_final_adjust_if_ready(distance_to_nav):
        """FINAL_ADJUST 计时/帧数锁定；到达 GOAL_NAV 时也必须可执行。"""
        nonlocal state, _final_from_reloc
        if state != NavState.FINAL_ADJUST:
            return False
        _adj_elapsed = time.time() - final_adjust_t
        _reached_min = final_adjust_frames >= FINAL_ADJUST_MIN_FRAMES
        _close = distance_to_nav <= FINAL_ADJUST_LOCK_DIST
        _force_lock = _adj_elapsed >= FINAL_ADJUST_FORCE_TIME
        if not (((_adj_elapsed >= FINAL_ADJUST_TIME) and
                 _reached_min and _close) or _force_lock):
            return False
        if _force_lock and not _close:
            print(f"\033[93m[nav_2d] FINAL_ADJUST 超时强制锁定 "
                  f"(目标仍远 {distance_to_nav:.2f}m, 帧数 "
                  f"{final_adjust_frames}, 终点 "
                  f"({final_target[0]:.2f},{final_target[1]:.2f}))\033[0m")
        else:
            _dbg(f"[nav_2d] FINAL_ADJUST -> FINAL_FOLLOW "
                 f"(平滑 {final_adjust_frames} 帧 / "
                 f"{_adj_elapsed:.1f}s, 目标距 {distance_to_nav:.2f}m "
                 f"<= {FINAL_ADJUST_LOCK_DIST:.2f}m, 锁定终点 "
                 f"({final_target[0]:.2f},{final_target[1]:.2f}))")
        state = NavState.FINAL_FOLLOW
        _final_from_reloc = False
        return True

    def _set_patrol_goal(ft, src="frontier"):
        """设置 F 点目标并切到 PATROL_PLAN (F = frontier 巡逻目标)。
        ft: frontier (x,z)。重置巡逻导航状态。最终目标由 object detection 另行设置。
        src: 目标来源标记, "frontier" 或 "vlm_dir" 等, 用于 HUD/log 说明。
        """
        nonlocal patrol_target, patrol_goal_nav, patrol_goal_nav_target
        nonlocal path, path_idx, _patrol_plan_fail_cnt, _patrol_blocked_diverted
        nonlocal state, _auto_vlm_asked, patrol_source, goal_source
        nonlocal _preturn_active
        _invalidate_vlm("new_patrol_goal")
        _reset_intrusion_escape(clear_path0=True)
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
        _preturn_active = False
        state = NavState.PATROL_PLAN

    def _start_preturn_odom_hold():
        """把预转向回退的冻结锚点交给现有 odom 保持窗口，防 SLAM snap-back。"""
        nonlocal _slam_anchor, _odom_anchor, _post_escape, _post_escape_t
        if _preturn_slam_anchor is None or _preturn_odom_anchor is None:
            return
        _slam_anchor = _preturn_slam_anchor
        _odom_anchor = _preturn_odom_anchor
        _post_escape = True
        _post_escape_t = time.time()

    def _arm_preturn(resume_plan_state, pose, slam_mode, odom_pose,
                     obstacle_points):
        """在一次成功规划后评估大角转向+前墙，并按需启动回退子阶段。"""
        nonlocal _preturn_active, _preturn_start_pose
        nonlocal _preturn_slam_anchor, _preturn_odom_anchor
        nonlocal _preturn_active_time, _preturn_last_tick
        nonlocal _preturn_resume_plan_state
        nonlocal _post_escape

        _preturn_active = False
        turn_deg = 0.0
        wall = None
        action = "skip_invalid_path"
        if pose is not None and path is not None and len(path) >= 2:
            dx = float(path[1][0]) - float(pose[0])
            dz = float(path[1][1]) - float(pose[1])
            if np.isfinite(dx) and np.isfinite(dz) and np.hypot(dx, dz) > 1e-6:
                target_yaw = float(np.arctan2(dx, dz))
                turn_deg = abs(float(np.degrees(
                    _angle_diff(target_yaw, float(pose[2])))))
                if turn_deg < PRETURN_MIN_TURN_DEG:
                    action = "skip_small_turn"
                elif slam_mode is None or slam_mode.name != "TRACKING":
                    action = "skip_not_tracking"
                else:
                    wall = detect_front_wall(
                        (pose[0], pose[1]), pose[2], obstacle_points)
                    if not wall.detected:
                        action = "skip_not_wall"
                    elif odom_pose is None:
                        action = "skip_no_odom"
                    else:
                        _preturn_start_pose = (
                            float(pose[0]), float(pose[1]), float(pose[2]))
                        _preturn_slam_anchor = _preturn_start_pose
                        _preturn_odom_anchor = tuple(float(v) for v in odom_pose)
                        _preturn_active_time = 0.0
                        _preturn_last_tick = time.time()
                        _preturn_resume_plan_state = resume_plan_state
                        _post_escape = False
                        _preturn_active = True
                        action = "retreat"

        if wall is None:
            hits = "-"
            ratio = "-"
            median = "-"
            residual = "-"
            wall_flag = 0
        else:
            hits = f"{wall.n_hits}/{wall.n_rays}"
            ratio = f"{wall.hit_ratio:.3f}"
            median = f"{wall.median_distance:.3f}"
            residual = f"{wall.line_residual:.3f}"
            wall_flag = int(wall.detected)
        print(f"[PRETURN] state={state.name} turn={turn_deg:.1f}deg "
              f"hits={hits} ratio={ratio} median={median}m "
              f"line={residual}m wall={wall_flag} action={action}")
        return _preturn_active

    def _run_preturn(pose, odom_pose, obstacle_points, fixed_y, halted_now):
        """执行一轮循环前处理；返回 True 表示本轮已接管 FOLLOW。"""
        nonlocal _preturn_active, _preturn_active_time, _preturn_last_tick
        nonlocal _preturn_resume_plan_state, path, path_idx, state, last_cmd
        if not _preturn_active:
            return False

        now = time.time()
        if halted_now:
            motion_thread.stop()
            _preturn_last_tick = now  # 急停时间不计入回退超时
            last_cmd = "ESTOP (preturn retreat paused)"
            return True

        if pose is None or odom_pose is None or _preturn_start_pose is None:
            motion_thread.stop()
            _preturn_active = False
            path_idx = 0
            last_cmd = "preturn aborted (pose/odom lost)"
            print("[PRETURN] result=abort reason=pose_or_odom_lost path_idx=0")
            return True

        dt = max(0.0, now - _preturn_last_tick)
        _preturn_last_tick = now
        _preturn_active_time += dt
        start_x, start_z, start_yaw = _preturn_start_pose
        move_x = float(pose[0]) - start_x
        move_z = float(pose[1]) - start_z
        forward_x = float(np.sin(start_yaw))
        forward_z = float(np.cos(start_yaw))
        back_progress = max(0.0, -(move_x * forward_x + move_z * forward_z))

        if back_progress >= PRETURN_RETREAT_DISTANCE:
            motion_thread.stop()
            _preturn_active = False
            _start_preturn_odom_hold()
            resume_state = _preturn_resume_plan_state
            _preturn_resume_plan_state = None
            blocked = True
            if path is not None and len(path) >= 2:
                try:
                    blocked = not check_nav_segment(
                        (pose[0], pose[1]), path[1], obstacle_snapshot).safe
                except Exception as exc:
                    print(f"[PRETURN] 连接段复检异常 {exc!r}, 按阻挡处理")
                    blocked = True
            if blocked:
                path = None
                path_idx = 0
                if resume_state is not None:
                    state = resume_state
                last_cmd = "preturn done; reconnect blocked -> replan"
                print(f"[PRETURN] result=done back={back_progress:.3f}m "
                      f"reconnect=blocked action=replan state={state.name}")
            else:
                path_idx = 1
                yaw_smoother.reset()
                out_smoother.reset()
                last_cmd = "preturn done; follow path[1]"
                print(f"[PRETURN] result=done back={back_progress:.3f}m "
                      f"reconnect=free action=follow path_idx=1")
            return True

        if _preturn_active_time >= PRETURN_RETREAT_TIMEOUT:
            motion_thread.stop()
            _preturn_active = False
            if back_progress > 0.01:
                _start_preturn_odom_hold()
            _preturn_resume_plan_state = None
            path_idx = 0
            last_cmd = "preturn timeout; follow original path"
            print(f"[PRETURN] result=abort reason=timeout "
                  f"back={back_progress:.3f}m path_idx=0")
            return True

        _rec_cmd(-PRETURN_RETREAT_SPEED, 0.0, pose)
        motion_thread.set_velocity(-PRETURN_RETREAT_SPEED, 0.0)
        motion_thread.start()
        last_cmd = (f"preturn retreat back={back_progress:.2f}/"
                    f"{PRETURN_RETREAT_DISTANCE:.2f}m")
        return True

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
            # path0 脱困: 保留旧 path，并沿脱困分支留下的索引续跟，
            # 避免作废后重规划再入困；未真正到达时该索引仍为 0。
            # 后退脱困 / 无 path: 仍作废旧路径, 让 FOLLOW 下一帧用最新地图重规划。
            if (path is not None and len(path) > 0 and _escape_via_path0):
                path_idx = min(max(int(path_idx), 0), len(path))
                print(f"[ODOM] 脱困结束(path0) -> 保留旧 path, "
                      f"path_idx={path_idx}; 进入 odom 保持窗口 "
                      f"(等 SLAM 追上)")
            else:
                path = None
                print("[ODOM] 脱困结束 -> 进入 odom 保持窗口 (等 SLAM 追上); "
                      "作废旧路径待重规划")
            _escape_via_path0 = False
            _escape_path0_drive_mode = None
            _escape_path0_anchor = None
            _escape_path0_path = None
            _escape_path0_line = None
            _escape_path0_aligned = False
            _escape_path1_yaw = None
        if _escaping:
            _post_escape = False   # 重新进入脱困, 不需要保持窗口
        _prev_escaping = _escaping
        if cur_pose is not None:
            odom_now = odom_holder.get()
            _use_odom = ((mode is not None and mode.name == "RELOC")
                         or _escaping or _post_escape or _preturn_active)
            if _use_odom and odom_now is not None:
                _slam_pose_fresh = cur_pose   # 当前 SLAM 位姿(回退判定用, 回退前不覆盖)
                if (_preturn_active and _preturn_slam_anchor is not None and
                        _preturn_odom_anchor is not None):
                    # 预转向回退期间锚点必须冻结，进度完全由 odom 相对位移给出。
                    _odom_pose = _odom_nav_pose(
                        _preturn_slam_anchor, _preturn_odom_anchor, odom_now,
                        yaw_sign=_yaw_sign)
                else:
                    if _odom_anchor is None or _slam_anchor is None:
                        _slam_anchor = _slam_pose_fresh
                        _odom_anchor = odom_now
                    _odom_pose = _odom_nav_pose(
                        _slam_anchor, _odom_anchor, odom_now,
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

        # VLM 结果只在主线程非阻塞消费。先做 epoch/状态有效性过滤，
        # 各 kind 的具体应用留给对应状态分支。
        _vlm_result = vlm_worker.poll()
        if _vlm_result is not None:
            if _vlm_result.kind in ("auto_detect", "reloc_presence",
                                     "reloc_detect"):
                _auto_det_t = float(_vlm_result.finished_at)
            elif _vlm_result.kind == "direction":
                _auto_vlm_ask_t = float(_vlm_result.finished_at)
            _vlm_discard_reason = None
            if _vlm_result.epoch != _vlm_epoch:
                _vlm_discard_reason = "stale_epoch"
            elif (_vlm_result.kind == "direction" and
                  (state != NavState.VLM_INQUIRY or
                   (mode is not None and mode.name == "RELOC"))):
                _vlm_discard_reason = "state_not_vlm_inquiry"
            elif (_vlm_result.kind == "reloc_presence" and
                  (mode is None or mode.name != "RELOC" or
                   _reloc_phase != "spin")):
                _vlm_discard_reason = "reloc_spin_ended"
            elif (_vlm_result.kind == "reloc_detect" and
                  (mode is None or mode.name != "RELOC" or
                   _reloc_phase != "detect_hold")):
                _vlm_discard_reason = "reloc_detect_hold_ended"
            elif (_vlm_result.kind == "auto_detect" and
                  ((mode is not None and mode.name == "RELOC") or
                   state in (NavState.ESCAPE, NavState.DONE))):
                _vlm_discard_reason = "high_priority_state"
            if _vlm_discard_reason is not None:
                _finish_vlm_result(
                    _vlm_result, "discard", _vlm_discard_reason)
                if _vlm_result.kind == "direction":
                    _auto_vlm_asked = False
                _vlm_result = None

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

        # ===== RELOC 主动恢复 (仅旁路正常状态机, 不改动 SCANNING 本身) =====
        if (not _reloc_giveup) and \
                mode is not None and mode.name == "RELOC":
            # RELOC 自动恢复阶段机: stop -> back -> spin；presence=yes 时临时进入
            # detect_hold 停车等待 bbox，失败后再回 spin。
            # 视觉检测挑出逻辑不再放在阶段机之前 (否则一进 reloc 就被挑出, 永远转不了圈),
            # 而是移到下方 spin 分支内: reloc 先完整 stop->back->spin 转圈尝试恢复 tracking,
            # 只有 spin(绕圈) 期间视觉看到可信目标才挑出转 FINAL_ADJUST 追 —— 满足"reloc 该转转"。
            if _reloc_phase is None:
                _invalidate_vlm("enter_reloc")
                _reloc_saved_state = state
                if state == NavState.ESCAPE:
                    _reloc_saved_state = (_escape_resume_state
                                          if _escape_resume_state is not None
                                          else NavState.WAITING)
                    _reset_intrusion_escape(clear_path0=True)
                    path = None
                    path_idx = 0
                    print("[INTRUSION] phase=abort reason=tracking_reloc "
                          f"action=resume_{_reloc_saved_state.name}")
                else:
                    _escape_path0_drive_mode = None
                    _escape_path0_anchor = None
                    _escape_path0_path = None
                    _escape_path0_line = None
                    _escape_path0_aligned = False
                    _escape_path1_yaw = None
                if _preturn_active:
                    _preturn_active = False
                    _preturn_resume_plan_state = None
                    path_idx = 0
                    print("[PRETURN] result=abort reason=tracking_reloc path_idx=0")
                _reloc_phase = "stop"
                _reloc_t = time.time()
                _reloc_start_t = time.time()
                _reloc_prev_state = _reloc_saved_state
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
            elif _reloc_phase in ("spin", "detect_hold"):
                if _reloc_phase == "spin":
                    if not halted:
                        motion_thread.set_velocity(0.0, SCAN_ANGULAR)
                        motion_thread.start()
                    last_cmd = "RELOC (绕圈找 tracking)"
                else:
                    # presence=yes 后，bbox 请求完成前必须持续保持静止。
                    motion_thread.stop()
                    last_cmd = "RELOC (目标确认，停车检测 bbox)"
                # ---- spin 阶段先做 presence；yes 后停车，再做 bbox ----
                # 仅在此阶段允许挑出: reloc 已先 stop->back->spin 完整转圈尝试恢复 tracking,
                # 绕圈期间若视觉看到目标 (用 odom 投影, 因 SLAM 仍在 RELOC 位姿不可信)
                # 才转 FINAL_ADJUST 追, 不压制 reloc 转圈找 tracking 的主路径。
                _det_now = time.time()
                if (_vlm_result is not None and
                        _vlm_result.kind == "reloc_presence"):
                    _res = _vlm_result
                    _vlm_result = None
                    _ctx = _res.context
                    _det_tgt = str(_ctx.get(
                        "detect_target", VLM_DETECT_TARGET_DEFAULT))
                    _present = (_res.error is None and
                                _presence_is_yes(_res.answer))
                    debugger.record_vlm({
                        "stage": "reloc_presence",
                        "request_id": _res.request_id,
                        "submit_step": _res.submit_step,
                        "result_step": step,
                        "answer": _res.answer,
                        "present": _present,
                        "error": _res.error,
                    }, step=step)
                    if _res.error:
                        print(f"\033[91m[RELOC][PRESENCE] VLM 错误: "
                              f"{_res.error}\033[0m")
                        _finish_vlm_result(
                            _res, "apply", "presence_error")
                    elif _present:
                        motion_thread.stop()
                        _reloc_phase = "detect_hold"
                        _reloc_t = time.time()
                        _finish_vlm_result(
                            _res, "apply", "presence_yes_stop")
                        print(f"[RELOC][PRESENCE] 画面存在 '{_det_tgt}' "
                              "-> 停止旋转并检测 bbox")
                    else:
                        _finish_vlm_result(
                            _res, "apply", "presence_no")
                        print(f"[RELOC][PRESENCE] 画面无 '{_det_tgt}' "
                              "-> 继续旋转")
                if (_vlm_result is not None and
                        _vlm_result.kind == "reloc_detect"):
                    _res = _vlm_result
                    _vlm_result = None
                    _vdets, _vmasks = _res.dets, _res.masks
                    _ctx = _res.context
                    _det_tgt = str(_ctx.get(
                        "detect_target", VLM_DETECT_TARGET_DEFAULT))
                    if _res.error:
                        print(f"\033[91m[RELOC][DET] VLM 错误: {_res.error}\033[0m")
                        vlm_latest = (None, None)
                        _finish_vlm_result(_res, "apply", "detect_error")
                        _reloc_phase = "spin"
                    elif _vdets:
                        _vb = _vdets[0].get("bbox_2d")
                        if _vb and len(_vb) == 4:
                            h, w = _ctx["image_shape"]
                            _x1, _y1, _x2, _y2 = qwen_bbox_to_pixel(
                                _vb, w, h)
                            _map_fallback = (_get_accumulated_map(slam)
                                             if _ctx.get("allow_map_fallback")
                                             else None)
                            _t3d = extract_target_3d_from_snapshot(
                                (_x1, _y1, _x2, _y2),
                                _ctx.get("pointcloud_2d"),
                                _ctx.get("image_shape"),
                                _ctx.get("camera_pos_3d"),
                                nav_y=_ctx.get("nav_y"),
                                allow_map_fallback=bool(
                                    _ctx.get("allow_map_fallback")),
                                map_points=_map_fallback,
                                debugger=debugger,
                                debug_step=_res.submit_step)
                            if debugger.enabled("coord_debug"):
                                debugger.record_coord({
                                    "stage": "project",
                                    "state": "RELOC_spin",
                                    "request_id": _res.request_id,
                                    "submit_step": _res.submit_step,
                                    "result_step": step,
                                    "use_odom": bool(_ctx.get("use_odom")),
                                    "final_from_reloc": True,
                                    "bbox_px": [_x1, _y1, _x2, _y2],
                                    "img_size": [int(w), int(h)],
                                    "cam_pos_3d": list(_ctx.get("camera_pos_3d")),
                                    "cur_pose": list(_ctx.get("submit_pose")),
                                    "allow_map_fallback": bool(
                                        _ctx.get("allow_map_fallback")),
                                    "target_3d": ([float(v) for v in _t3d]
                                                  if _t3d is not None else None),
                                }, step=step)
                            if _t3d is not None:
                                # 提前挑出 reloc: 规划使用结果返回时的当前位姿/地图。
                                _finish_vlm_result(_res, "apply", "target_3d")
                                _invalidate_vlm("reloc_target")
                                motion_thread.stop()
                                _reloc_phase = None
                                _reloc_giveup = True
                                final_target = (_t3d[0], _t3d[2])
                                final_goal_nav = None
                                _final_from_reloc = True
                                goal_source = "vlm_det"
                                state = NavState.FINAL_ADJUST
                                final_adjust_t = time.time()
                                final_adjust_frames = 1
                                path = None
                                path_idx = 0
                                vlm_latest = (_vdets, _vmasks)
                                _final_bbox_anchor = {
                                    "target_world": tuple(
                                        float(v) for v in _t3d[:3]),
                                    "bbox_px": (_x1, _y1, _x2, _y2),
                                    "image_shape": tuple(_ctx["image_shape"]),
                                    "camera_pose": tuple(
                                        _ctx["camera_pose_3d"]),
                                    "detection": dict(_vdets[0]),
                                }
                                print(f"[RELOC] 检测到 '{_det_tgt}' -> 挑出 RELOC "
                                      f"转 FINAL_ADJUST 追目标 "
                                      f"({final_target[0]:.2f},{final_target[1]:.2f})")
                                debugger.record_vlm({
                                    "stage": "reloc_detect_exit",
                                    "request_id": _res.request_id,
                                    "submit_step": _res.submit_step,
                                    "result_step": step,
                                    "bbox_2d": list(_vb),
                                    "target_xz": final_target,
                                }, step=step)
                                continue
                            print("[RELOC][DET] bbox->3D 失败, 继续 reloc")
                            vlm_latest = (None, None)
                            _finish_vlm_result(_res, "apply", "bbox_3d_fail")
                            _reloc_phase = "spin"
                        else:
                            vlm_latest = (None, None)
                            _finish_vlm_result(_res, "apply", "invalid_bbox")
                            _reloc_phase = "spin"
                    else:
                        vlm_latest = (None, None)
                        _finish_vlm_result(_res, "apply", "no_detection")
                        _reloc_phase = "spin"

                if (_reloc_phase == "spin" and
                        _det_now - _auto_det_t >= OBJ_DET_PERIOD and
                        img is not None and cur_pose is not None):
                    _det_tgt = VLM_DETECT_TARGET_DEFAULT
                    _submit_vlm(
                        "reloc_presence", img, _det_tgt,
                        VLM_PRESENCE_PROMPT,
                        cur_pose, nav_y, _use_odom,
                        allow_map_fallback=True,
                        detect_target=_det_tgt,
                        trigger="reloc_spin",
                        final_from_reloc=True)
                elif (_reloc_phase == "detect_hold" and
                      not vlm_worker.busy() and
                      img is not None and cur_pose is not None):
                    _det_tgt = VLM_DETECT_TARGET_DEFAULT
                    _submit_vlm(
                        "reloc_detect", img, _det_tgt, None,
                        cur_pose, nav_y, _use_odom,
                        allow_map_fallback=True,
                        detect_target=_det_tgt,
                        trigger="reloc_detect_hold",
                        final_from_reloc=True)
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
        if id(obs_current) != _obstacle_points_id:
            _obstacle_points_id = id(obs_current)
            _obstacle_revision += 1
            obstacle_snapshot = build_obstacle_snapshot(
                obs_current,
                fixed_y=(nav_y if nav_y is not None else 0.0),
                revision=_obstacle_revision,
            )

        # (网页 f 键 frontier 请求逻辑已移除: 防止误触打断导航状态机, 仅保留空格急停)

        # ===== AUTO: 周期目标检测 (o 键逻辑) =====
        _det_now = time.time()
        _det_period = (FINAL_ADJUST_DET_PERIOD
                       if state == NavState.FINAL_ADJUST else OBJ_DET_PERIOD)
        if _vlm_result is not None and _vlm_result.kind == "auto_detect":
            _res = _vlm_result
            _vlm_result = None
            _ctx = _res.context
            _vdets, _vmasks = _res.dets, _res.masks
            _det_tgt = str(_ctx.get(
                "detect_target", VLM_DETECT_TARGET_DEFAULT))
            _trigger = str(_ctx.get("trigger", "auto"))
            _need_presence = bool(_ctx.get("need_presence", True))
            _present = ((not _need_presence) or
                        _presence_is_yes(_res.answer))
            if _need_presence:
                debugger.record_vlm({
                    "stage": "presence",
                    "request_id": _res.request_id,
                    "submit_step": _res.submit_step,
                    "result_step": step,
                    "answer": _res.answer,
                    "present": _present,
                    "trigger": _trigger,
                }, step=step)
            if _present and state != NavState.FINAL_ADJUST:
                _presence_confirmed = True

            if _res.error:
                print(f"\033[91m[nav_auto][DET] VLM 错误: {_res.error}\033[0m")
                vlm_latest = (None, None)
                debugger.record_vlm({
                    "stage": "detect_error",
                    "request_id": _res.request_id,
                    "submit_step": _res.submit_step,
                    "result_step": step,
                    "detail": str(_res.error),
                    "trigger": _trigger,
                }, step=step)
                _finish_vlm_result(_res, "apply", "detect_error")
            elif not _present:
                print(f"[nav_auto][DET] 问询: 画面无 '{_det_tgt}' "
                      f"-> 跳过检测 (answer='{_res.answer}')")
                vlm_latest = (None, None)
                _finish_vlm_result(_res, "apply", "presence_no")
            elif _vdets:
                _vb = _vdets[0].get("bbox_2d")
                if _vb and len(_vb) == 4:
                    h, w = _ctx["image_shape"]
                    _x1, _y1, _x2, _y2 = qwen_bbox_to_pixel(
                        _vb, w, h)
                    _bbox_px = [_x1, _y1, _x2, _y2]
                    _allow_fb = bool(_ctx.get("allow_map_fallback"))
                    _map_fallback = (_get_accumulated_map(slam)
                                     if _allow_fb else None)
                    _t3d = extract_target_3d_from_snapshot(
                        (_x1, _y1, _x2, _y2),
                        _ctx.get("pointcloud_2d"),
                        _ctx.get("image_shape"),
                        _ctx.get("camera_pos_3d"),
                        nav_y=_ctx.get("nav_y"),
                        allow_map_fallback=_allow_fb,
                        map_points=_map_fallback,
                        debugger=debugger,
                        debug_step=_res.submit_step)
                    if debugger.enabled("coord_debug"):
                        debugger.record_coord({
                            "stage": "project",
                            "state": (state.name if hasattr(state, "name")
                                      else str(state)),
                            "request_id": _res.request_id,
                            "submit_step": _res.submit_step,
                            "result_step": step,
                            "use_odom": bool(_ctx.get("use_odom")),
                            "final_from_reloc": bool(_final_from_reloc),
                            "bbox_px": _bbox_px,
                            "cam_pos_3d": list(_ctx.get("camera_pos_3d")),
                            "cur_pose": list(_ctx.get("submit_pose")),
                            "allow_map_fallback": _allow_fb,
                            "target_3d": ([float(v) for v in _t3d]
                                          if _t3d is not None else None),
                        }, step=step)
                    if _t3d is not None:
                        _state_before_det = state
                        _target_accepted = False
                        _locked_t3d = tuple(float(v) for v in _t3d[:3])
                        if state == NavState.FINAL_ADJUST:
                            if _final_from_reloc or final_target is None:
                                final_target = (_t3d[0], _t3d[2])
                                _target_accepted = True
                                final_goal_nav = None
                                final_adjust_frames += 1
                                path = None
                            else:
                                _dx = _t3d[0] - final_target[0]
                                _dz = _t3d[2] - final_target[1]
                                _jump = (_dx * _dx + _dz * _dz) ** 0.5
                                if _jump <= FINAL_ADJUST_REJECT_DIST:
                                    final_target = (
                                        FINAL_ADJUST_SMOOTH_ALPHA * _t3d[0]
                                        + (1.0 - FINAL_ADJUST_SMOOTH_ALPHA) * final_target[0],
                                        FINAL_ADJUST_SMOOTH_ALPHA * _t3d[2]
                                        + (1.0 - FINAL_ADJUST_SMOOTH_ALPHA) * final_target[1])
                                    if _final_bbox_anchor is not None:
                                        _old_t3d = \
                                            _final_bbox_anchor["target_world"]
                                        _locked_t3d = tuple(
                                            FINAL_ADJUST_SMOOTH_ALPHA
                                            * float(_t3d[i])
                                            + (1.0 -
                                               FINAL_ADJUST_SMOOTH_ALPHA)
                                            * float(_old_t3d[i])
                                            for i in range(3))
                                    _target_accepted = True
                                    final_goal_nav = None
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
                            _target_accepted = True
                            final_goal_nav = None
                            goal_source = "vlm_det"
                        _final_plan_fail_cnt = 0
                        _final_plan_retry_t = 0.0
                        print(f"[nav_auto][DET] VLM 目标 "
                              f"({final_target[0]:.2f},{final_target[1]:.2f}) "
                              f"label={_vdets[0].get('label','?')} "
                              f"-> FINAL_ADJUST"
                              f"{' (EMA平滑)' if state == NavState.FINAL_ADJUST else ''}")
                        if state in (
                                NavState.SCANNING,
                                NavState.VLM_INQUIRY,
                                NavState.PATROL_PLAN,
                                NavState.PATROL_FOLLOW):
                            path = None
                            path_idx = 0
                            final_adjust_t = time.time()
                            final_adjust_frames = 1
                            state = NavState.FINAL_ADJUST
                        vlm_latest = (_vdets, _vmasks)
                        if (_target_accepted and
                                _ctx.get("camera_pose_3d") is not None):
                            _final_bbox_anchor = {
                                "target_world": _locked_t3d,
                                "bbox_px": tuple(_bbox_px),
                                "image_shape": tuple(_ctx["image_shape"]),
                                "camera_pose": tuple(
                                    _ctx["camera_pose_3d"]),
                                "detection": dict(_vdets[0]),
                            }
                        debugger.record_vlm({
                            "stage": "detect",
                            "request_id": _res.request_id,
                            "submit_step": _res.submit_step,
                            "result_step": step,
                            "bbox_2d": list(_vb),
                            "bbox_2d_px": _bbox_px,
                            "target_xz": final_target,
                            "manual": False,
                            "trigger": _trigger,
                        }, step=step)
                        _finish_vlm_result(_res, "apply", "target_3d")
                        if _state_before_det != state:
                            _invalidate_vlm("new_final_target")
                    else:
                        print("[nav_auto][DET] bbox->3D 失败")
                        vlm_latest = (None, None)
                        debugger.record_vlm({
                            "stage": "bbox_3d_fail",
                            "request_id": _res.request_id,
                            "submit_step": _res.submit_step,
                            "result_step": step,
                            "bbox_2d": list(_vb),
                            "bbox_2d_px": _bbox_px,
                            "trigger": _trigger,
                        }, step=step)
                        _finish_vlm_result(_res, "apply", "bbox_3d_fail")
                else:
                    print("[nav_auto][DET] VLM 未返回有效 bbox")
                    vlm_latest = (None, None)
                    debugger.record_vlm({
                        "stage": "no_detection",
                        "request_id": _res.request_id,
                        "submit_step": _res.submit_step,
                        "result_step": step,
                        "detail": "invalid_bbox",
                        "trigger": _trigger,
                    }, step=step)
                    _finish_vlm_result(_res, "apply", "invalid_bbox")
            else:
                print(f"[nav_auto][DET] 未检测到目标 "
                      f"(target='{_det_tgt}') -> 继续 follow path")
                vlm_latest = (None, None)
                debugger.record_vlm({
                    "stage": "no_detection",
                    "request_id": _res.request_id,
                    "submit_step": _res.submit_step,
                    "result_step": step,
                    "detail": f"empty_dets target='{_det_tgt}'",
                    "trigger": _trigger,
                }, step=step)
                _finish_vlm_result(_res, "apply", "no_detection")

        if (_det_now - _auto_det_t >= _det_period and
                img is not None and cur_pose is not None and
                ((final_target is None and
                  state not in (NavState.FINAL_PLAN, NavState.FINAL_FOLLOW))
                 or state == NavState.FINAL_ADJUST)):
            _det_tgt = VLM_DETECT_TARGET_DEFAULT
            _trigger = "auto"
            _need_presence = not (
                _presence_confirmed and state != NavState.FINAL_ADJUST)
            _submit_vlm(
                "auto_detect", img, _det_tgt, VLM_PRESENCE_PROMPT,
                cur_pose, nav_y, _use_odom,
                allow_map_fallback=(state == NavState.FINAL_ADJUST),
                detect_target=_det_tgt,
                trigger=_trigger,
                need_presence=_need_presence,
                final_from_reloc=bool(_final_from_reloc))

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
            # nav_frontier 保留该枚举以兼容现有状态机和 Debug 页面；
            # 这里的实际语义是 FRONTIER_SELECT，不提交 VLM direction 请求。
            _prn_state(state)
            motion_thread.stop()
            last_cmd = "idle (frontier select)"
            if cur_pose is None:
                last_cmd = "stop (frontier: pose unavailable)"
            else:
                _frontier_now = time.time()
                if _frontier_now - _frontier_query_t >= 1.0:
                    _frontier_query_t = _frontier_now
                    _ft, _nst = auto_trigger_patrol(
                        slam, cur_pose, nav_y, obs_current)
                    if _ft is not None:
                        _set_patrol_goal(_ft, src="frontier")
                        _dbg(f"[nav_frontier] frontier={_ft} -> {_nst}")
                    else:
                        last_cmd = "stop (no frontier, wait map)"
                        _dbg("[nav_frontier] 无可用 frontier, 等待地图更新")
                else:
                    last_cmd = "idle (frontier query cooldown)"

        elif state == NavState.ESCAPE:
            _prn_state(state)
            _fixed_y = nav_y if nav_y is not None else 0.0
            if halted:
                motion_thread.stop()
                if _escape_phase in ("path0", "path1_align", "fallback_retreat"):
                    _escaping = True
                last_cmd = "ESTOP (intrusion escape paused)"
            elif cur_pose is None:
                motion_thread.stop()
                if _escape_phase in ("path0", "path1_align", "fallback_retreat"):
                    _escaping = True
                last_cmd = "intrusion escape waiting pose"
            elif _escape_target is None or _escape_resume_state is None:
                motion_thread.stop()
                print("[INTRUSION] phase=abort reason=missing_context action=waiting")
                _reset_intrusion_escape(clear_path0=True)
                path = None
                path_idx = 0
                state = NavState.WAITING
            else:
                _esc_obs = obs_current
                _esc_tree = obstacle_snapshot.tree
                _robot_check = check_nav_point(
                    (cur_pose[0], cur_pose[1]), obstacle_snapshot)
                _esc_clearance = _robot_check.clearance

                if _escape_phase == "plan":
                    motion_thread.stop()
                    _new_path, _actual_goal = None, None
                    _target_check = check_nav_point(
                        _escape_target, obstacle_snapshot)
                    if (not _target_check.safe and
                            _escape_semantic_target is not None):
                        _resolution = resolve_nav_goal(
                            _escape_semantic_target,
                            (cur_pose[0], cur_pose[1]),
                            obstacle_snapshot)
                        _resolved_path, _actual_goal, _selected = \
                            plan_goal_resolution(
                                (cur_pose[0], cur_pose[1]), _resolution,
                                _esc_obs, fixed_y=_fixed_y,
                                plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                               if _plan_pose is not None else None),
                                obstacle_snapshot=obstacle_snapshot)
                        if _selected.goal is not None and _resolved_path is not None:
                            _escape_target = _selected.goal
                            _new_path = _resolved_path
                            if _escape_resume_state in (
                                    NavState.FINAL_FOLLOW,
                                    NavState.FINAL_ADJUST,
                                    NavState.FINAL_PLAN):
                                final_goal_nav = _selected.goal
                            else:
                                patrol_goal_nav = _selected.goal
                            _escape_replan_from_semantic = False
                            print(f"[INTRUSION] phase=plan nav_goal_reselected "
                                  f"clearance={_selected.clearance:.3f}m "
                                  f"action=plan_same_semantic_target")
                        else:
                            _escape_replan_from_semantic = True
                    if (_escape_replan_from_semantic and
                            not check_nav_point(
                                _escape_target, obstacle_snapshot).safe):
                        _new_path, _actual_goal = None, None
                    elif _new_path is None:
                        _new_path, _actual_goal = plan_path(
                            (cur_pose[0], cur_pose[1]),
                            _escape_target,
                            _esc_obs,
                            fixed_y=_fixed_y,
                            plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                           if _plan_pose is not None else None),
                            obstacle_snapshot=obstacle_snapshot)
                    _path0_line = (
                        make_escape_path_start_line(cur_pose, _new_path[0])
                        if _new_path is not None and len(_new_path) >= 1
                        else None)
                    if (_new_path is not None and len(_new_path) >= 2 and
                            _path0_line is not None):
                        path = _new_path
                        path_idx = 0
                        _escape_plan_fail_cnt = 0
                        _escape_phase = "path0"
                        _escape_via_path0 = True
                        _escape_path0_drive_mode = _path0_line["drive_mode"]
                        _escape_path0_anchor = (
                            float(path[0][0]), float(path[0][1]))
                        _escape_path0_path = path
                        _escape_path0_line = _path0_line
                        _escape_path0_aligned = False
                        _escape_path1_yaw = None
                        _escape_dir = None
                        yaw_smoother.reset()
                        out_smoother.reset()
                        _root_clearance = check_nav_point(
                            _escape_path0_anchor,
                            obstacle_snapshot).clearance
                        print(f"[INTRUSION] phase=plan target=({_escape_target[0]:.3f},"
                              f"{_escape_target[1]:.3f}) attempt=0 "
                              f"path0=({_escape_path0_anchor[0]:.3f},"
                              f"{_escape_path0_anchor[1]:.3f}) "
                              f"clearance={_root_clearance:.3f}m "
                              f"drive={_escape_path0_drive_mode} "
                              f"rel={np.degrees(_path0_line['relative_angle']):.1f}deg "
                              f"fixed_yaw={np.degrees(_path0_line['fixed_yaw']):.1f}deg "
                              f"action=align_then_straight")
                        last_cmd = "intrusion path_new ready"
                    else:
                        path = None
                        path_idx = 0
                        _escape_plan_fail_cnt += 1
                        _reason = ("nav_goal_unresolved"
                                   if _escape_replan_from_semantic
                                   else "rrt_or_path_validation_failed")
                        if _escape_plan_fail_cnt >= PLAN_FAIL_MAX:
                            _escape_phase = "fallback_retreat"
                            _escape_dir = nav_path.find_escape_dir(
                                (cur_pose[0], cur_pose[1]),
                                _esc_tree, fixed_y=_fixed_y)
                            _action = "fallback_retreat"
                        else:
                            _action = "stop_retry"
                        print(f"[INTRUSION] phase=plan target=({_escape_target[0]:.3f},"
                              f"{_escape_target[1]:.3f}) "
                              f"attempt={_escape_plan_fail_cnt}/{PLAN_FAIL_MAX} "
                              f"reason={_reason} action={_action}")
                        last_cmd = f"intrusion plan failed ({_escape_plan_fail_cnt})"

                elif _escape_phase == "path0":
                    _path0_valid = (
                        path is not None and len(path) >= 2 and
                        _escape_path0_path is path and
                        _escape_path0_line is not None and
                        _escape_path0_anchor ==
                        (float(path[0][0]), float(path[0][1])))
                    if not _path0_valid:
                        motion_thread.stop()
                        path = None
                        path_idx = 0
                        _escape_phase = "plan"
                        _escape_path0_drive_mode = None
                        _escape_path0_anchor = None
                        _escape_path0_path = None
                        _escape_path0_line = None
                        _escape_path0_aligned = False
                        _escape_path1_yaw = None
                        _escape_via_path0 = False
                        print("[INTRUSION] phase=path0 reason=path_replaced "
                              "action=replan_same_target")
                    else:
                        _p0_dist = float(np.hypot(
                            _escape_path0_anchor[0] - float(cur_pose[0]),
                            _escape_path0_anchor[1] - float(cur_pose[1])))
                        _line_start = _escape_path0_line["start"]
                        _line_unit = _escape_path0_line["unit"]
                        _travel_x = float(cur_pose[0]) - float(_line_start[0])
                        _travel_z = float(cur_pose[1]) - float(_line_start[1])
                        _line_progress = (
                            _travel_x * float(_line_unit[0]) +
                            _travel_z * float(_line_unit[1]))
                        _line_remaining = (
                            float(_escape_path0_line["distance"]) -
                            _line_progress)
                        _line_cross = abs(
                            _travel_x * float(_line_unit[1]) -
                            _travel_z * float(_line_unit[0]))
                        _line_reached = (
                            _line_remaining <= ESCAPE_PATH0_ARRIVE_EPS or
                            _p0_dist <= ESCAPE_PATH0_ARRIVE_EPS)
                        if _line_reached and _robot_check.safe:
                            motion_thread.stop()
                            _p1_dx = float(path[1][0]) - float(path[0][0])
                            _p1_dz = float(path[1][1]) - float(path[0][1])
                            if np.hypot(_p1_dx, _p1_dz) <= 1e-6:
                                path_idx = 1
                                _resume_state = _escape_resume_state
                                _resume_target = _escape_target
                                _escaping = False
                                _reset_intrusion_escape(clear_path0=False)
                                state = _resume_state
                                yaw_smoother.reset()
                                out_smoother.reset()
                                print(f"[INTRUSION] phase=path0 "
                                      f"target=({_resume_target[0]:.3f},"
                                      f"{_resume_target[1]:.3f}) "
                                      f"dist={_p0_dist:.3f}m "
                                      f"progress={_line_progress:.3f}m "
                                      f"cross={_line_cross:.3f}m "
                                      f"clearance={_esc_clearance:.3f}m "
                                      f"reason=degenerate_path1 "
                                      f"action=resume_path path_idx={path_idx} "
                                      f"state={state.name}")
                                last_cmd = "intrusion path1 degenerate; resume path_new"
                            else:
                                _escape_path1_yaw = float(np.arctan2(
                                    _p1_dx, _p1_dz))
                                _escape_phase = "path1_align"
                                _escaping = True
                                print(f"[INTRUSION] phase=path0 "
                                      f"dist={_p0_dist:.3f}m "
                                      f"progress={_line_progress:.3f}m "
                                      f"cross={_line_cross:.3f}m "
                                      f"clearance={_esc_clearance:.3f}m "
                                      f"path1_yaw={np.degrees(_escape_path1_yaw):.1f}deg "
                                      f"action=align_path1")
                                last_cmd = "intrusion path0 reached; align path1"
                        elif _line_reached:
                            motion_thread.stop()
                            path = None
                            path_idx = 0
                            _escape_phase = "plan"
                            _escape_via_path0 = False
                            _escape_path0_drive_mode = None
                            _escape_path0_anchor = None
                            _escape_path0_path = None
                            _escape_path0_line = None
                            _escape_path0_aligned = False
                            _escape_path1_yaw = None
                            print(f"[INTRUSION] phase=path0 "
                                  f"progress={_line_progress:.3f}m "
                                  f"cross={_line_cross:.3f}m "
                                  f"clearance={_esc_clearance:.3f}m "
                                  f"reason=anchor_became_unsafe "
                                  f"action=replan_same_target")
                            last_cmd = "intrusion path0 unsafe; replan"
                        else:
                            _el, _ea, _alpha, _escape_path0_aligned = \
                                escape_to_path_start_velocity(
                                    cur_pose, _escape_path0_line,
                                    _line_remaining,
                                    _escape_path0_aligned)
                            if not halted:
                                motion_thread.set_velocity(_el, _ea)
                                motion_thread.start()
                            _escaping = True
                            _line_stage = ("straight" if _escape_path0_aligned
                                           else "align")
                            last_cmd = (
                                f"intrusion path0[{_escape_path0_drive_mode}/"
                                f"{_line_stage}] remain={_line_remaining:.2f}m "
                                f"cross={_line_cross:.2f}m")

                elif _escape_phase == "path1_align":
                    _path1_valid = (
                        path is not None and len(path) >= 2 and
                        _escape_path0_path is path and
                        _escape_path1_yaw is not None)
                    if not _path1_valid:
                        motion_thread.stop()
                        path = None
                        path_idx = 0
                        _escape_phase = "plan"
                        _escape_via_path0 = False
                        _escape_path0_drive_mode = None
                        _escape_path0_anchor = None
                        _escape_path0_path = None
                        _escape_path0_line = None
                        _escape_path0_aligned = False
                        _escape_path1_yaw = None
                        print("[INTRUSION] phase=path1_align "
                              "reason=path_replaced action=replan_same_target")
                        last_cmd = "intrusion path1 align invalid; replan"
                    else:
                        _el, _ea, _alpha, _p1_aligned = \
                            escape_align_to_yaw_velocity(
                                cur_pose, _escape_path1_yaw)
                        if _p1_aligned:
                            motion_thread.stop()
                            path_idx = 1
                            _resume_state = _escape_resume_state
                            _resume_target = _escape_target
                            _aligned_yaw = _escape_path1_yaw
                            _escaping = False
                            _reset_intrusion_escape(clear_path0=False)
                            state = _resume_state
                            yaw_smoother.reset()
                            out_smoother.reset()
                            print(f"[INTRUSION] phase=path1_align "
                                  f"target=({_resume_target[0]:.3f},"
                                  f"{_resume_target[1]:.3f}) "
                                  f"yaw={np.degrees(cur_pose[2]):.1f}deg "
                                  f"path1_yaw={np.degrees(_aligned_yaw):.1f}deg "
                                  f"error={np.degrees(_alpha):.1f}deg "
                                  f"action=resume_path path_idx={path_idx} "
                                  f"state={state.name}")
                            last_cmd = "intrusion path1 aligned; resume path_new"
                        else:
                            motion_thread.set_velocity(_el, _ea)
                            motion_thread.start()
                            _escaping = True
                            last_cmd = (
                                f"intrusion path1_align "
                                f"error={np.degrees(_alpha):.1f}deg")

                elif _escape_phase == "fallback_retreat":
                    if _robot_check.safe:
                        motion_thread.stop()
                        _escaping = False
                        _escape_dir = None
                        _escape_plan_fail_cnt = 0
                        if (_escape_replan_from_semantic and
                                _escape_resume_state in (
                                    NavState.FINAL_FOLLOW,
                                    NavState.FINAL_ADJUST,
                                    NavState.FINAL_PLAN)):
                            # Retreat only solved robot intrusion; the old
                            # navigation station remains invalid.
                            _mark_invalid_nav_replan("escape_final_nav_invalid")
                            final_goal_nav = None
                            path = None
                            path_idx = 0
                            _resume_state = _escape_resume_state
                            _reset_intrusion_escape(clear_path0=True)
                            state = NavState.FINAL_PLAN
                            last_cmd = "intrusion clear; reselect final nav goal"
                            print("[INTRUSION] phase=fallback_retreat "
                                  f"clearance={_esc_clearance:.3f}m "
                                  "action=final_plan_reselect")
                            continue
                        if (_escape_replan_from_semantic and
                                _escape_resume_state in (
                                    NavState.PATROL_FOLLOW,
                                    NavState.PATROL_PLAN)):
                            _mark_invalid_nav_replan("escape_patrol_nav_invalid")
                            patrol_goal_nav = None
                            patrol_goal_nav_target = None
                            path = None
                            path_idx = 0
                            _reset_intrusion_escape(clear_path0=True)
                            state = NavState.PATROL_PLAN
                            last_cmd = "intrusion clear; reselect patrol nav goal"
                            print("[INTRUSION] phase=fallback_retreat "
                                  f"clearance={_esc_clearance:.3f}m "
                                  "action=patrol_plan_reselect")
                            continue
                        _escape_phase = "plan"
                        print(f"[INTRUSION] phase=fallback_retreat "
                              f"clearance={_esc_clearance:.3f}m "
                              f"action=replan_same_target")
                        last_cmd = "intrusion fallback clear; replan"
                    else:
                        if _escape_dir is None:
                            _escape_dir = nav_path.find_escape_dir(
                                (cur_pose[0], cur_pose[1]),
                                _esc_tree, fixed_y=_fixed_y)
                        if _escape_dir is not None:
                            _el, _ea = nav_path.escape_velocity(
                                cur_pose, _escape_dir)
                            _drive = "obstacle_away"
                        else:
                            _el, _ea = -FOLLOW_LIN_MAX * 0.5, 0.0
                            _drive = "blind_reverse"
                        motion_thread.set_velocity(_el, _ea)
                        motion_thread.start()
                        _escaping = True
                        last_cmd = (f"intrusion fallback[{_drive}] "
                                    f"clearance={_esc_clearance:.2f}m")
                else:
                    motion_thread.stop()
                    _escape_phase = "plan"
                    last_cmd = "intrusion initialize planning"

        elif state == NavState.PATROL_PLAN:
            _prn_state(state)
            last_cmd = "idle (planning)"
            if cur_pose is not None and patrol_target is not None:
                path, _actual_goal = None, None
                _patrol_nav_invalid = (
                    patrol_goal_nav is not None and
                    not check_nav_point(
                        patrol_goal_nav, obstacle_snapshot).safe)
                if patrol_goal_nav is None or _patrol_nav_invalid:
                    if _patrol_nav_invalid:
                        _mark_invalid_nav_replan("patrol_nav_invalid")
                    _resolution = resolve_nav_goal(
                        patrol_target, (cur_pose[0], cur_pose[1]),
                        obstacle_snapshot)
                    path, _actual_goal, _selected = plan_goal_resolution(
                        (cur_pose[0], cur_pose[1]), _resolution,
                        obs_current, fixed_y=nav_y,
                        plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                       if _plan_pose is not None else None),
                        obstacle_snapshot=obstacle_snapshot)
                    patrol_goal_nav = _selected.goal
                elif patrol_goal_nav is not None:
                    path, _actual_goal = plan_path(
                        (cur_pose[0], cur_pose[1]), patrol_goal_nav,
                        obs_current, fixed_y=nav_y,
                        plan_start_2d=((_plan_pose[0], _plan_pose[1])
                                       if _plan_pose is not None else None),
                        obstacle_snapshot=obstacle_snapshot)
                patrol_goal_nav_target = patrol_target
                # 例: VLM 方向目标被障碍挡住, RRT 退到最近自由点(sub-opt), 但机器人
                # 当前位姿已在该自由点(或已走过) -> 朝它走已无意义, 改用 frontier 探索。
                _divert_to_frontier = False
                if _divert_to_frontier:
                    # 仍留在 PATROL_PLAN, 下一循环朝 frontier 重新规划
                    pass
                elif path is not None and len(path) > 0:
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
                    _arm_preturn(
                        NavState.PATROL_PLAN, cur_pose, mode, odom_now, obs_current)
                    state = NavState.PATROL_FOLLOW
                else:
                    # RRT 无解: 累计失败次数
                    _patrol_plan_fail_cnt += 1
                    if _patrol_plan_fail_cnt >= PLAN_FAIL_MAX:
                        # 连续多次 RRT 失败: 改用 frontier 探索, 不进 WAITING, 持续尝试
                        # 注意: 不能用 nav_path._last_plan_fail_t 做节流 -- plan_path 每次
                        # 失败都会重置它, nav_auto 读到的值恒 < 2.0, 会把本分支卡成死代码。
                        # 满 10 次立即尝试 frontier (不依赖 2s 节流); 触发后 _set_patrol_goal
                        # 内部清零 _patrol_plan_fail_cnt, 不会重复触发。
                        _ft, _nst = auto_trigger_patrol(
                            slam, cur_pose, nav_y, obs_current)
                        if _ft is not None:
                            print(f"[nav_2d][WARN] PATROL 规划不成功 (已连续 "
                                  f"{_patrol_plan_fail_cnt} 次 RRT 失败, 目标被阻挡/"
                                  f"不可达) -> 改用 frontier 探索 (不进 WAITING)")
                            _set_patrol_goal(_ft)   # 重置计数/状态, 下一循环朝 frontier 重规划
                            # path=None 已在 _set_patrol_goal 内设置; 留在 PATROL_PLAN
                        elif time.time() - _patrol_frontier_t >= 2.0:
                            # 无可用 frontier: nav_auto 自有节流, 避免每帧查内存; 不进 WAITING
                            _patrol_frontier_t = time.time()
                            print(f"[nav_2d][WARN] PATROL 规划不成功 (已连续 "
                                  f"{_patrol_plan_fail_cnt} 次 RRT 失败) 且无可用 "
                                  f"frontier -> 继续等待地图更新, 每 2s 重试 (不进 WAITING)")
                    elif time.time() - _patrol_retry_t >= 2.0:
                        _patrol_retry_t = time.time()
                        _dbg(f"[nav_2d] 路径规划失败 (第 {_patrol_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试 (最新地图)")

        elif state == NavState.PATROL_FOLLOW:
            _prn_state(state)
            if _run_preturn(cur_pose, odom_now, obs_current, nav_y, halted):
                pass
            elif cur_pose is None:
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
                # PATROL 必须先判定到达，再判断是否需要脱困。否则导航站位位于
                # 净空边缘时，会先进入 ESCAPE，脱困后又重复规划同一 PATROL。
                _arrive_eps = (FRONTIER_ARRIVE_EPS if patrol_source == "frontier"
                               else PATROL_ARRIVE_EPS)
                if (d_tgt <= _arrive_eps or _de <= _arrive_eps or
                        path_idx >= len(path)):
                    # 暂时略过旧的 ``_sub_reached and not _tgt_reached``
                    # PATROL 保持原有到达语义：到达安全导航站位
                    # 即完成本轮巡逻，不执行 FINAL 专用的 GOAL 朝向对准。
                    motion_thread.stop()
                    _dbg(f"[nav_2d] PATROL_FOLLOW -> VLM_INQUIRY "
                          f"(到达巡逻导航站位, 距 nav={d_tgt:.2f}m / "
                          f"距路径终点={_de:.2f}m "
                          f"<= {_arrive_eps:.2f}m)")
                    state = NavState.VLM_INQUIRY
                    _auto_vlm_asked = False
                    _vlm_reask_pending = False
                else:
                    _robot_check = check_nav_point(
                        (cur_pose[0], cur_pose[1]), obstacle_snapshot,
                        clearance=ROBOT_RADIUS)
                    d_robot = _robot_check.clearance
                    if not _robot_check.safe:
                        if _begin_intrusion_escape(
                                NavState.PATROL_FOLLOW, _nav, d_robot):
                            continue

                    # frontier 目标失效: 未到达时若已被 walked memory 覆盖，
                    # 放弃旧目标并重新询问。
                    if (patrol_source == "frontier" and
                            patrol_target is not None and
                            _goal_walked(
                                slam.memory, float(patrol_target[0]),
                                float(patrol_target[1]), nav_y)):
                        _dbg(f"[nav_2d][FRONTIER-STALE] 巡逻 frontier 点 "
                             f"{patrol_target} 已被走过 -> VLM_INQUIRY")
                        patrol_target = None
                        patrol_source = None
                        patrol_goal_nav = None
                        patrol_goal_nav_target = None
                        motion_thread.stop()
                        state = NavState.VLM_INQUIRY
                        _auto_vlm_asked = False
                        _vlm_reask_pending = False
                        continue

                    # 已自由 -> 复位脱困方向 (每帧重算, 无需缓存旧点)
                    if _robot_check.safe:
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

                    _path_check = runtime.validate_remaining_path(
                        path, path_idx, (cur_pose[0], cur_pose[1]),
                        obstacle_snapshot)
                    if not _path_check.safe:
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _mark_invalid_nav_replan(
                                f"patrol_path_invalid:{_path_check.reason}")
                            if (patrol_goal_nav is not None and
                                    not check_nav_point(
                                        patrol_goal_nav,
                                        obstacle_snapshot).safe):
                                patrol_goal_nav = None
                            _dbg(f"[nav_2d] 路径失效 reason={_path_check.reason} "
                                 "-> PATROL_PLAN (同语义目标重规划)")
                            nav_path._last_replan_t = now_t
                            state = NavState.PATROL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                    else:
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
                if time.time() < _final_plan_retry_t:
                    motion_thread.stop()
                    last_cmd = "final plan retry cooldown"
                    continue
                path, _actual_goal = _final_plan_goal(
                    cur_pose, obs_current, nav_y)
                if path is not None and len(path) > 0:
                    _final_plan_fail_cnt = 0  # 成功规划, 清零失败计数
                    _final_plan_retry_t = 0.0
                    path_idx = 0
                    _dbg(f"[nav_2d] FINAL_PLAN -> FINAL_FOLLOW "
                          f"({len(path)} waypoints, "
                          f"nav_goal=({final_goal_nav[0]:.2f},"
                          f"{final_goal_nav[1]:.2f}))")
                    yaw_smoother.reset()
                    out_smoother.reset()
                    _arm_preturn(
                        NavState.FINAL_PLAN, cur_pose, mode, odom_now, obs_current)
                    state = NavState.FINAL_FOLLOW
                else:
                    _final_plan_fail_cnt += 1
                    if _final_plan_fail_cnt >= PLAN_FAIL_MAX:
                        _final_plan_fail_cnt = 0
                        _mark_invalid_nav_replan("final_rrt_failed")
                        final_goal_nav = None
                        print("[nav_2d][WARN] FINAL 路径连续失败，"
                              "清除导航站位并等待地图更新后重新解析")
                        _final_plan_retry_t = time.time() + 2.0
                    elif time.time() - nav_path._last_plan_fail_t >= 2.0:
                        _dbg(f"[nav_2d] 最终路径规划失败 (第 {_final_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试 (最新地图)")
                        nav_path._last_plan_fail_t = time.time()
                        _final_plan_retry_t = time.time() + 2.0

        elif state in (NavState.FINAL_FOLLOW, NavState.FINAL_ADJUST):
            _prn_state(state)
            _is_adjust = (state == NavState.FINAL_ADJUST)
            if _run_preturn(cur_pose, odom_now, obs_current, nav_y, halted):
                pass
            elif cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_2d] pose 丢失, 停车等待")
            elif path is None:
                # FINAL_ADJUST 首帧 (检测命中时 path 被清空) / FINAL_FOLLOW 路径被清空:
                # 立即规划一次到当前 final_target, 之后走下面的跟随逻辑。
                if final_target is not None:
                    obs = obs_current
                    path, _actual_goal = _final_plan_goal(
                        cur_pose, obs_current, nav_y)
                    if path is not None and len(path) > 0:
                        _final_adj_plan_fail_cnt = 0
                        path_idx = 0
                        if len(path) > 1:
                            _p0_dist = float(np.hypot(
                                float(path[0][0]) - float(cur_pose[0]),
                                float(path[0][1]) - float(cur_pose[1])))
                            if _p0_dist < FOLLOW_WAYPOINT_THRESHOLD:
                                # 新规划的 path[0] 通常就是当前位姿。若留给
                                # follow_path_step 处理，它会推进索引同时返回
                                # (0,0)，造成 FINAL_ADJUST 每次更新路径时顿一下。
                                path_idx = 1
                                _dbg(f"[nav_2d] FINAL path[0] 已在 waypoint "
                                     f"阈值内 ({_p0_dist:.3f}m < "
                                     f"{FOLLOW_WAYPOINT_THRESHOLD:.3f}m) "
                                     f"-> 直接从 path[1] 跟随")
                        yaw_smoother.reset()
                        out_smoother.reset()
                        _arm_preturn(state, cur_pose, mode, odom_now, obs_current)
                        _dbg(f"[nav_2d] {('FINAL_ADJUST' if _is_adjust else 'FINAL_FOLLOW')} "
                             f"首规划 ({len(path)} wp, nav=({final_goal_nav[0]:.2f},"
                             f"{final_goal_nav[1]:.2f}))")
                    else:
                        # RRT 无解: 累计失败, 满阈值转 FINAL_PLAN 走'重定位到最近可达'逻辑
                        _final_adj_plan_fail_cnt += 1
                        if _final_adj_plan_fail_cnt >= PLAN_FAIL_MAX:
                            _final_adj_plan_fail_cnt = 0
                            _mark_invalid_nav_replan(
                                "final_adjust_rrt_failed")
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
                _robot_check = check_nav_point(
                    (cur_pose[0], cur_pose[1]), obstacle_snapshot,
                    clearance=ROBOT_RADIUS)
                d_robot = _robot_check.clearance
                if not _robot_check.safe:
                    if _begin_intrusion_escape(state, _nav, d_robot):
                        continue
                if (d_tgt <= PATROL_ARRIVE_EPS or
                        _de <= PATROL_ARRIVE_EPS):
                    motion_thread.stop()
                    if _is_adjust:
                        if not _lock_final_adjust_if_ready(d_tgt):
                            last_cmd = "hold (adjust, at GOAL_NAV)"
                            _dbg(f"[nav_2d] FINAL_ADJUST 已到 GOAL_NAV "
                                 f"(距 nav={d_tgt:.2f}m), "
                                 "原地保持继续微调")
                    else:
                        _dbg(f"[nav_2d] FINAL_FOLLOW -> DONE "
                             f"(到达 GOAL_NAV, "
                             f"距 nav={d_tgt:.2f}m / "
                             f"距路径终点={_de:.2f}m "
                             f"<= {PATROL_ARRIVE_EPS:.2f}m)")
                        _invalidate_vlm("done")
                        state = NavState.DONE
                else:
                    # 已自由 -> 复位脱困方向 (每帧重算, 无需缓存旧点)
                    if _robot_check.safe:
                        _escape_dir = None
                    _path_check = runtime.validate_remaining_path(
                        path, path_idx, (cur_pose[0], cur_pose[1]),
                        obstacle_snapshot)
                    if not _path_check.safe:
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _mark_invalid_nav_replan(
                                f"final_path_invalid:{_path_check.reason}")
                            if (final_goal_nav is not None and
                                    not check_nav_point(
                                        final_goal_nav,
                                        obstacle_snapshot).safe):
                                final_goal_nav = None
                            if _is_adjust:
                                # FINAL_ADJUST: 原地重规划, 不离开终调状态
                                # (终调只由 时间/帧数/距离 三条件或 FORCE_TIME 退出)
                                _dbg(f"[nav_2d] FINAL_ADJUST 路径失效 "
                                     f"reason={_path_check.reason} -> "
                                     "原地重规划 (state 保持)")
                                path = None
                                nav_path._last_replan_t = now_t
                                continue  
                            else:
                                _dbg("[nav_2d] 最终路径被阻挡 -> FINAL_PLAN (重规划)")
                                nav_path._last_replan_t = now_t
                                state = NavState.FINAL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                    else:
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
                            if _is_adjust:
                                if not _lock_final_adjust_if_ready(d_tgt):
                                    last_cmd = "hold (adjust, at GOAL_NAV)"
                            else:
                                _dbg("[nav_2d] FINAL_FOLLOW -> DONE "
                                     "(路径完成并到达 GOAL_NAV)")
                                _invalidate_vlm("done")
                                state = NavState.DONE
                        elif _is_adjust:
                            # 终调阶段路径走完但未到目标: 由周期重规划重新连线, 不切状态
                            last_cmd = "adjust: path done, replan pending"
                            _dbg(f"[nav_2d] FINAL_ADJUST 路径走完未达目标 "
                                  f"(距 {d_tgt:.2f}m), 等周期重规划")
                        else:
                            _dbg(f"[nav_2d] 路径已走完但未到达 nav 终点 "
                                  f"(距 {d_tgt:.2f}m) -> FINAL_PLAN "
                                  f"(重规划)")
                            state = NavState.FINAL_PLAN

                    # 未到 GOAL_NAV 时仍保留原终调锁定逻辑；到达后由
                    # 上方分支执行同一 helper，不再要求朝向语义 GOAL。
                    if _is_adjust:
                        _lock_final_adjust_if_ready(d_tgt)

        _moving_states = (NavState.SCANNING,
                          NavState.ESCAPE,
                          NavState.PATROL_FOLLOW,
                          NavState.FINAL_FOLLOW,
                          NavState.FINAL_ADJUST)
        if ((state not in _moving_states or halted)
                and motion_thread.is_running()):
            motion_thread.stop()

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
        # 同时输出到本 episode console 日志，便于直接检索。
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
        if (_goal_src_changed or _patrol_src_changed or
                (_goal_tgt_changed and state != NavState.FINAL_ADJUST) or
                _nav_invalid_replan_pending):
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
                "from_invalid_replan": bool(_nav_invalid_replan_pending),
                "invalid_replan_reason": _nav_invalid_replan_reason,
            }, feature="state_events")
            _gs = goal_source if goal_source is not None else "none"
            _ps = patrol_source if patrol_source is not None else "none"
            _gt = (f"({_active_goal_tgt[0]:.2f},{_active_goal_tgt[1]:.2f})"
                   if _active_goal_tgt is not None else "none")
            _ir = (_nav_invalid_replan_reason
                   if _nav_invalid_replan_reason is not None else "none")
            print(f"[nav_2d] >>> GOAL source={_gs} patrol={_ps} target={_gt} "
                  f"state={state.name} "
                  f"from_invalid_replan={int(_nav_invalid_replan_pending)} "
                  f"invalid_reason={_ir}")
            _nav_invalid_replan_pending = False
            _nav_invalid_replan_reason = None
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
        runtime.transition(state)
        if (path and cur_pose is not None and
                not runtime.validate_remaining_path(
                    path, path_idx, (cur_pose[0], cur_pose[1]),
                    obstacle_snapshot).safe):
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
            _overlay_vlm = vlm_latest
            if (state == NavState.FINAL_FOLLOW and
                    _final_bbox_anchor is not None and
                    _bbox_calibration is not None and img is not None):
                _bbox_camera_pose = slam.get_pose_full()
                _projected_bbox = project_locked_target_bbox(
                    _final_bbox_anchor["target_world"],
                    _bbox_camera_pose,
                    img.shape[:2],
                    _bbox_calibration,
                    _final_bbox_anchor["bbox_px"],
                    _final_bbox_anchor["image_shape"],
                    _final_bbox_anchor["camera_pose"],
                )
                if _projected_bbox is not None:
                    _projected_det = dict(
                        _final_bbox_anchor["detection"])
                    _projected_det["bbox_2d"] = _projected_bbox
                    _overlay_vlm = ([_projected_det], None)
                else:
                    _overlay_vlm = (None, None)
            overlay_frame = draw_debug_overlay(img, {"lines": lines,
                                                       "rects": rects,
                                                       "robot_pose": cur_pose,
                                                       "using_odom": _use_odom,
                                                       "vlm_dets": _overlay_vlm[0],
                                                       "vlm_masks": _overlay_vlm[1]})
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
            memory_gaussians = [
                (float(r["mu_W"][0]), float(r["mu_W"][2]),
                 float(r.get("radius_W", 0.4)))
                for r in _mem.export_centers_W()
                if int(r.get("sid", 0)) == SID_WALKED
            ]
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
    _invalidate_vlm("exit")
    vlm_worker.stop(timeout=1.0)
    motion_thread.stop()
    estop_keyboard.stop()      # 恢复终端原始模式
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
