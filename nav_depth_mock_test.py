#!/usr/bin/env python
"""nav_depth_mock.py - 轻量 SLAM 导航 (odom + Astra depth) + Web Debug。

与 nav_mock.py 同结构, 但把 MASt3R-SLAM 这套"重"视觉 SLAM 换成两个
更轻量的系统:
  1) 轮式里程计 (OdomHolder, /odom)        -> 位姿 (死推算 nav pose)
  2) Astra 深度相机 (/camera_f/depth/...)   -> 障碍点云 (投影到 nav 世界系)
两者结合即"纯用 odom + astra depth 建图": 位姿来自 odom, 障碍来自 depth,
地图随 odom 死推算累积 (无回环)。

状态机 / 避障 / VLM / 手动 / Web debug 全部沿用 nav_mock 的方式。

坐标系 (与 nav_page / nav_path / MASt3R 一致):
  X = 右,  Y = 下 (高度向下为正),  Z = 前 (顶视屏下)
  robot_pose = (x, z, yaw); 前向向量 (sin yaw, cos yaw) -> (X, Z)
  首帧有效 odom 锚定为世界原点 (0,0,yaw=0); 之后相对锚点映到 nav 系。
  外参: 相机 ≡ 机器人原点。顶视是 ego-map (车永远画布中心)。

运行: 先起摄像头/底盘 ROS 节点, 再 python nav_mock_depth.py
调试: 浏览器 http://localhost:5001 (RGB + 顶视地图)
"""
import os
import sys
import time
import signal
import threading
import math
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ---- 轻量 SLAM 用到的本地模块 (与 nav_mock 完全一致) ----
from nav_control import (MockState, NavState, mock, MotionThread, KeyboardThread,
                         MOTION_HZ, SCAN_ANGULAR, RELOC_STOP_TIME, RELOC_BACK_TIME,
                         MANUAL_HOLD_TIMEOUT, _angle_diff)
import nav_path
from nav_path import (ROBOT_RADIUS, PLAN_BOX_PAD, REPLAN_PERIOD, PATROL_ARRIVE_EPS,
                      FOLLOW_LIN_MAX, FOLLOW_YAW_EMA, FOLLOW_OUT_EMA, FOLLOW_DEADBAND,
                      FOLLOW_STRAIGHT_ALPHA, FOLLOW_SLEW,
                      crop_to_box, plan_path, follow_path_step, is_path_blocked,
                      find_escape_point, escape_velocity)
from nav_rrt import build_obstacle_tree, tree_nearest_dist

import nav_page
from nav_page import (YawSmoother, OutputSmoother,
                     draw_debug_overlay, draw_map_view)

from nav_helpers import (key_to_bbox, run_vlm_inquiry, OdomHolder,
                         query_frontier_for_viz, _goal_walked)
from nav_vlm import VlmDetector, qwen_bbox_to_pixel
from nav_constants import (VLM_DETECT_TARGET_DEFAULT, VLM_URL_DEFAULT,
                           VLM_H_PROMPT, MOBILE_SAM_CHECKPOINT_PATH,
                           BOOT_SCAN_ENABLED)
from nav_memory import MemorySystem, SID_WALKED


# ============================================================
#  Astra depth 参数 (按实际硬件调整)
# ============================================================
DEPTH_TOPIC      = "/camera_f/depth/image_raw"      # 16UC1(mm) 或 32FC1(m)
DEPTH_INFO_TOPIC = "/camera_f/depth/camera_info"    # 内参; 缺失则用下方兜底
RGB_TOPIC        = "/camera_f/color/image_raw"
# 相机相对底盘的外参 (假设水平正对前方安装):
DEPTH_CAM_HEIGHT = 0.50   # 相机离地高度 (m); 障碍高度 = -y_c + H
DEPTH_CAM_YAW    = 0.0    # 相机相对底盘偏航 (rad); 0=正对前方
# 深度有效范围 / 障碍高度带:
DEPTH_MIN        = 0.20   # 有效深度下限 (m)
DEPTH_MAX        = 4.50   # 有效深度上限 (m)
OBS_HEIGHT_MIN   = 0.02   # 障碍高度下限 (m, 地面以上)
OBS_HEIGHT_MAX   = 0.55   # 障碍高度上限 (m)
DEPTH_STRIDE     = 2      # 投影采样步长 (像素); 越大点越稀、越快
OBS_VOXEL        = 0.05   # 障碍点云体素降采样尺寸 (m)
# 兜底内参 (masterslam/config/intrinsics.yaml; 若 camera_info 可用则被覆盖):
FB_FX, FB_FY, FB_CX, FB_CY = 517.3, 516.5, 318.6, 255.3
# odom 相对偏航 -> nav yaw 旋向; 实机左右反了则改 -1
NAV_YAW_SIGN = 1.0


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
        print(f"[nav_depth][STATE] {s.name}", flush=True)
        _printed_state = s
        _last_state_print_t = _now


_last_action_key = None


def log_action(tag, method, detail=""):
    global _last_action_key
    key = (tag, method)
    if key != _last_action_key:
        line = f"[nav_depth] >>> ACTION={tag}  METHOD={method}"
        if detail:
            line += f"  {detail}"
        print(line)
        _last_action_key = key


def _cmd_str(linear, angular):
    """速度 -> 可读动作描述。"""
    if abs(linear) < 0.01 and abs(angular) < 0.01:
        return "stop"
    if abs(angular) < 0.05:
        return f"forward({linear:.2f})"
    if abs(linear) < 0.05:
        d = "L" if angular > 0 else "R"
        return f"turn_{d}({abs(angular):.2f})"
    d = "L" if angular > 0 else "R"
    return f"arc(v={linear:.2f},{d}{abs(angular):.2f})"


# ============================================================
#  DepthSlam - 轻量 SLAM 封装 (odom 位姿 + Astra depth 障碍)
# ============================================================
class _TrackingMode:
    name = "TRACKING"


_TRACKING_MODE = _TrackingMode()


class _FakeKF:
    """假关键帧: 单位 Sim3, 使 MemorySystem 直接在 nav 世界系读写记忆。"""
    def __init__(self):
        # t(3) + q(4, 单位四元数 w=1) + s(1)
        self.T_WC = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                              dtype=np.float64)


def _wrap_pi(a):
    """角度归一化到 (-pi, pi]。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DepthSlam:
    """用 odom + Astra depth 实现 nav_mock 所需的 SLAM 接口子集。

    位姿: 首帧 odom 锚到 (0,0,0), 再映到 nav (X右, Z前, yaw)。
    障碍: depth 光学系 -> nav 世界系 [X右, Y下, Z前]。
    地图: 仅保留"当前帧"周围障碍 (死推算世界系), 不持久累积 (先看效果)。
    """

    def __init__(self, rgb_topic=RGB_TOPIC, depth_topic=DEPTH_TOPIC,
                 depth_info_topic=DEPTH_INFO_TOPIC, odom_holder=None):
        self._bridge = CvBridge()
        self.odom = odom_holder
        self.cam_height = DEPTH_CAM_HEIGHT
        self.cam_yaw = DEPTH_CAM_YAW
        self.fx, self.fy, self.cx, self.cy = FB_FX, FB_FY, FB_CX, FB_CY
        self._have_info = False
        # 首帧有效 odom (ox0, oy0, yaw0); 锁定后进程内不变
        self._odom_anchor = None

        from RGBRosConnector import RGBRosConnector
        self._rgb = RGBRosConnector(topic=rgb_topic)

        self._depth = None
        self._depth_shape = None
        self._lock = threading.Lock()
        self._cam_img_pts = None   # (H,W,3) 相机光学系点云 (无效处为 0)
        self._obstacles = np.empty((0, 3), dtype=np.float64)  # nav 世界系 [X,Y下,Z]

        rospy.Subscriber(depth_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(depth_topic, Image, self._depth_cb, queue_size=1)
        print(f"[nav_depth] 订阅深度: {depth_topic}  (内参: {depth_info_topic})")

        self.memory = None  # 由 main() 注入 MemorySystem

    # ---- callbacks ----
    def _info_cb(self, msg):
        if not self._have_info and msg.K:
            K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
            self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
            self._have_info = True
            print(f"[nav_depth] 深度内参来自 camera_info: "
                  f"fx={self.fx:.1f} fy={self.fy:.1f} "
                  f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def _depth_cb(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception:
            return
        if depth is None:
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float64) / 1000.0
        else:
            depth = depth.astype(np.float64)
        with self._lock:
            self._depth = depth
            self._depth_shape = depth.shape

    # ---- 投影 ----
    def _project_camera(self, depth):
        """depth(H,W, m) -> (H,W,3) 相机光学系点 (x右, y下, z前), 无效处 0。"""
        H, W = depth.shape
        idx = slice(0, None, DEPTH_STRIDE)
        us, vs = np.meshgrid(np.arange(W)[idx], np.arange(H)[idx])
        Z = depth[vs, us]
        mask = (Z > DEPTH_MIN) & (Z < DEPTH_MAX)
        us = us[mask].astype(np.float64)
        vs = vs[mask].astype(np.float64)
        Z = Z[mask]
        out = np.zeros((H, W, 3), dtype=np.float64)
        if us.size == 0:
            return out
        Xc = (us - self.cx) * Z / self.fx
        Yc = (vs - self.cy) * Z / self.fy
        out[vs.astype(np.int64), us.astype(np.int64)] = np.stack([Xc, Yc, Z], axis=1)
        return out

    def _voxel_downsample(self, pts, v):
        if len(pts) == 0:
            return pts
        keys = np.floor(pts / v).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[idx]

    def odom_to_nav(self, ox, oy, oyaw):
        """ROS odom (前x, 左y, yaw) -> nav (X右, Z前, yaw)。

        参数:
            ox, oy, oyaw: 当前 raw odom
        返回:
            (X, Z, psi); 首次调用锁定锚点使当前位姿为 (0,0,0)
        """
        if self._odom_anchor is None:
            self._odom_anchor = (float(ox), float(oy), float(oyaw))
            print(f"[nav_depth] odom 锚点锁定: "
                  f"({ox:.3f}, {oy:.3f}, yaw={oyaw:.3f}) -> nav (0,0,0)")
        ox0, oy0, psi0 = self._odom_anchor
        dx = float(ox) - ox0
        dy = float(oy) - oy0
        c0 = math.cos(psi0)
        s0 = math.sin(psi0)
        # 相对启动车体: 前 x_b / 左 y_b
        x_b = c0 * dx + s0 * dy
        y_b = -s0 * dx + c0 * dy
        psi_b = _wrap_pi(float(oyaw) - psi0)
        # body0 -> nav: X=右=-左, Z=前
        X = -y_b
        Z = x_b
        psi = NAV_YAW_SIGN * psi_b
        return (float(X), float(Z), float(psi))

    def _body_to_world(self, bf, bl, pose):
        """车体前/左 点云 -> nav 世界 XZ。

        参数:
            bf, bl: 前/左分量 (ndarray 或标量)
            pose: (x_r, z_r, psi) nav 位姿
        返回:
            (X, Z) 世界坐标
        """
        x_r, z_r, psi = pose
        # 先施加相机相对底盘 yaw
        cy_ = math.cos(self.cam_yaw)
        sy_ = math.sin(self.cam_yaw)
        bf2 = bf * cy_ - bl * sy_
        bl2 = bf * sy_ + bl * cy_
        s = math.sin(psi)
        c = math.cos(psi)
        # 前向 (sin, cos), 左向 (-cos, sin) -> (X, Z)
        X = x_r + bf2 * s - bl2 * c
        Z = z_r + bf2 * c + bl2 * s
        return X, Z

    def _update_obstacles(self):
        """depth -> 光学系 -> nav 世界系 [X右, Y下, Z前]。"""
        with self._lock:
            depth = self._depth
        if depth is None:
            return
        cam = self._project_camera(depth)   # (H,W,3) 光学系 Xc右 Yc下 Zc前
        self._cam_img_pts = cam
        valid = np.linalg.norm(cam, axis=2) > 1e-6
        if not valid.any():
            self._obstacles = np.empty((0, 3), dtype=np.float64)
            return
        pts_c = cam[valid]                   # (N,3)
        Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        # 物理"地面以上"高度 (水平相机); 过滤用, 存储仍用 Yc 向下
        height_up = self.cam_height - Yc
        hmask = (height_up >= OBS_HEIGHT_MIN) & (height_up <= OBS_HEIGHT_MAX)
        bf = Zc[hmask]                       # 前
        bl = -Xc[hmask]                      # 左
        Y_down = Yc[hmask]
        if bf.size == 0:
            self._obstacles = np.empty((0, 3), dtype=np.float64)
            return
        pose = self.get_pose()
        if pose is None:
            self._obstacles = np.empty((0, 3), dtype=np.float64)
            return
        wx, wz = self._body_to_world(bf, bl, pose)
        pts = np.stack([wx, Y_down, wz], axis=1)  # [X右, Y下, Z前]
        pts = self._voxel_downsample(pts, OBS_VOXEL)
        self._obstacles = pts

    # ---- SLAM 接口子集 ----
    def start(self):
        def _refresh():
            r = rospy.Rate(15)
            while not rospy.is_shutdown():
                self._update_obstacles()
                r.sleep()
        self._th = threading.Thread(target=_refresh, daemon=True)
        self._th.start()

    def stop(self):
        pass

    def shutdown(self):
        pass

    def get_img(self):
        return self._rgb.get_frame()

    def get_pose(self):
        """导航世界系位姿 (x=右, z=前, yaw); 首帧 odom 锚到 (0,0,0)。

        返回:
            (X, Z, yaw) 或 None (无 odom)
        """
        if self.odom is None:
            return None
        o = self.odom.get()
        if o is None:
            return None
        return self.odom_to_nav(float(o[0]), float(o[1]), float(o[2]))

    def get_pose_full(self):
        """3D 位姿 (X, Y=0, Z); 导航平面 Y 取 0。"""
        p = self.get_pose()
        if p is None:
            return None
        return (float(p[0]), 0.0, float(p[1]))

    def get_pose_keyframe(self):
        return self.get_pose()

    def get_pose_full_keyframe(self):
        return self.get_pose_full()

    def get_pointcloud_2d(self):
        return self._cam_img_pts

    def get_map(self):
        return (self._obstacles, None)

    def get_obstacles(self):
        return self._obstacles

    def get_mode(self):
        return _TRACKING_MODE

    def num_keyframes(self):
        return 1


def extract_target_3d_depth(slam, bbox, nav_y=None, max_dist=3.0):
    """从 bbox + 当前深度点云提取目标 3D 世界坐标 (nav 系)。

    参数:
        slam: DepthSlam
        bbox: (x1,y1,x2,y2) 像素
        nav_y: 保留接口; 过滤仍用物理高度带
        max_dist: 最大深度 (m)
    返回:
        (X右, Y下, Z前) 或 None
    """
    cam = slam.get_pointcloud_2d()
    img = slam.get_img()
    if cam is None or img is None:
        return None
    h_pc, w_pc = cam.shape[:2]
    h_img, w_img = img.shape[:2]
    sx = w_pc / w_img if w_img > 0 else 1.0
    sy = h_pc / h_img if h_img > 0 else 1.0
    x1 = int(max(0, min(bbox[0] * sx, w_pc - 1)))
    y1 = int(max(0, min(bbox[1] * sy, h_pc - 1)))
    x2 = int(max(0, min(bbox[2] * sx, w_pc - 1)))
    y2 = int(max(0, min(bbox[3] * sy, h_pc - 1)))
    region = cam[y1:y2 + 1, x1:x2 + 1].reshape(-1, 3)
    valid = np.linalg.norm(region, axis=1) > 1e-6
    pts = region[valid]
    if len(pts) == 0:
        return None
    Zc = pts[:, 2]
    height_up = slam.cam_height - pts[:, 1]
    mask = (Zc > 0.3) & (Zc < max_dist) & \
           (height_up >= OBS_HEIGHT_MIN) & (height_up <= OBS_HEIGHT_MAX)
    pts = pts[mask]
    if len(pts) == 0:
        return None
    med = np.median(pts, axis=0)     # Xc, Yc, Zc
    pose = slam.get_pose()
    if pose is None:
        return None
    Xc, Yc, Zc_ = float(med[0]), float(med[1]), float(med[2])
    bf = Zc_
    bl = -Xc
    wx, wz = slam._body_to_world(bf, bl, pose)
    return (float(wx), float(Yc), float(wz))


# ============================================================
#  主状态机
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="nav_depth_mock 轻量 SLAM 导航 (odom+depth)")
    ap.add_argument("--vlm-url", default=VLM_URL_DEFAULT,
                   help="Qwen VLM 服务 (OpenAI 兼容), 默认 keyboard_qwen 地址")
    ap.add_argument("--depth-topic", default=DEPTH_TOPIC,
                   help="Astra 深度话题 (默认 /camera_f/depth/image_raw)")
    ap.add_argument("--rgb-topic", default=RGB_TOPIC,
                   help="RGB 话题 (默认 /camera_f/color/image_raw)")
    ap.add_argument("--depth-info-topic", default=DEPTH_INFO_TOPIC,
                   help="深度 camera_info 话题")
    args = ap.parse_args()

    rospy.init_node("nav_depth_mock", anonymous=True)

    odom_holder = OdomHolder()
    rospy.Subscriber("/odom", Odometry, odom_holder.cb, queue_size=1)
    print("[nav_depth] Subscribed to /odom (位姿来源)")

    slam = DepthSlam(rgb_topic=args.rgb_topic,
                     depth_topic=args.depth_topic,
                     depth_info_topic=args.depth_info_topic,
                     odom_holder=odom_holder)
    slam.start()

    # ---- walked 记忆 (用假 keyframe 直接在 nav 世界系读写) ----
    mem = MemorySystem()
    mem.bind_keyframes([_FakeKF()])
    mem.set_write_enabled(True)
    slam.memory = mem
    print(f"[nav_depth] walked 记忆已启用 enable={mem.enable}")

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tracer_ros", "tracer_http_interface", "scripts"))
    from tracer_http_interface.scripts.rw_api import TracerRobot
    bot = TracerRobot(base_url="http://localhost:8080")

    vlm_detector = VlmDetector(vlm_url=args.vlm_url,
                               sam_ckpt=MOBILE_SAM_CHECKPOINT_PATH)
    print(f"[nav_depth] VLM 检测已启用: {args.vlm_url}")

    nav_page.mock = mock
    nav_page.start_flask(port=5001)

    keyboard = KeyboardThread(manual_lin=0.10, manual_ang=0.30)
    keyboard.start()

    _nav_stop = threading.Event()

    def _on_sigint(signum, frame):
        print("\n[nav_depth] 收到 Ctrl+C, 正在退出...")
        _nav_stop.set()
        keyboard._running = False
        keyboard._restore_term()
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
        threading.Timer(0.5, _force_exit).start()

    signal.signal(signal.SIGINT, _on_sigint)

    state = NavState.WAITING
    _boot_scan_done = False
    rate = rospy.Rate(10)

    path = None
    path_idx = 0
    patrol_target = None
    final_target = None
    patrol_goal_nav = None
    patrol_goal_nav_target = None
    final_goal_nav = None
    d_tgt = None
    d_robot = None
    nav_y = 0.0
    _patrol_plan_fail_cnt = 0
    _final_plan_fail_cnt = 0
    scan_accumulated = 0.0
    scan_prev_yaw = None

    step = 0
    prev_t = time.time()
    fps = 0.0
    _nav_step_t = time.time()
    _nav_grab_ms = 0.0
    _nav_draw_ms = 0.0
    last_cmd = "idle"
    vlm_latest = (None, None)
    prev_halt = False
    prev_manual_on = False
    motion_thread = MotionThread(bot, hz=MOTION_HZ)
    keyboard.motion_thread = motion_thread

    _escaping = False
    _escape_point = None
    _yaw_sign = -1.0
    _vlog_t = 0.0

    yaw_smoother = YawSmoother(alpha=FOLLOW_YAW_EMA)
    out_smoother = OutputSmoother(alpha=FOLLOW_OUT_EMA,
                                  deadband=FOLLOW_DEADBAND,
                                  slew=FOLLOW_SLEW)
    _last_action_key = None

    print("[nav_depth] 启动状态机导航, 等待 odom + depth 就绪...")
    print("[nav_depth] 打开浏览器查看 debug "
          "(本机 http://localhost:5001, 其他设备用本机 IP:5001)")

    lookahead_target = None

    while (not rospy.is_shutdown()) and (not _nav_stop.is_set()):
        _tg0 = time.time()
        img = slam.get_img()
        raw_pose = slam.get_pose()
        mode = slam.get_mode()
        cur_pose = raw_pose
        _nav_grab_ms = (time.time() - _tg0) * 1000.0

        _use_odom = True
        _nav_source = "odom"

        if mem is not None and cur_pose is not None:
            try:
                mem.maybe_write_walked(
                    np.array([cur_pose[0], 0.0, cur_pose[1]],
                             dtype=np.float32), time.time())
            except Exception:
                pass

        obs_current = slam.get_obstacles()
        n_map = len(obs_current) if obs_current is not None else 0
        _map_pts = obs_current

        halted = mock.get_halt()
        if halted and not prev_halt:
            bot.stop()
            motion_thread.stop()
            log_action("ESTOP", "stop", "急停上升沿")
            prev_halt = True
        elif not halted and prev_halt:
            prev_halt = False
        if halted:
            last_cmd = "ESTOP (halt)"

        manual_on = mock.get_manual_mode()

        if mock.peek_frontier_request():
            mock.take_frontier_request()
            if cur_pose is None:
                print("[nav_depth][FRONTIER] 收到 f 请求, 但 pose 未就绪, 忽略")
            elif mem is None:
                print("[nav_depth][FRONTIER] 收到 f 请求, 但 MemorySystem 未就绪, 忽略")
            else:
                _fr = query_frontier_for_viz(
                    mem,
                    pose_xz=(cur_pose[0], cur_pose[1]),
                    nav_y=nav_y if nav_y is not None else 0.0,
                    obstacle_points=obs_current,
                    robot_radius=ROBOT_RADIUS,
                    yaw=float(cur_pose[2]),
                )
                _fxz = _fr.get("frontier_xz")
                if _fxz is not None:
                    _fx, _fz = float(_fxz[0]), float(_fxz[1])
                    patrol_target = (_fx, _fz)
                    _patrol_plan_fail_cnt = 0
                    patrol_goal_nav = None
                    patrol_goal_nav_target = None
                    path = None
                    path_idx = 0
                    print(f"[nav_depth][FRONTIER] 最近 frontier "
                          f"({patrol_target[0]:.2f},{patrol_target[1]:.2f}) "
                          f"dist={_fr.get('frontier_dist_m')}m -> PATROL_PLAN")
                    state = NavState.PATROL_PLAN
                else:
                    print("[nav_depth][FRONTIER] 未找到可用 frontier, 忽略")

        _det_key = mock.take_det_key()
        if _det_key == "o" and img is not None:
            _det_tgt = (mock.get_detect_target() or VLM_DETECT_TARGET_DEFAULT)
            print(f"[nav_depth][DET] 触发 VLM 检测: '{_det_tgt}'")
            _vres = vlm_detector.detect(img, _det_tgt)
            _vdets, _vmasks, _vtgt = _vres
            if _vdets == "error":
                print(f"\033[91m[nav_depth][DET] VLM 错误: {_vmasks}\033[0m")
                vlm_latest = (None, None)
            elif _vdets:
                _vb = _vdets[0].get("bbox_2d")
                if _vb and len(_vb) == 4 and cur_pose is not None:
                    h, w = img.shape[:2]
                    _x1, _y1, _x2, _y2 = qwen_bbox_to_pixel(_vb, w, h)
                    _t3d = extract_target_3d_depth(slam, (_x1, _y1, _x2, _y2),
                                                   nav_y=nav_y)
                    if _t3d is not None:
                        final_target = (_t3d[0], _t3d[2])
                        _final_plan_fail_cnt = 0
                        print(f"[nav_depth][DET] VLM 目标 "
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
                        print("[nav_depth][DET] bbox->3D 失败, 等待")
                        vlm_latest = (None, None)
                else:
                    print("[nav_depth][DET] VLM 未返回有效 bbox")
                    vlm_latest = (None, None)
            else:
                print("[nav_depth][DET] VLM 未检测到目标 "
                      f"(target='{_vtgt}')")
                vlm_latest = (None, None)

        _h_key = mock.take_h_key()
        if _h_key == "h" and img is not None:
            print(f"[nav_depth][H] 触发 VLM 问答: prompt='{VLM_H_PROMPT}'")
            _ans = vlm_detector.ask(img, VLM_H_PROMPT)
            mock.set_vlm_h_answer(_ans)
            print(f"[nav_depth][H] VLM 回答: {_ans}")

        if state == NavState.WAITING:
            _prn_state(state)
            last_cmd = "idle (waiting)"
            if not _boot_scan_done and cur_pose is not None:
                if BOOT_SCAN_ENABLED:
                    scan_accumulated = 0.0
                    scan_prev_yaw = None
                    _dbg("[nav_depth] WAITING -> SCANNING (开局先扫描一圈建图)")
                    state = NavState.SCANNING
                    _boot_scan_done = True
                else:
                    _dbg("[nav_depth] WAITING -> VLM_INQUIRY (跳过开局扫描)")
                    state = NavState.VLM_INQUIRY
                    _boot_scan_done = True

        elif state == NavState.SCANNING:
            _prn_state(state)
            if cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_depth] SCANNING: pose 丢失, 停车等待")
            else:
                if not halted:
                    motion_thread.set_velocity(0.0, SCAN_ANGULAR)
                    motion_thread.start()
                    log_action("SCAN", "MotionThread(continuous cmd_vel)",
                               f"angular={SCAN_ANGULAR:.2f}")
                if scan_prev_yaw is not None:
                    d = cur_pose[2] - scan_prev_yaw
                    d = (d + np.pi) % (2 * np.pi) - np.pi
                    scan_accumulated += abs(d)
                scan_prev_yaw = cur_pose[2]

                if scan_accumulated >= 2 * np.pi:
                    motion_thread.stop()
                    _dbg(f"[nav_depth] SCANNING 完成 "
                         f"(累计 {np.degrees(scan_accumulated):.0f}°)")
                    if patrol_target is not None:
                        patrol_goal_nav = None
                        patrol_goal_nav_target = None
                        path = None
                        path_idx = 0
                        _patrol_plan_fail_cnt = 0
                        state = NavState.PATROL_PLAN
                        _dbg("[nav_depth] SCANNING 完成 -> 恢复 PATROL_PLAN")
                    elif final_target is not None:
                        final_goal_nav = None
                        path = None
                        path_idx = 0
                        _final_plan_fail_cnt = 0
                        state = NavState.FINAL_PLAN
                        _dbg("[nav_depth] SCANNING 完成 -> 恢复 FINAL_PLAN")
                    else:
                        state = NavState.VLM_INQUIRY
                        _dbg("[nav_depth] SCANNING 完成 -> VLM_INQUIRY")
                else:
                    last_cmd = (f"scan "
                                f"({np.degrees(scan_accumulated):.0f}/360°)")

        elif state == NavState.VLM_INQUIRY:
            _prn_state(state)
            last_cmd = "idle (vlm wait)"
            vlm_key = mock.peek_vlm_key()
            if img is None or cur_pose is None:
                if vlm_key is not None:
                    print(f"[nav_depth] VLM_INQUIRY: 方向键 '{vlm_key}' 已收到，"
                          f"但 image/pose 未就绪，等待")
                    last_cmd = "idle (vlm wait: no image/pose)"
            elif vlm_key is not None:
                print(f"[nav_depth] VLM_INQUIRY: 收到方向键 '{vlm_key}', 开始检测目标...")
                bbox = run_vlm_inquiry(img, cur_pose)
                if bbox is None:
                    if vlm_key is not None:
                        print(f"[nav_depth] VLM_INQUIRY: 方向键 '{vlm_key}' 未找到目标")
                        last_cmd = f"idle (vlm: no target for '{vlm_key}')"
                else:
                    t3d = extract_target_3d_depth(slam, bbox, nav_y=nav_y)
                    if t3d is not None:
                        _tx, _tz = float(t3d[0]), float(t3d[2])
                        if _goal_walked(mem, _tx, _tz, nav_y):
                            print(f"[nav_depth] VLM 目标 ({_tx:.2f},{_tz:.2f}) "
                                  f"已在 walked 记忆内(走过), 跳过")
                        else:
                            patrol_target = (_tx, _tz)
                            _patrol_plan_fail_cnt = 0
                            if (patrol_goal_nav is not None and
                                    patrol_target != patrol_goal_nav_target):
                                _dbg(f"[nav_depth] patrol 目标已更新, "
                                     f"旧 sub-opt-goal {patrol_goal_nav} 失效 "
                                     f"-> PATROL_PLAN")
                                patrol_goal_nav = None
                                patrol_goal_nav_target = None
                                state = NavState.PATROL_PLAN
                            else:
                                _nav = (patrol_goal_nav if patrol_goal_nav
                                        is not None else patrol_target)
                                d_tgt = ((_nav[0] - cur_pose[0]) ** 2 +
                                         (_nav[1] - cur_pose[1]) ** 2) ** 0.5
                                if d_tgt < PATROL_ARRIVE_EPS:
                                    last_cmd = "idle (at patrol)"
                                    _dbg(f"[nav_depth] VLM 已在导航终点附近 "
                                         f"({d_tgt:.2f}m < {PATROL_ARRIVE_EPS:.2f}m)")
                                else:
                                    patrol_goal_nav = None
                                    patrol_goal_nav_target = None
                                    _dbg(f"[nav_depth] VLM 目标 "
                                         f"3D=({t3d[0]:.2f},{t3d[1]:.2f},"
                                         f"{t3d[2]:.2f}) -> 2D=({patrol_target[0]:.2f},"
                                         f"{patrol_target[1]:.2f})")
                                    _dbg("[nav_depth] VLM_INQUIRY -> PATROL_PLAN")
                                    state = NavState.PATROL_PLAN
                    else:
                        if vlm_key is not None:
                            print(f"[nav_depth] VLM_INQUIRY: 方向键 '{vlm_key}' "
                                  f"未找到目标 (框内无法投影到 3D: 无深度)")
                            last_cmd = f"idle (vlm: no 3D for '{vlm_key}')"
                        _dbg("[nav_depth] VLM bbox -> 3D 失败, 等待")

        elif state == NavState.PATROL_PLAN:
            _prn_state(state)
            last_cmd = "idle (planning)"
            if cur_pose is not None and patrol_target is not None:
                path, patrol_goal_nav = plan_path(
                    (cur_pose[0], cur_pose[1]),
                    patrol_target, obs_current, fixed_y=nav_y,
                    plan_start_2d=((cur_pose[0], cur_pose[1])))
                patrol_goal_nav_target = patrol_target
                if path is not None and len(path) > 0:
                    _patrol_plan_fail_cnt = 0
                    path_idx = 0
                    _dbg(f"[nav_depth] PATROL_PLAN -> PATROL_FOLLOW "
                         f"({len(path)} waypoints)")
                    yaw_smoother.reset()
                    out_smoother.reset()
                    state = NavState.PATROL_FOLLOW
                else:
                    _patrol_plan_fail_cnt += 1
                    if _patrol_plan_fail_cnt >= PLAN_FAIL_MAX:
                        print(f"[nav_depth][ERROR] PATROL 规划不成功 "
                              f"(已连续 {_patrol_plan_fail_cnt} 次 RRT 规划失败) "
                              f"-> 放弃目标, 回到 WAITING")
                        _patrol_plan_fail_cnt = 0
                        patrol_target = None
                        patrol_goal_nav = None
                        patrol_goal_nav_target = None
                        state = NavState.WAITING
                    elif time.time() - nav_path._last_plan_fail_t >= 2.0:
                        _dbg(f"[nav_depth] 路径规划失败 (第 {_patrol_plan_fail_cnt}/"
                             f"{PLAN_FAIL_MAX} 次 RRT), 等待重试")
                        nav_path._last_plan_fail_t = time.time()

        elif state == NavState.PATROL_FOLLOW:
            _prn_state(state)
            if cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_depth] pose 丢失, 停车等待")
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
                bo_tree = build_obstacle_tree(
                    crop_to_box(obs_global, (cur_pose[0], cur_pose[1]),
                                patrol_target, PLAN_BOX_PAD), nav_y)
                d_robot = tree_nearest_dist(bo_tree, nav_y,
                                            cur_pose[0], cur_pose[1])
                if (d_robot >= ROBOT_RADIUS and
                        (d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS
                         or path_idx >= len(path))):
                    motion_thread.stop()
                    _dbg(f"[nav_depth] PATROL_FOLLOW -> VLM_INQUIRY (到达巡逻点)")
                    state = NavState.VLM_INQUIRY
                else:
                    if d_robot >= ROBOT_RADIUS:
                        _escape_point = None
                    if _escape_point is None:
                        now_t = time.time()
                        if now_t - _vlog_t >= 0.5:
                            _dbg(f"[nav_depth]   [DEBUG] 未到达 "
                                 f"nav={d_tgt:.2f}m path_end={_de:.2f}m "
                                 f"path_idx={path_idx}/{len(path)} "
                                 f"cur_pose=({cur_pose[0]:.2f},{cur_pose[1]:.2f}) "
                                 f"d_robot={d_robot:.2f}m")
                        now_t = time.time()
                        if now_t - nav_path._last_periodic_replan_t >= REPLAN_PERIOD:
                            nav_path._last_periodic_replan_t = now_t
                            new_path, new_nav = plan_path(
                                (cur_pose[0], cur_pose[1]),
                                patrol_target, obs_global, fixed_y=nav_y,
                                plan_start_2d=((cur_pose[0], cur_pose[1])))
                        else:
                            pass
                        if new_path is not None and len(new_path) >= 2:
                            path = new_path
                            patrol_goal_nav = new_nav
                            patrol_goal_nav_target = patrol_target
                        elif new_nav is not None:
                            if _goal_walked(mem, float(new_nav[0]),
                                            float(new_nav[1]), nav_y):
                                print(f"[nav_depth][DBG-REPLAN] sub-opt-goal="
                                      f"{new_nav} 已在 walked 记忆内, 跳过该绕行")
                            else:
                                patrol_goal_nav = new_nav
                                patrol_goal_nav_target = patrol_target
                                _dbg(f"[nav_depth][DBG-REPLAN] RRT 暂无解, "
                                     f"仅刷新 sub-opt-goal={new_nav}")
                            _best_i = 0
                            _best_d = float('inf')
                            for _pi in range(len(path)):
                                _pd = ((path[_pi][0] - cur_pose[0]) ** 2 +
                                       (path[_pi][1] - cur_pose[1]) ** 2) ** 0.5
                                if _pd < 0.3 and _pi + 1 < len(path):
                                    _best_i = _pi + 1
                                    _best_d = _pd
                                elif _pd < _best_d and _pi >= _best_i:
                                    _best_d = _pd
                                    _best_i = _pi
                            path_idx = max(0, min(_best_i, len(path) - 1))
                        else:
                            pass

                    if d_robot < ROBOT_RADIUS:
                        if (_escape_point is not None and
                                tree_nearest_dist(bo_tree, nav_y,
                                    float(_escape_point[0]),
                                    float(_escape_point[1])) < ROBOT_RADIUS):
                            _escape_point = None
                        if _escape_point is None:
                            _escape_point = nav_path.find_escape_point(
                                (cur_pose[0], cur_pose[1]), obs_global,
                                fixed_y=nav_y)
                        if time.time() - nav_path._last_intrusion_warn_t >= 2.0:
                            if _escape_point is not None:
                                _dbg(f"[nav_depth] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"RRT 脱困点 E={_escape_point}, 朝其移动")
                            else:
                                _dbg(f"[nav_depth] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"无自由脱困点, 兜底直线后退")
                            nav_path._last_intrusion_warn_t = time.time()
                        if _escape_point is not None:
                            _el, _ea = nav_path.escape_velocity(
                                cur_pose, _escape_point)
                            if not halted:
                                motion_thread.set_velocity(_el, _ea)
                                motion_thread.start()
                            last_cmd = f"escape->E (near obs {d_robot:.2f}m)"
                        else:
                            if not halted:
                                motion_thread.set_velocity(
                                    -FOLLOW_LIN_MAX * 0.5, 0.0)
                                motion_thread.start()
                            last_cmd = f"escape blind (near obs {d_robot:.2f}m)"
                        _escaping = True
                    elif is_path_blocked(path, path_idx, obs_global,
                                         fixed_y=nav_y, tree=bo_tree):
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _dbg("[nav_depth] 路径被阻挡 -> PATROL_PLAN (重规划)")
                            nav_path._last_replan_t = now_t
                            state = NavState.PATROL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                    else:
                        sm_yaw = yaw_smoother.update(cur_pose[2])
                        sm_pose = (cur_pose[0], cur_pose[1], sm_yaw)
                        cmd, path_idx, lookahead_target, alpha = follow_path_step(
                            path, path_idx, sm_pose)
                        _db = 0.0 if abs(alpha) > FOLLOW_STRAIGHT_ALPHA \
                            else FOLLOW_DEADBAND
                        cmd = (cmd[0], out_smoother.update(cmd[1], deadband=_db))
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
                                _dbg(f"[nav_depth]   FOLLOW cmd "
                                     f"lin={cmd[0]:.2f} ang={cmd[1]:.2f} "
                                     f"wp={path_idx}/{len(path)} "
                                     f"cur_pose=({cur_pose[0]:.2f},"
                                     f"{cur_pose[1]:.2f}) "
                                     f"dist_wp={_d_wp:.2f}m d_tgt={d_tgt:.2f}m")
                                _vlog_t = now_t
                            last_cmd = _cmd_str(cmd[0], cmd[1])
                        if path_idx >= len(path):
                            motion_thread.stop()
                            if d_tgt <= PATROL_ARRIVE_EPS:
                                _dbg("[nav_depth] PATROL_FOLLOW -> VLM_INQUIRY "
                                     "(到达导航终点, 完成寻路)")
                                state = NavState.VLM_INQUIRY
                            else:
                                _dbg(f"[nav_depth] 路径已走完但未到达 nav 终点 "
                                     f"(距 {d_tgt:.2f}m) -> PATROL_PLAN (重规划)")
                                state = NavState.PATROL_PLAN
            else:
                pass

        elif state == NavState.FINAL_PLAN:
            _prn_state(state)
            last_cmd = "idle (final plan)"
            if cur_pose is not None and final_target is not None:
                path, final_goal_nav = plan_path(
                    (cur_pose[0], cur_pose[1]),
                    final_target, obs_current, fixed_y=nav_y,
                    plan_start_2d=((cur_pose[0], cur_pose[1])))
                if path is not None and len(path) > 0:
                    _final_plan_fail_cnt = 0
                    path_idx = 0
                    _dbg(f"[nav_depth] FINAL_PLAN -> FINAL_FOLLOW "
                         f"({len(path)} waypoints)")
                    yaw_smoother.reset()
                    out_smoother.reset()
                    state = NavState.FINAL_FOLLOW
                else:
                    _final_plan_fail_cnt += 1
                    if _final_plan_fail_cnt >= PLAN_FAIL_MAX:
                        print(f"[nav_depth][ERROR] FINAL 规划不成功 "
                              f"(已连续 {_final_plan_fail_cnt} 次 RRT 规划失败) "
                              f"-> 放弃目标, 回到 WAITING")
                        _final_plan_fail_cnt = 0
                        final_target = None
                        state = NavState.WAITING
                    elif time.time() - nav_path._last_plan_fail_t >= 2.0:
                        _dbg(f"[nav_depth] FINAL 路径规划失败 (第 "
                             f"{_final_plan_fail_cnt}/{PLAN_FAIL_MAX} 次), 等待重试")
                        nav_path._last_plan_fail_t = time.time()

        elif state == NavState.FINAL_FOLLOW:
            _prn_state(state)
            if cur_pose is None:
                motion_thread.stop()
                last_cmd = "stop (pose lost)"
                _dbg("[nav_depth] pose 丢失, 停车等待")
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
                bo_tree = build_obstacle_tree(
                    crop_to_box(obs_global, (cur_pose[0], cur_pose[1]),
                                final_target, PLAN_BOX_PAD), nav_y)
                d_robot = tree_nearest_dist(bo_tree, nav_y,
                                            cur_pose[0], cur_pose[1])
                if (d_robot >= ROBOT_RADIUS and
                        (d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS)):
                    motion_thread.stop()
                    _dbg(f"[nav_depth] FINAL_FOLLOW -> DONE (到达检测目标)")
                    state = NavState.DONE
                else:
                    if d_robot >= ROBOT_RADIUS:
                        _escape_point = None
                    if _escape_point is None:
                        now_t = time.time()
                        if now_t - nav_path._last_periodic_replan_t >= REPLAN_PERIOD:
                            nav_path._last_periodic_replan_t = now_t
                            new_path, new_nav = plan_path(
                                (cur_pose[0], cur_pose[1]),
                                final_target, obs_global, fixed_y=nav_y,
                                plan_start_2d=((cur_pose[0], cur_pose[1])))
                            if new_path is not None and len(new_path) >= 2:
                                path = new_path
                                final_goal_nav = new_nav
                            elif new_nav is not None:
                                if _goal_walked(mem, float(new_nav[0]),
                                                float(new_nav[1]), nav_y):
                                    print(f"[nav_depth][DBG-REPLAN] final "
                                          f"sub-opt-goal={new_nav} 已在 walked "
                                          f"记忆内, 跳过该绕行")
                                else:
                                    final_goal_nav = new_nav
                                    _dbg(f"[nav_depth][DBG-REPLAN] RRT 暂无解, "
                                         f"仅刷新 final sub-opt-goal={new_nav}")

                    if d_robot < ROBOT_RADIUS:
                        if (_escape_point is not None and
                                tree_nearest_dist(bo_tree, nav_y,
                                    float(_escape_point[0]),
                                    float(_escape_point[1])) < ROBOT_RADIUS):
                            _escape_point = None
                        if _escape_point is None:
                            _escape_point = nav_path.find_escape_point(
                                (cur_pose[0], cur_pose[1]), obs_global,
                                fixed_y=nav_y)
                        if time.time() - nav_path._last_intrusion_warn_t >= 2.0:
                            if _escape_point is not None:
                                _dbg(f"[nav_depth] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"RRT 脱困点 E={_escape_point}, 朝其移动")
                            else:
                                _dbg(f"[nav_depth] [WARN] 位姿侵入障碍 "
                                     f"(距最近障碍 {d_robot:.2f}m < "
                                     f"robot_radius {ROBOT_RADIUS:.2f}m), "
                                     f"无自由脱困点, 兜底直线后退")
                            nav_path._last_intrusion_warn_t = time.time()
                        if _escape_point is not None:
                            _el, _ea = nav_path.escape_velocity(
                                cur_pose, _escape_point)
                            if not halted:
                                motion_thread.set_velocity(_el, _ea)
                                motion_thread.start()
                            last_cmd = f"escape->E (near obs {d_robot:.2f}m)"
                        else:
                            if not halted:
                                motion_thread.set_velocity(
                                    -FOLLOW_LIN_MAX * 0.5, 0.0)
                                motion_thread.start()
                            last_cmd = f"escape blind (near obs {d_robot:.2f}m)"
                        _escaping = True
                    elif is_path_blocked(path, path_idx, obs_global,
                                         fixed_y=nav_y, tree=bo_tree):
                        now_t = time.time()
                        if now_t - nav_path._last_replan_t >= 1.0:
                            motion_thread.stop()
                            _dbg("[nav_depth] 路径被阻挡 -> FINAL_PLAN (重规划)")
                            nav_path._last_replan_t = now_t
                            state = NavState.FINAL_PLAN
                        else:
                            motion_thread.stop()
                            last_cmd = "stop (replan cooldown)"
                    else:
                        sm_yaw = yaw_smoother.update(cur_pose[2])
                        sm_pose = (cur_pose[0], cur_pose[1], sm_yaw)
                        cmd, path_idx, lookahead_target, alpha = follow_path_step(
                            path, path_idx, sm_pose)
                        _db = 0.0 if abs(alpha) > FOLLOW_STRAIGHT_ALPHA \
                            else FOLLOW_DEADBAND
                        cmd = (cmd[0], out_smoother.update(cmd[1], deadband=_db))
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
                                _dbg(f"[nav_depth]   FOLLOW cmd "
                                     f"lin={cmd[0]:.2f} ang={cmd[1]:.2f} "
                                     f"wp={path_idx}/{len(path)} "
                                     f"cur_pose=({cur_pose[0]:.2f},"
                                     f"{cur_pose[1]:.2f}) "
                                     f"dist_wp={_d_wp:.2f}m d_tgt={d_tgt:.2f}m")
                                _vlog_t = now_t
                            last_cmd = _cmd_str(cmd[0], cmd[1])
            else:
                pass

        # ===== FPS =====
        now = time.time()
        dt = now - prev_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_t = now
        step += 1

        # ===== Debug overlay =====
        has_img = img is not None
        has_pose = cur_pose is not None
        mode_str = "ODOM"
        if cur_pose is not None:
            # cur_pose 已是 nav (X右, Z前, yaw); 首帧锚后启动约 (0,0,0)
            pose_str = (f"nav x={cur_pose[0]:.2f} z={cur_pose[1]:.2f} "
                        f"yaw={np.degrees(cur_pose[2]):.0f}°")
        else:
            pose_str = "N/A"
        nav_y_str = f"{nav_y:.2f}" if nav_y is not None else "N/A"
        path_str = (f"{path_idx}/{len(path)}" if path else "none")
        blocked_str = "no"
        if path and is_path_blocked(path, path_idx, obs_current):
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
            (f"[SLAM] mode={mode_str} (odom+depth)  img={'Y' if has_img else 'N'}  "
             f"pose={'Y' if has_pose else 'N'}  mapY={nav_y_str}",
             (0, 165, 255)),
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
            lines.append(f"[SCAN]  {np.degrees(scan_accumulated):.0f}/360°")

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
        overlay_frame = draw_debug_overlay(
            img, {"lines": lines, "rects": rects,
                  "robot_pose": cur_pose, "using_odom": _use_odom,
                  "vlm_dets": vlm_latest[0], "vlm_masks": vlm_latest[1]})
        mock.set_frame(overlay_frame)

        obs_pts = obs_current
        memory_gaussians, n_memory_walked = [], 0
        mem_shape, mem_kappa = "disk", 1.0
        frontier_viz = {"frontier_xz": None, "frontier_dist_m": None,
                        "frontier_ms": None}
        if mem is not None:
            try:
                memory_gaussians = [
                    (float(r["mu_W"][0]), float(r["mu_W"][2]),
                     float(r.get("radius_W", 0.4)))
                    for r in mem.export_centers_W()
                    if int(r.get("sid", 0)) == SID_WALKED
                ]
            except Exception:
                memory_gaussians = []
            n_memory_walked = len(memory_gaussians)
            fcfg = mem.cfg.get("frontier_grid", {}) if hasattr(mem, "cfg") else {}
            mem_shape = fcfg.get("shape", "disk")
            mem_kappa = float(fcfg.get("kappa", 1.0))
            frontier_viz = query_frontier_for_viz(
                mem,
                pose_xz=((cur_pose[0], cur_pose[1])
                         if cur_pose is not None else None),
                nav_y=nav_y if nav_y is not None else 0.0,
                obstacle_points=obs_pts,
                robot_radius=ROBOT_RADIUS,
                yaw=(float(cur_pose[2]) if cur_pose is not None else None),
            )

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
            print(f"[NAVSTEP] cyc={_nav_cyc_ms:.0f}ms  grab={_nav_grab_ms:.1f}ms  "
                  f"work={_nav_work_ms:.1f}ms  draw={_nav_draw_ms:.1f}ms  "
                  f"fps={fps:.1f}")
            _nav_step_t = _nav_now

        rate.sleep()

    # 清理
    motion_thread.stop()
    keyboard.stop()      # 恢复终端原始模式
    bot.stop()
    _nav_stop.set()      # 唤醒并退出深度后台刷新线程
    slam.stop()
    try:
        rospy.signal_shutdown("main loop exit")
    except Exception:
        pass
    print("[nav_depth] 导航结束")


if __name__ == "__main__":
    main()
