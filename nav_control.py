#!/usr/bin/env python
"""nav_control — 状态定义 + 共享状态 + 底盘控制线程。

由 nav_state.py 改名扩展而来:
- NavState  状态机枚举
- MockState 线程安全共享状态 (Flask 线程 ↔ 主循环)
- mock      共享单例 (MockState 实例)
- MotionThread / EstopKeyboardThread  底盘控制线程
- 控制相关常量 (MOTION_HZ / SCAN_ANGULAR / RELOC_*)
- _angle_diff  共享几何工具
"""

import sys
import os
import time
import select
import threading
import numpy as np
import open3d as o3d   # 点云体素降采样依赖 (已确认 AGX 环境有 Open3D)
from enum import Enum, auto


# ============================================================
#  状态定义
# ============================================================
class NavState(Enum):
    WAITING = auto()
    SCANNING = auto()
    VLM_INQUIRY = auto()
    PATROL_PLAN = auto()
    PATROL_FOLLOW = auto()
    FINAL_PLAN = auto()
    FINAL_FOLLOW = auto()
    FINAL_ADJUST = auto()  # 检测命中后终调阶段: 边走边持续检测, EMA 平滑终点, 计时后锁定进 FINAL_FOLLOW
    ESCAPE = auto()       # 新 goal 设定后若仍侵入障碍, 先脱困再导航 (不覆盖 nav goal)
    DONE = auto()


# ============================================================
#  Mock 状态 (Flask 线程 ↔ 主循环共享, 线程安全)
# ============================================================
class MockState:
    """线程安全的 mock 事件状态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._vlm_key = None      # 'j'/'k'/'l' 或 None
        self._det_key = None      # 'o' 或 None
        self._detect_target = None  # 网页设的 VLM 检测目标词, o 键消费
        self._h_key = None        # 'h' 或 None (自由问答)
        self._vlm_h_answer = None  # h 键 VLM 问答的文本答案 (网页显示用)
        self._frontier_req = False  # 'f' 触发: 仅查询一次最近 frontier 并设为 patrol 目标
        self._latest_frame = None # 带 overlay 的 BGR 帧 (供 MJPEG)
        self._latest_map = None   # 顶视地图帧 (供 MJPEG)
        self._halt = False        # 急停标志 (网页空格键 toggle)

    # -- VLM mock --
    def set_vlm_key(self, key):
        with self._lock:
            self._vlm_key = key

    def take_vlm_key(self):
        with self._lock:
            k = self._vlm_key
            self._vlm_key = None
            return k

    def peek_vlm_key(self):
        with self._lock:
            return self._vlm_key

    # -- DET mock --
    def set_det_key(self, key):
        with self._lock:
            self._det_key = key

    def take_det_key(self):
        with self._lock:
            k = self._det_key
            self._det_key = None
            return k

    def peek_det_key(self):
        with self._lock:
            return self._det_key

    # -- VLM 检测目标词 (网页输入框设, o 键消费) --
    def set_detect_target(self, s):
        with self._lock:
            self._detect_target = s

    def get_detect_target(self):
        with self._lock:
            return self._detect_target

    # -- H 键自由问答 (网页/终端 h 触发, 仅显示不移动) --
    def set_h_key(self, key):
        with self._lock:
            self._h_key = key

    def take_h_key(self):
        with self._lock:
            k = self._h_key
            self._h_key = None
            return k

    def peek_h_key(self):
        with self._lock:
            return self._h_key

    # -- h 键 VLM 问答答案 (网页轮询显示) --
    def set_vlm_h_answer(self, s):
        with self._lock:
            self._vlm_h_answer = s

    def get_vlm_h_answer(self):
        with self._lock:
            return self._vlm_h_answer

    # -- 手动 frontier 请求 (网页 f 键, 一次性触发) --
    def set_frontier_request(self):
        with self._lock:
            self._frontier_req = True

    def take_frontier_request(self):
        with self._lock:
            v = self._frontier_req
            self._frontier_req = False
            return v

    def peek_frontier_request(self):
        with self._lock:
            return self._frontier_req

    # -- 帧 (供 MJPEG 流) --
    def set_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    # -- 顶视地图帧 (供 MJPEG 流) --
    def set_map(self, frame):
        with self._lock:
            self._latest_map = frame

    def get_map_frame(self):
        with self._lock:
            return self._latest_map

    # -- 急停 (网页空格键 toggle) --
    def set_halt(self, v):
        with self._lock:
            self._halt = bool(v)

    def get_halt(self):
        with self._lock:
            return self._halt

# 共享 mock 单例 (原在 nav_mock.py; 控制层和页面共用)
mock = MockState()


# ============================================================
#  控制相关常量  (单一真相源见 nav_constants.py)
# ============================================================
# MOTION_HZ / SCAN_ANGULAR / RELOC_STOP_TIME / RELOC_BACK_TIME 统一在 nav_constants.py
# 定义并从 ROBOT_RADIUS 派生; 此处 re-export 以兼容既有 import。
from nav_constants import (
    MOTION_HZ,
    SCAN_ANGULAR,
    RELOC_STOP_TIME,
    RELOC_BACK_TIME,
)
# 底盘需 ~20Hz 连续 cmd_vel 才能保持运动 (服务端看门狗 1s 兜底归零), 故本线程
# 以 MOTION_HZ 背景重发最新速度. 频率越高越丝滑、网络请求也越多 (localhost 无压力);
# 太低 (<10Hz) 底盘可能收不到后续指令而微顿. 20Hz (50ms/次) 是平滑与开销的折中.
# ============================================================
#  底盘控制 (连续 cmd_vel 后台线程)
# ============================================================
class MotionThread:
    """通用底盘控制线程 (扫描 + 路径跟随 + 键盘共用)。

    关键: 本线程以固定 MOTION_HZ 持续发送 *连续 cmd_vel* 驱动底盘。底盘需近连续
    的 cmd_vel 流 (~20Hz) 才能保持运动 (服务端看门狗 1s 无指令自动归零)。之所以
    用独立后台线程而非在主循环里发: 主循环重活(GIL)/网络抖动会让请求流时快时慢
    -> 底盘一顿一顿; 独立线程只做"发速度 + 定频休眠", 极轻量, 不被主循环饿死,
    始终以稳态 20Hz 重发最新速度 -> 丝滑。

    (曾用 timed_move: 一次请求让服务端自驱 TIMED_MOVE_DURATION 秒。虽平滑但 stop
     须等当前 timed_move 跑完 -> 大 duration 时急停延迟明显、"停得冲"。改回连续
     cmd_vel 后, stop 即时生效 (线程一个发布周期内退出并发 0), 无急停延迟。)

    主循环只需 set_velocity(lin, ang) 设定想要的线/角速度, 线程以 MOTION_HZ 持续
    重发最新速度; 何时停由主循环调用 stop() 决定。
    """

    def __init__(self, bot, hz=20):
        self.bot = bot
        self.hz = hz
        self._lin = 0.0
        self._ang = 0.0
        self._running = False
        self._stop_flag = False
        self._stop_evt = threading.Event()  # 定频休眠期间即时唤醒 stop, 无急停延迟
        self._thread = None
        self._lock = threading.Lock()

    def set_velocity(self, lin, ang):
        with self._lock:
            self._lin = float(lin)
            self._ang = float(ang)

    def start(self):
        """启动线程 (幂等: 已在跑则仅保持运行, 不重启)。"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_flag = False
            self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_running(self):
        with self._lock:
            return self._running

    def _run(self):
        # 连续 cmd_vel 驱动: 以固定 MOTION_HZ 持续发送最新速度.
        #   底盘需近连续 cmd_vel (~20Hz) 才能保持运动; 每轮发一次并休眠 dt=1/hz,
        #   使 /cmd_vel 稳定以 MOTION_HZ 刷新. 独立后台线程不受主循环重活影响,
        #   故请求流不抖 -> 底盘不顿挫. (相比 timed_move: stop 即时, 无急停延迟.)
        dt = 1.0 / self.hz if self.hz > 0 else 0.05
        # --- 实测探针: 统计相邻两次 cmd_vel 发布的墙钟间隔(gap) 与单次 HTTP 耗时,
        #     每 ~1s 打印一行 [MOTIONHZ]。判读: 稳态 gap 应≈dt (50ms@20Hz);
        #     若 gap_max 远大于 dt (如 200~500ms) => 线程被 GIL/CPU 抢占饿死 =>
        #     /cmd_vel 出空档 => 底盘减速顿挫 (走停走停的执行层根因)。
        #     http_max 大 => 服务端处理慢/请求排队。确诊后可移除本段。
        _last_post = None
        _win_t = time.time()
        _n = 0
        _gmin = _gmax = 0.0
        _hmin = _hmax = 0.0
        while not self._stop_flag:
            with self._lock:
                lin, ang = self._lin, self._ang
            try:
                _t0 = time.time()
                self.bot.cmd_vel(lin, ang)
                _http = time.time() - _t0
            except Exception as e:
                print(f"[nav_2d] motion cmd_vel 失败: {e}")
                time.sleep(0.1)   # 出错时退避, 避免热循环; 网络恢复后自动续驱
                continue
            # 探针累计
            _now = time.time()
            if _last_post is not None:
                _gap = _now - _last_post
                if _n == 0:
                    _gmin = _gmax = _gap
                    _hmin = _hmax = _http
                else:
                    _gmin = min(_gmin, _gap); _gmax = max(_gmax, _gap)
                    _hmin = min(_hmin, _http); _hmax = max(_hmax, _http)
                _n += 1
            _last_post = _now
            if _now - _win_t >= 1.0 and _n > 0:
                print(f"[MOTIONHZ] {(_now-_win_t):.2f}s n={_n} "
                      f"gap=[{_gmin*1000:.0f},{_gmax*1000:.0f}]ms "
                      f"http=[{_hmin*1000:.0f},{_hmax*1000:.0f}]ms "
                      f"(dt={dt*1000:.0f}ms)")
                _win_t = _now; _n = 0
            # 定频休眠: 期间响应 stop (wait 命中即时退出, 不必睡满 dt)
            if self._stop_evt.wait(dt):
                break
        # 退出前兜底停车 (即时归零, 不依赖看门狗)
        try:
            self.bot.stop()
        except Exception:
            pass

    def stop(self):
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_flag = True
            self._stop_evt.set()   # 即时唤醒 _run 的定频休眠 -> stop 无延迟
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class EstopKeyboardThread:
    """终端 Space 急停监听线程。"""

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


# ============================================================
#  共享几何工具
# ============================================================
def _angle_diff(a, b):
    """最短有符号角差 a-b, 归一化到 (-pi, pi] (环绕感知)。"""
    d = (a - b) % (2.0 * np.pi)
    if d > np.pi:
        d -= 2.0 * np.pi
    return d


# ============================================================
#  点云后台刷新缓存 (把最重的 get_map + 丝状滤波移出 10Hz 控制循环)
# ============================================================
PC_HZ = 4.0                      # 点云重建+滤波刷新频率 (Hz), 远低于控制 10Hz
PC_INTERVAL_DT = 1.0 / PC_HZ    # 刷新间隔 (s)
MAP_DS_VOXEL = 0.05             # 全图点云体素降采样尺寸 (m); 0=禁用.
PC_ACCUM_CAP = 500_000          # 持久降采样地图点数上限; 超过则整体重体素 (控规模)
PC_REKEY_KF = 8                 # 每新增 N 个关键帧, 整体重体素刷新 BA 位姿 (仅关键帧边界触发)


class PointCloudCache:
    """后台线程: 以低频 (PC_HZ) 重建整图点云+丝状滤波并缓存, 供 10Hz 主循环只读复用,
    避免 get_map/filter 随地图增大拖慢控制、导致 cmd_vel 抖动。整体替换引用保证线程安全;
    nav_y 每帧 set_nav_y 写入; get_obstacle_points/probe 依赖注入避免反向 import。"""

    def __init__(self, slam, interval_dt, stop_event, get_obstacle_points,
                 probe=None):
        self.slam = slam
        self.interval_dt = interval_dt
        self._stop = stop_event
        self._get_obstacle_points = get_obstacle_points
        self._probe = probe if probe is not None else (lambda k, m: None)
        self._lock = threading.Lock()
        self._map_pts = None                                # (N,3) 或 None
        self._obs = np.empty((0, 3), dtype=np.float64)     # 滤波后障碍点
        self._n_map = 0
        self._nav_y = None
        self._ds_voxel = MAP_DS_VOXEL                       # 全图体素降采样尺寸
        self._accum = np.empty((0, 3), dtype=np.float64)   # 持久降采样地图 (增量合并, 不存裸点)
        self._last_kf = 0                                  # 已合并的关键帧游标
        self._last_dedup_kf = 0                            # 上次整体重体素的游标
        self._map_raw_n = 0                                # 等价裸点总数 (增量累计)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.name = "PointCloudCache"

    def set_nav_y(self, y):
        self._nav_y = y

    def start(self):
        self._thread.start()

    def snapshot(self):
        """返回当前缓存快照: (_map_pts, obs_current, n_map)。"""
        with self._lock:
            return self._map_pts, self._obs, self._n_map

    def _run(self):
        _ds_voxel = self._ds_voxel
        while not self._stop.is_set():
            t0 = time.time()
            dt_map = dt_ds = dt_obs = 0.0
            n_kf = -1
            n_map_ds = -1
            n_obs = -1
            rebuilt = False
            try:
                n_kf = self.slam.num_keyframes()
                if n_kf > self._last_kf:
                    # 有新增关键帧: 仅取新增部分降采样合并入持久地图 (不再每帧重建全图裸点, 根因)
                    new_pts, n2 = self.slam.get_keyframe_points_since(self._last_kf)
                    self._last_kf = n2
                    if new_pts is not None and len(new_pts) > 0:
                        ta = time.time()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(
                            np.ascontiguousarray(new_pts[:, :3], dtype=np.float64))
                        new_ds = np.asarray(
                            pcd.voxel_down_sample(voxel_size=_ds_voxel).points,
                            dtype=np.float64)
                        dt_ds = time.time() - ta
                        self._accum = np.concatenate([self._accum, new_ds], axis=0)
                        self._map_raw_n += len(new_pts)
                        rebuilt = True
                    # 定期/超限整体重体素: 控规模+用最新 BA 位姿刷新 (仅关键帧边界触发)
                    if (n2 - self._last_dedup_kf) >= PC_REKEY_KF \
                            or len(self._accum) > PC_ACCUM_CAP:
                        ta = time.time()
                        full, _ = self.slam.get_map()
                        dt_map = time.time() - ta
                        if full is not None and len(full) > 0:
                            self._map_raw_n = len(full)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(
                                np.ascontiguousarray(full[:, :3], dtype=np.float64))
                            self._accum = np.asarray(
                                pcd.voxel_down_sample(voxel_size=_ds_voxel).points,
                                dtype=np.float64)
                            self._last_dedup_kf = n2
                            rebuilt = True
                # 障碍提取始终基于降采样后的持久地图 (规模可控, 快)
                tc = time.time()
                nav_y = self._nav_y
                obs = None
                if nav_y is not None and len(self._accum) > 0:
                    obs = self._get_obstacle_points(self.slam, nav_y,
                                                    map_pts=self._accum)
                dt_obs = time.time() - tc
                if obs is None:
                    obs = np.empty((0, 3), dtype=np.float64)
                n_map_ds = len(self._accum)
                n_obs = len(obs)
                with self._lock:
                    self._map_pts = self._accum if len(self._accum) > 0 else None
                    self._obs = obs
                    self._n_map = n_map_ds
            except Exception:
                # SLAM 关闭中 / 取图异常: 保留上一帧缓存, 不致命
                pass
            # 维持固定低频刷新; 用 Event.wait 让 stop 能即时唤醒
            dt = time.time() - t0
            slp = self.interval_dt - dt
            if slp > 0:
                self._stop.wait(slp)
