"""
Mast3rSlamWrapper - MASt3R-SLAM 实时封装类

内部订阅 ROS RGB 话题，持续取帧喂给 SLAM。
SLAM 处理速度跟不上时自动丢弃中间帧，只处理最新的一帧。
外部随时调用 get_pose() / get_pointcloud() 获取当前结果。

用法:
    from mast3r_slam_wrapper import Mast3rSlamWrapper

    slam = Mast3rSlamWrapper(rgb_topic="/camera_f/color/image_raw")
    slam.start()   # 启动内部线程

    # 在路径规划循环中:
    pose = slam.get_pose()        # (x, y, yaw) 或 None
    cloud = slam.get_pointcloud()  # (N,3) numpy 或 None

    slam.stop()

架构:
    ROS RGB话题 ---> [_img_callback] ---> _latest_msg (只保留最新)
                                            |
                                    [_process_loop]
                                            v
                                  只处理最新帧，中间帧丢弃
                                            |
                                    SLAM 前端跟踪
                                            |
                                get_pose() / get_pointcloud()
"""

import sys
import time
import threading
import dataclasses
import pathlib

import numpy as np
import torch
import lietorch
import torch.multiprocessing as mp
import cv2
import rospy
import yaml

# ===== 路径设置: 让 mast3r_slam 包可导入 =====
_SLAM_DIR = pathlib.Path(__file__).resolve().parent.parent / "MASt3R-SLAM"
if str(_SLAM_DIR) not in sys.path:
    sys.path.insert(0, str(_SLAM_DIR))

from mast3r_slam.config import load_config, config, set_global_config

# 处理帧端到端延迟上限 (s): 取到的相机帧距真实捕获时刻超过此值即视为"积压旧帧",
# 直接丢弃不进 _process_frame。防"停车后还要 3-4 秒追旧帧"——订阅线程被前端重负载
# 饿死时, 内核 socket 缓冲会攒下停车前的旧帧, FIFO 消化完位姿/显示才到位。
SLAM_STALE_DROP_S = 0.5
# RELOC 期间前端处理限流间隔 (秒): 重定位只需少量候选帧, 限流释放 GIL 给 ROS
# 回调线程维持 RGB 显示, 避免 RELOC 时画面完全冻结。
RELOC_FRONTEND_MIN_DT = 0.12

def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base (override 优先), 返回新 dict。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
    resize_img,
)
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.global_opt import FactorGraph
from nav_memory import MemorySystem, sim3_translation, SID_WALKED

import sensor_msgs.msg
from cv_bridge import CvBridge


# ============================================================
#  无可视化时的消息替代 (避免 import visualization 依赖 OpenGL)
# ============================================================
@dataclasses.dataclass
class _WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


# ============================================================
#  后端函数 (从 main.py 原样复制, 不修改原文件)
# ============================================================
def _relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("\033[93mRELOCALIZING against kf", n_kf - 1, "and", kf_idx, "\033[0m")
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("\033[93mSuccess! Relocalized\033[0m")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("\033[93mFailed to relocalize\033[0m")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def _run_backend(cfg, model, states, keyframes, K):
    """后端进程: 全局因子图优化 + 回环检测 + 重定位。"""
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    
    retrieval_database = load_retriever(model,retriever_path=RETRIEVER_WEIGHT)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = _relocalization(
                frame, keyframes, factor_graph, retrieval_database
            )
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # 图构建
        kf_idx = []
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ":", lc_inds)

        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


# ============================================================
#  主类
# ============================================================
MAST3R_WEIGHT = str(_SLAM_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
RETRIEVER_WEIGHT = str(_SLAM_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
class Mast3rSlamWrapper:
    """MASt3R-SLAM 实时封装类。

    内部订阅 ROS RGB 话题，持续取帧喂给 SLAM。
    SLAM 处理速度跟不上时自动丢弃中间帧，只处理最新的一帧。

    参数:
        rgb_topic:    ROS RGB 话题名 (默认 "/camera_f/color/image_raw")
        config_path:  YAML 配置文件路径 (默认自动找 MASt3R-SLAM/config/base.yaml)
        device:       CUDA 设备名
        no_viz:       是否禁用 OpenGL 可视化 (默认 True)
        calib_path:   标定 YAML 路径 (可选; 不传则用纯单目无标定模式)
        img_size:      MASt3R 推理分辨率 (默认 512)
    """

    def __init__(
        self,
        rgb_topic="/camera_f/color/image_raw",
        config_path=None,
        device="cuda:0",
        no_viz=True,
        calib_path=None,
        img_size=512,
    ):
        # --- 多进程初始化 ---
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        self.rgb_topic = rgb_topic
        self.device = device
        self.no_viz = no_viz
        self.img_size = img_size

        # --- 加载配置 ---
        if config_path is None:
            config_path = str(_SLAM_DIR / "config" / "base.yaml")
        load_config(config_path)

        # --- 标定 (可选) ---
        self._calib_K = None
        if calib_path:
            import yaml
            from mast3r_slam.dataloader import Intrinsics

            with open(calib_path, "r") as f:
                intr = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            calib_data = intr["calibration"]
            intrinsics = Intrinsics.from_calib(
                img_size, intr["width"], intr["height"], calib_data
            )
            self._calib_K = torch.from_numpy(intrinsics.K_frame).to(
                device, dtype=torch.float32
            )

        # --- 多进程 Manager + 队列 ---
        self.manager = mp.Manager()
        self.main2viz = new_queue(self.manager, no_viz)
        self.viz2main = new_queue(self.manager, no_viz)
        self.last_msg = _WindowMsg()

        # --- 加载模型 ---
        print("[SLAM] Loading MASt3R model...")
        
        self.model = load_mast3r(path=MAST3R_WEIGHT,device=device)
        self.model.share_memory()
        print("[SLAM] Model loaded.")

        # --- 语义记忆: 仅挂接 MemorySystem ---
        #   cfg (几何/阈值) 与 vocab (语义词表) 全部由 nav_constants.py 装配
        #   (ROBOT_RADIUS 派生 + MEM_VOCAB_ENTRIES), 与 nav 统一单一真相源.
        self.memory = MemorySystem()
        self._memory_debug_every = int(self.memory.cfg.get("debug_every", 30))
        print(
            f"[memory] enabled={self.memory.enable} "
            f"vocab_entries={len(self.memory.vocab._id_to_name)} "
            f"walked.min_trans={self.memory.cfg['walked']['min_trans']:.3f} "
            f"(< PATROL_ARRIVE_EPS, 见 nav_constants.py) "
            f"debug_every={self._memory_debug_every} "
            f"(API: nav_memory.*)"
        )

    # --- SLAM 状态 (懒初始化) ---
        self.states = None
        self.keyframes = None
        self.tracker = None
        self.backend_proc = None
        self.viz_proc = None
        self._initialized = False
        self._frame_idx = 0
        self._add_new_kf = False

        # --- ROS 订阅相关 ---
        self._bridge = CvBridge()
        self._sub = None
        self._latest_msg = None   # 供 SLAM 处理线程消费 (消费即置空)
        self._display_msg = None  # 供显示线程读; 处理线程不置空, 只被回调覆盖
        self._latest_msg_lock = threading.Lock()
        # --- 诊断: 观测 ROS 回调是否还在触发 (不改帧流/get_img 行为) ---
        self._cb_count = 0        # 累计收到的回调帧数
        self._cb_last_t = 0.0     # 最近一次回调墙钟时间
        self._cb_stall_warned = False  # 停流告警去抖
        self._reloc_last_proc_t = 0.0  # RELOC 前端限流: 上次处理帧的墙钟时刻
        # --- 位姿 CPU 缓存 (由 SLAM 处理线程刷新, getter 只读, 主循环不碰 GPU) ---
        #   每处理完一帧, _refresh_pose_cache() 把 4 份原始 T_WC.data 从 GPU 拷成
        #   CPU numpy 存这里; get_pose/get_pose_full/get_pose_keyframe/
        #   get_pose_full_keyframe 改读这些缓存, 套用与原来完全相同的数学 ->
        #   返回值逐位一致, 但 .cpu() 同步从主循环挪到已 GPU-bound 的 SLAM 线程,
        #   主循环彻底脱离 GPU 排队 (原每圈 5 次 CUDA 同步 -> 0)。
        #   时间分辨率 = SLAM 帧率 (每处理一帧刷一次), 导航决策足够。
        self._pose_cache_lock = threading.Lock()
        self._pc_cur_sim3 = None  # 当前帧 T_WC.data 展平 (Sim3, len>=7): 供 get_pose
        self._pc_cur_se3 = None   # as_SE3(当前帧 T_WC).data 展平 (7): 供 get_pose_full
        self._pc_kf_sim3 = None   # 最后关键帧 T_WC.data 展平: 供 get_pose_keyframe
        self._pc_kf_se3 = None    # as_SE3(最后关键帧 T_WC).data 展平: 供 get_pose_full_keyframe
        self._stop_event = threading.Event()
        self._thread = None
        #   _stopped: 幂等标志, 保证线程/子进程清理只执行一次。
        self._stopped = False

    # --------------------------------------------------------
    #  内部: 第一帧时根据图像尺寸创建 SharedStates / Keyframes
    # --------------------------------------------------------
    def _lazy_init(self, img):
        resized = resize_img(img, self.img_size)
        h, w = resized["img"][0].shape[1:]
        print(f"[SLAM] Processed image shape: ({h}, {w})")

        self.keyframes = SharedKeyframes(self.manager, h, w)
        self.states = SharedStates(self.manager, h, w)

        K = self._calib_K
        if K is not None:
            self.keyframes.set_intrinsics(K)

        self.tracker = FrameTracker(self.model, self.keyframes, self.device)

        # 记忆绑定关键帧缓冲 (只读 T_WC / 长度)
        self.memory.bind_keyframes(self.keyframes)
        self.memory.set_write_enabled(False)
        print("[memory] bound to SharedKeyframes")

        # 启动后端子进程
        self.backend_proc = mp.Process(
            target=_run_backend,
            args=(config, self.model, self.states, self.keyframes, K),
        )
        self.backend_proc.start()

        # 启动可视化子进程 (可选)
        if not self.no_viz:
            from mast3r_slam.visualization import run_visualization

            self.viz_proc = mp.Process(
                target=run_visualization,
                args=(config, self.states, self.keyframes, self.main2viz, self.viz2main),
            )
            self.viz_proc.start()

        self._initialized = True
        print("[SLAM] Initialized. Backend process started.")

    # --------------------------------------------------------
    #  内部: ROS 消息转 numpy (RGB, float32, 0-1)
    # --------------------------------------------------------
    def _msg_to_img(self, msg):
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        return img_rgb.astype(np.float32) / 255.0

    # --------------------------------------------------------
    #  ROS 回调: 只保留最新帧
    # --------------------------------------------------------
    def _img_callback(self, msg):
        with self._latest_msg_lock:
            self._latest_msg = msg   # 供 SLAM 处理线程消费 (会被处理线程置空)
            self._display_msg = msg  # 供显示线程读; 处理线程不置空
            self._cb_count += 1      # 诊断: 回调仍在触发的证据
            self._cb_last_t = time.time()
            self._cb_stall_warned = False

    # --------------------------------------------------------
    #  内部: 处理一帧 (原 process_frame 逻辑)
    # --------------------------------------------------------
    def _process_frame(self, img):
        if not self._initialized:
            self._lazy_init(img)

        mode = self.states.get_mode()
        if mode == Mode.TERMINATED:
            return

        # 记忆写门闩 + KF pop 同步 (backend reloc 失败会 pop_last)
        if mode == Mode.TRACKING:
            self.memory.set_write_enabled(True)
        else:
            self.memory.set_write_enabled(False)
        self.memory.sync_keyframe_count(len(self.keyframes))

        # 检查可视化/控制消息
        msg = try_get_msg(self.viz2main)
        if msg is not None:
            self.last_msg = msg
        if self.last_msg.is_terminated:
            self.states.set_mode(Mode.TERMINATED)
            return

        if self.last_msg.is_paused and not self.last_msg.next:
            self.states.pause()
            return
        if not self.last_msg.is_paused:
            self.states.unpause()

        # 获取上一帧位姿作为初始值
        T_WC = (
            lietorch.Sim3.Identity(1, device=self.device)
            if self._frame_idx == 0
            else self.states.get_frame().T_WC
        )

        frame = create_frame(
            self._frame_idx, img, T_WC,
            img_size=self.img_size, device=self.device,
        )

        # === INIT: 第一帧 ===
        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self.states.set_mode(Mode.TRACKING)
            self.states.set_frame(frame)
            self._frame_idx += 1
            return

        # === TRACKING ===
        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = self.tracker.track(frame)
            self._add_new_kf = add_new_kf
            if try_reloc:
                self.states.set_mode(Mode.RELOC)
            self.states.set_frame(frame)

        # === RELOC ===
        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            self.states.set_frame(frame)
            self.states.queue_reloc()
            while config["single_thread"]:
                with self.states.lock:
                    if self.states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        # 新关键帧
        if self._add_new_kf:
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self.memory.sync_keyframe_count(len(self.keyframes))
            while config["single_thread"]:
                with self.states.lock:
                    if len(self.states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        # --- 记忆: walked 写入 + 周期调试输出 ---
        if mode == Mode.TRACKING and self.memory.enable:
            robot_W = sim3_translation(frame.T_WC)
            wrote = self.memory.maybe_write_walked(
                robot_W, timestamp=time.time()
            )
            # if (
            #     self._memory_debug_every > 0
            #     and self._frame_idx % self._memory_debug_every == 0
            # ):
            #     self._log_memory_debug(robot_W, wrote=wrote, mode=mode)

        self._frame_idx += 1

    # --------------------------------------------------------
    #  内部: 处理循环线程
    #  持续取最新帧，SLAM 忙时丢弃中间帧
    # --------------------------------------------------------
    def _process_loop(self):
        """内部线程: 从 ROS 取最新帧，喂给 SLAM。

        丢弃策略: 回调只保留最新 msg。处理线程每次取走最新 msg 后，
        如果处理期间新到了 3 帧，_latest_msg 已被覆盖为第 3 帧，
        处理完后直接处理第 3 帧，第 1、2 帧被丢弃。
        """
        while not self._stop_event.is_set():
            # 取当前 SLAM 模式; states 懒初始化完成前为 None (首帧会触发 _process_frame
            # 创建 states), 此时 mode=None 不影响下方 RELOC 限流判断 (None!=RELOC)。
            mode = self.states.get_mode() if self.states is not None else None
            if mode == Mode.TERMINATED:
                break

            # 取最新消息 (原子操作: 取出后立刻置 None)
            with self._latest_msg_lock:
                msg = self._latest_msg
                self._latest_msg = None
                cb_count = self._cb_count
                cb_last_t = self._cb_last_t
                cb_warned = self._cb_stall_warned

            # 诊断: 回调 (ROS 新帧到达) 是否已停止 — 暴露"没有新帧进来"
            if cb_last_t > 0 and (time.time() - cb_last_t) > 2.0 and not cb_warned:
                print(f"[SLAM] RGB 回调已 {time.time() - cb_last_t:.1f}s 无新帧 "
                      f"(累计收到 {cb_count} 帧) — 相机节点/ROS 上游可能停流")
                with self._latest_msg_lock:
                    self._cb_stall_warned = True

            if msg is None:
                time.sleep(0.01)
                continue

            # 记录相机帧头时间戳, 用于丢弃积压旧帧.
            _h = getattr(msg, "header", None)
            _st = _h.stamp if _h is not None else None
            _stamp = _st.to_sec() if (_st is not None and _st.to_sec() > 0) else 0.0
            # 陈旧帧丢弃: 订阅线程被前端重负载饿死时, 内核 socket 缓冲会攒下停车前的旧帧,
            # 这些帧 FIFO 消化完位姿/显示才到位 -> "停车后还滞后 3-4s". 超龄直接跳过,
            # 让前端始终处理当前帧, 位姿立即跟到停车位置.
            if _stamp > 0 and (time.time() - _stamp) > SLAM_STALE_DROP_S:
                continue

            img = self._msg_to_img(msg)

            # ---- RELOC 期间前端限流 (修 RGB 卡死) ----
            # RELOC 时前端每帧跑 mono 推理, 后端同时跑重定位匹配, 二者抢 GIL/GPU;
            # 前端若每帧连轴转会长期霸占 GIL, 饿死 ROS 回调线程 -> _display_msg 停
            # 更 -> RGB 显示冻结。重定位只需少量候选帧, 这里在每帧处理之间 sleep 释放
            # GIL 给 ROS 回调, 让最新帧持续写进 _display_msg, 显示保持流动。
            # (位姿缓存 _refresh_pose_cache 保留: RELOC 恢复当圈 nav 需要正确位姿)
            if mode == Mode.RELOC:
                _gap = time.time() - self._reloc_last_proc_t
                if _gap < RELOC_FRONTEND_MIN_DT:
                    time.sleep(RELOC_FRONTEND_MIN_DT - _gap)
                self._reloc_last_proc_t = time.time()
            self._process_frame(img)
            self._refresh_pose_cache()

    # --------------------------------------------------------
    #  内部: 刷新位姿 CPU 缓存 (仅在 SLAM 处理线程调用)
    # --------------------------------------------------------
    def _refresh_pose_cache(self):
        """把当前帧 / 最后关键帧的 T_WC 从 GPU 拷成 CPU numpy, 存入缓存。

        取的是与 4 个 getter 完全相同的原始张量:
          - 当前帧:   states.get_frame().T_WC          -> Sim3 data + as_SE3 data
          - 关键帧:   keyframes.last_keyframe().T_WC    -> Sim3 data + as_SE3 data
        getter 读这些缓存后套用与原来一致的数学 (arctan2 / 索引 / tuple),
        因此返回值逐位相同。此处的 .cpu() 同步发生在已 GPU-bound 的 SLAM
        线程 (刚跑完推理, GPU 已同步, 拷贝很快), 不再拖慢主循环。
        """
        cur_sim3 = cur_se3 = kf_sim3 = kf_se3 = None
        try:
            from mast3r_slam.lietorch_utils import as_SE3
        except Exception:
            as_SE3 = None

        # 当前帧 (供 get_pose / get_pose_full)
        if self.states is not None:
            frame = self.states.get_frame()
            if frame is not None and frame.T_WC is not None:
                cur_sim3 = frame.T_WC.data.clone().cpu().numpy().reshape(-1)
                if as_SE3 is not None:
                    try:
                        cur_se3 = as_SE3(frame.T_WC).data.clone().cpu().numpy().reshape(-1)
                    except Exception:
                        cur_se3 = None

        # 最后关键帧 (供 get_pose_keyframe / get_pose_full_keyframe)
        if self.keyframes is not None and len(self.keyframes) > 0:
            kf = self.keyframes.last_keyframe()
            if kf is not None and kf.T_WC is not None:
                kf_sim3 = kf.T_WC.data.clone().cpu().numpy().reshape(-1)
                if as_SE3 is not None:
                    try:
                        kf_se3 = as_SE3(kf.T_WC).data.clone().cpu().numpy().reshape(-1)
                    except Exception:
                        kf_se3 = None

        with self._pose_cache_lock:
            self._pc_cur_sim3 = cur_sim3
            self._pc_cur_se3 = cur_se3
            self._pc_kf_sim3 = kf_sim3
            self._pc_kf_se3 = kf_se3

    # --------------------------------------------------------
    #  启动 (在 rospy.init_node 之后调用)
    # --------------------------------------------------------
    def start(self):
        """启动内部处理线程和 ROS 订阅。

        必须在 rospy.init_node() 之后调用。
        """
        if self._thread is not None:
            print("[SLAM] Already started.")
            return

        # buff_size 必须远大于单条图像消息 (640x480x3 ~= 900KB), 否则 rospy 默认 64KB
        # 接收缓冲会让大图消息在内核 socket 缓冲区积压: queue_size=1 只在反序列化后的
        # Python 队列生效, 管不到 socket 层的积压, 接收线程只能按 FIFO 逐条消化旧帧 ->
        # 画面/位姿延迟累积, 停车后才"缓缓追上"。设 2**24(16MB) 一次 recv 吞掉整条大消息,
        # 不再积压, 配合 queue_size=1 才能真正总取最新帧、丢弃旧帧。tcp_nodelay 关 Nagle。
        self._sub = rospy.Subscriber(
            self.rgb_topic,
            sensor_msgs.msg.Image,
            self._img_callback,
            queue_size=1,
            buff_size=2 ** 24,
            tcp_nodelay=True,
        )
        print(f"[SLAM] Subscribed to {self.rgb_topic}")

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("[SLAM] Processing thread started.")

    # --------------------------------------------------------
    #  获取当前位姿
    # --------------------------------------------------------
    def get_pose(self):
        """获取当前相机 (地面) 位姿。

        返回:
            (x, z, yaw) 元组, 或 None (还没处理过帧或 MODE==INIT)

        x, z: 相机在世界坐标系中的 *地面* 平移 (导航 X-Z 平面), 与地图
              点云 get_map / 障碍物 / 规划路径同源同全局帧, 可直接用于
              路径跟随与移动控制。
        yaw:  在 X-Z 地面平面内的航向角 (弧度), 绕竖直 Y 轴;
              与导航 (x, z) 平面一致 (非绕 Z 轴的偏航)

        重要修正: 此前返回 (x, y, yaw), y 是相机 Y(高度/下) 轴, 与导航
              所需的地面 Z 轴不一致, 直接用于跟随会把"相机高度"当"前进
              深度", 坐标错乱。这里改用 data[2] (Z) 作为第二分量, 与
              get_keyframe_pose 保持一致。
        """
        if self.states is None:
            return None

        mode = self.states.get_mode()
        if mode == Mode.INIT:
            return None

        # 读 CPU 缓存 (由 SLAM 线程 _refresh_pose_cache 每帧刷新), 不碰 GPU。
        # 缓存的是当前帧 frame.T_WC.data 展平, 与原来直接 GPU 读的数组逐位一致。
        with self._pose_cache_lock:
            data = self._pc_cur_sim3
        if data is None or len(data) < 7:
            return None

        x = float(data[0])
        z = float(data[2])
        qx, qy, qz, qw = data[3], data[4], data[5], data[6]
        # 航向 (X-Z 地面平面内的 yaw, 绕竖直 Y 轴):
        #   取相机前进向量 (R·[0,0,1]) 在 X-Z 平面的投影角
        #   = atan2(R[0,2], R[2,2])
        #   = atan2(2(qx·qz + qw·qy), 1 - 2(qx^2 + qy^2))
        #   (之前误用"绕 Z 轴偏航"公式, 量的是 X-Y 平面,
        #    与导航 X-Z 平面不一致 -> 机器人转向相反/画龙)
        
        yaw = float(np.arctan2(
            2.0 * (qx * qz + qw * qy),
            1.0 - 2.0 * (qx ** 2 + qy ** 2),
        ))
        return (x, z, yaw)

    def get_pose_full(self):
        """获取完整 7-DoF 位姿 (Sim3 转 SE3)。

        返回:
            (x, y, z, qx, qy, qz, qw) 元组, 或 None
        """
        if self.states is None:
            return None

        mode = self.states.get_mode()
        if mode == Mode.INIT:
            return None

        # 读 CPU 缓存 (as_SE3(当前帧 T_WC).data 展平), 不碰 GPU; 与原 GPU 读逐位一致。
        with self._pose_cache_lock:
            data = self._pc_cur_se3
        if data is None:
            return None
        return tuple(float(v) for v in data)

    # --------------------------------------------------------
    #  获取最后一个被 BA 校正的关键帧位姿 (导航/显示用, 不漂)
    # --------------------------------------------------------
    def get_pose_keyframe(self):
        """获取最后一个关键帧的位姿 (x, y, yaw), 已被后端全局 BA 校正。

        与 get_pose() 的区别:
          - get_pose()     读当前帧 states.get_frame().T_WC, 是前端跟踪的
                           一次性估计, 不被后端回灌校正, 快转后会漂。
          - 本函数         读最后一个关键帧, 其 T_WC 由 run_backend 的全局
                           BA 持续校正 (update_T_WCs), 处于真实轨迹上,
                           旋转/运动停止后不会"迟滞", 定位准确。
        代价: 时间分辨率 = 关键帧粒度 (只在新增/校正关键帧时更新),
              不如当前帧丝滑。导航/显示用足够。

        返回 (x, y, yaw) 或 None。
        """
        # 读 CPU 缓存 (最后关键帧 T_WC.data 展平), 不碰 GPU; 与原 GPU 读逐位一致。
        # 缓存每帧刷新一次 (SLAM 帧率), 关键帧 BA 校正随刷新反映, 导航足够。
        with self._pose_cache_lock:
            data = self._pc_kf_sim3
        if data is None or len(data) < 7:
            return None
        x = float(data[0])
        z = float(data[2])
        qx, qy, qz, qw = data[3], data[4], data[5], data[6]
        # 航向 (X-Z 地面平面内的 yaw, 绕竖直 Y 轴):
        #   取相机前进向量 (R·[0,0,1]) 在 X-Z 平面的投影角
        #   = atan2(R[0,2], R[2,2])
        #   = atan2(2(qx·qz + qw·qy), 1 - 2(qx^2 + qy^2))
        #   (之前误用"绕 Z 轴偏航"公式, 量的是 X-Y 平面,
        #    与导航 X-Z 平面不一致 -> 机器人转向相反/画龙)
        yaw = float(np.arctan2(
            2.0 * (qx * qz + qw * qy),
            1.0 - 2.0 * (qx ** 2 + qy ** 2),
        ))
        return (x, z, yaw)

    def get_pose_full_keyframe(self):
        """获取最后一个关键帧的完整 7-DoF 位姿 (x, y, z, qx, qy, qz, qw)。

        用于需要相机世界坐标 (距离过滤等) 的场景, 与地图点云同源。
        """
        # 读 CPU 缓存 (as_SE3(最后关键帧 T_WC).data 展平), 不碰 GPU; 与原 GPU 读逐位一致。
        with self._pose_cache_lock:
            data = self._pc_kf_se3
        if data is None:
            return None
        return tuple(float(v) for v in data)

    # --------------------------------------------------------
    #  获取当前帧点云
    # --------------------------------------------------------
    def get_pointcloud(self, world_frame=True, conf_threshold=0.0):
        """获取当前帧的点云。

        参数:
            world_frame:     是否转换到世界坐标系 (默认 True)
            conf_threshold:  置信度阈值, 低于此值的点会被过滤

        返回:
            numpy.ndarray, shape (N, 3), float32, 或 None
        """
        if self.states is None:
            return None

        frame = self.states.get_frame()
        if frame is None or frame.X_canon is None:
            return None

        X = frame.X_canon.clone()

        if world_frame:
            T_WC = lietorch.Sim3(frame.T_WC.data.clone())
            X = T_WC.act(X)

        X_np = X.cpu().numpy()

        if conf_threshold > 0 and frame.C is not None:
            C = frame.C.clone().cpu().numpy().reshape(-1)
            mask = C > conf_threshold
            X_np = X_np[mask]

        return X_np

    # --------------------------------------------------------
    #  获取当前帧的逐像素点云 (H, W, 3)
    # --------------------------------------------------------
    def get_pointcloud_2d(self, world_frame=True):
        """获取当前帧的逐像素点云, 保持 (H, W, 3) 形状。

        和 get_pointcloud() 的区别: 返回的数组保持图像的 2D 结构,
        可以直接用 bbox 切片取出对应区域的 3D 点。

        注意:
            H, W 是 SLAM 内部处理后的分辨率 (可能和 get_img() 不同)。
            使用 bbox 时需要按比例缩放坐标:
              scale_x = W_slam / W_orig
              scale_y = H_slam / H_orig
            不做置信度过滤 (过滤会破坏像素对应关系)。

        参数:
            world_frame: 是否转换到世界坐标系 (默认 True)

        返回:
            numpy.ndarray, shape (H, W, 3), float32, 或 None
        """
        if self.states is None:
            return None

        frame = self.states.get_frame()
        if frame is None or frame.X_canon is None:
            return None

        X = frame.X_canon.clone()

        if world_frame:
            T_WC = lietorch.Sim3(frame.T_WC.data.clone())
            X = T_WC.act(X)

        X_np = X.cpu().numpy()

        # img_shape 是 (1, 2) = [[H, W]]
        h, w = frame.img_shape[0].cpu().tolist()
        X_np = X_np.reshape(h, w, 3)

        return X_np

    # --------------------------------------------------------
    #  获取所有关键帧累积点云 (完整地图)
    # --------------------------------------------------------
    def get_map(self, conf_threshold=1.5):
        """获取所有关键帧累积的地图点云。

        返回:
            (points, colors) 元组:
              points: numpy.ndarray (N, 3), float32, 世界坐标
              colors: numpy.ndarray (N, 3), uint8, RGB
            或 (None, None) 如果没有关键帧
        """
        if self.keyframes is None or len(self.keyframes) == 0:
            return None, None

        all_points = []
        all_colors = []

        for i in range(len(self.keyframes)):
            kf = self.keyframes[i]
            pW = kf.T_WC.act(kf.X_canon).cpu().numpy().reshape(-1, 3)
            color = (kf.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
            if kf.C is not None:
                C = kf.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
                valid = C > conf_threshold
                pW = pW[valid]
                color = color[valid]
            all_points.append(pW)
            all_colors.append(color)

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        return points, colors

    # --------------------------------------------------------
    #  增量取点: 仅返回索引 >= idx 的关键帧世界系点云 (持久地图用)
    # --------------------------------------------------------
    def get_keyframe_points_since(self, idx, conf_threshold=1.5):
        """增量取点: 仅返回索引 >= idx 的关键帧世界系点云堆叠 (N, 3) float64。

        用于 PointCloudCache 增量合并, 避免每帧 get_map() 重算全部关键帧
        (裸点数随时间单调膨胀到数百万, 重算 + 体素降采样每帧吃掉 ~250ms)。

        返回 (points, n_current):
          - points 为 None 表示 idx 之后无新增关键帧;
          - n_current 为当前关键帧总数, 调用方据此推进游标。
        """
        if self.keyframes is None:
            return None, 0
        n = len(self.keyframes)
        if idx >= n:
            return None, n
        chunks = []
        for i in range(idx, n):
            kf = self.keyframes[i]
            pW = kf.T_WC.act(kf.X_canon).cpu().numpy().reshape(-1, 3)
            if kf.C is not None:
                C = kf.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
                valid = C > conf_threshold
                pW = pW[valid]
            if len(pW) > 0:
                chunks.append(pW.astype(np.float64))
        if not chunks:
            return None, n
        return np.concatenate(chunks, axis=0), n

    # --------------------------------------------------------
    #  获取当前模式
    # --------------------------------------------------------
    def get_mode(self):
        """返回当前 SLAM 模式 (Mode 枚举或 None)。"""
        if self.states is None:
            return None
        return self.states.get_mode()

    # --------------------------------------------------------
    #  获取最新 RGB 帧 (numpy, BGR uint8)
    # --------------------------------------------------------
    def get_img(self):
        """获取 ROS 回调收到的最新 RGB 帧（未经过 SLAM 处理）。

        返回:
            numpy.ndarray, shape (H,W,3), dtype=uint8, BGR 格式
            如果还没有收到任何帧, 返回 None

        注意: 读的是 _display_msg, 不是 _latest_msg。
        _latest_msg 被 SLAM 处理线程消费即置空, 导致显示层在处理期间读到 None 而闪黑;
        _display_msg 只被回调覆盖、处理线程不置空, 因此只要上游有流, 显示就稳定。
        上游真停流仍由 _cb_last_t / _cb_count 独立检测并打印告警, 不会被掩盖。
        """
        with self._latest_msg_lock:
            msg = self._display_msg
            last_t = self._cb_last_t
            warned = self._cb_stall_warned

        if msg is None:
            return None

        # 上游真停流检测: 超过阈值仍返回当前帧, 但打印一次告警
        if last_t > 0 and (time.time() - last_t) > 2.0 and not warned:
            print(f"[SLAM] RGB 回调已 {time.time() - last_t:.1f}s 无新帧 "
                  f"(累计收到 {self._cb_count} 帧) — 相机节点/ROS 上游可能停流")
            with self._latest_msg_lock:
                self._cb_stall_warned = True

        # 用 cv_bridge 直接转成 BGR uint8 (OpenCV 可直接 imshow)
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return cv_img

    def num_keyframes(self):
        """返回当前关键帧数量。"""
        if self.keyframes is None:
            return 0
        return len(self.keyframes)

    # --------------------------------------------------------
    #  记忆调试输出 (功能请用 self.memory.* / nav_memory)
    # --------------------------------------------------------
    def _log_memory_debug(self, robot_W, wrote=False, mode=None):
        """周期打印记忆状态, 不封装业务 API。"""
        try:
            n_all = self.memory.count()
            n_walk = self.memory.count(SID_WALKED)
            wr = self.memory.query_walked(robot_W, n_samples=128)
            pos = np.asarray(robot_W, dtype=np.float32).reshape(3)
            print(
                f"\033[96m[memory] frame={self._frame_idx} mode={mode} "
                f"write={'on' if self.memory.write_enabled else 'off'} "
                f"wrote={int(bool(wrote))} "
                f"n={n_all} walked={n_walk} other={n_all - n_walk} "
                f"walked_score={wr.score:.3f} cands={wr.n_candidates} "
                f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})\033[0m"
            )
        except Exception as e:
            print(f"[memory] debug log failed: {e}")

    # --------------------------------------------------------
    #  停止
    # --------------------------------------------------------
    def stop(self):
        """停止 SLAM, 停止内部线程和后端子进程。幂等: 多次调用只清理一次。"""
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5)
            print("[SLAM] Processing thread stopped.")

        if self._sub is not None:
            try:
                self._sub.unregister()
                print("[SLAM] ROS subscriber unregistered.")
            except Exception:
                pass
            self._sub = None

        if self.states is not None:
            self.states.set_mode(Mode.TERMINATED)

        if self.backend_proc is not None:
            self.backend_proc.join(timeout=5)
            if self.backend_proc.is_alive():
                self.backend_proc.terminate()
            print("[SLAM] Backend stopped.")

        if self.viz_proc is not None:
            self.viz_proc.join(timeout=5)
            if self.viz_proc.is_alive():
                self.viz_proc.terminate()
            print("[SLAM] Visualization stopped.")

        print("[SLAM] Stopped.")

    # --------------------------------------------------------
    #  结束 (stop 的别名): 供嵌入宿主的 atexit / 清理逻辑调用
    # --------------------------------------------------------
    def shutdown(self):
        """结束 SLAM, 等价于 stop()。"""
        return self.stop()
