#!/usr/bin/env python
"""系统级底层驱动与服务单例包装 (NavServices)：集中管理 ROS 节点、底盘、SLAM、VLM 及后台线程。"""
from __future__ import annotations

import os
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import rospy
from nav_msgs.msg import Odometry

from mast3r_slam_wrapper import Mast3rSlamWrapper
from nav_control import MotionThread, EstopKeyboardThread, PointCloudCache, mock, PC_INTERVAL_DT, MOTION_HZ
from nav_helpers import OdomHolder, get_obstacle_points, load_camera_calibration, CameraCalibration
from nav_vlm import VlmDetector, AsyncVlmWorker
from nav_constants import MOBILE_SAM_CHECKPOINT_PATH, DEBUG_FEATURES
import nav_page


@dataclass
class NavServices:
    """集中管理所有硬件 I/O 与后台子系统的单例集合。"""
    project_root: str
    slam: Mast3rSlamWrapper
    bot: Any
    motion_thread: MotionThread
    vlm_worker: AsyncVlmWorker
    odom_holder: OdomHolder
    pc_cache: PointCloudCache
    estop_keyboard: EstopKeyboardThread
    nav_stop_event: threading.Event
    rate: rospy.Rate
    bbox_calibration: Optional[CameraCalibration] = None
    _stopped: bool = False

    def stop_all(self):
        """统一幂等清理，供 SIGINT 与 atexit 调用。"""
        if self._stopped:
            return
        self._stopped = True
        print("\n[nav_services] 正在停止所有后台线程与服务...")

        try:
            self.estop_keyboard.stop()
        except Exception:
            pass
        try:
            self.motion_thread.stop()
        except Exception:
            pass
        try:
            self.vlm_worker.stop(timeout=1.0)
        except Exception:
            pass
        try:
            self.bot.stop()
        except Exception:
            pass
        self.nav_stop_event.set()
        try:
            self.slam.shutdown()
        except Exception:
            pass
        try:
            rospy.signal_shutdown("services stopped")
        except Exception:
            pass
        print("[nav_services] 所有服务已安全退出。")


def bootstrap_services(vlm_url: str, project_root: Optional[str] = None) -> NavServices:
    """初始化全部硬件、通信接口与后台线程。"""
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 若在 nav_system 目录，回退一级到项目根目录
        if os.path.basename(project_root) == "nav_system":
            project_root = os.path.dirname(project_root)

    # 1. 导入底盘 HTTP API
    tracer_dir = os.path.join(project_root, "tracer_ros", "tracer_http_interface", "scripts")
    if tracer_dir not in sys.path:
        sys.path.insert(0, tracer_dir)
    from tracer_http_interface.scripts.rw_api import TracerRobot

    # 2. ROS 节点初始化
    rospy.init_node("nav_2d", anonymous=True)

    # 3. 启动 SLAM
    slam = Mast3rSlamWrapper(rgb_topic="/camera_f/color/image_raw")
    slam.start()

    # 4. 相机内参标定 (可选)
    bbox_calib = None
    try:
        calib_path = os.path.join(project_root, "masterslam", "config", "intrinsics.yaml")
        if os.path.exists(calib_path):
            bbox_calib = load_camera_calibration(calib_path)
    except Exception as exc:
        print(f"[nav_services][WARN] bbox 标定文件加载失败: {exc}")

    # 5. 里程计监听
    odom_holder = OdomHolder()
    rospy.Subscriber("/odom", Odometry, odom_holder.cb, queue_size=1)

    # 6. 底盘控制与运动线程
    bot = TracerRobot(base_url="http://localhost:8080")
    motion_thread = MotionThread(bot, hz=MOTION_HZ)

    # 7. VLM 异步 Worker
    vlm_detector = VlmDetector(vlm_url=vlm_url, sam_ckpt=MOBILE_SAM_CHECKPOINT_PATH)
    vlm_worker = AsyncVlmWorker(vlm_detector)
    vlm_worker.start()

    # 8. Web 界面与急停键盘监听
    nav_page.mock = mock
    if DEBUG_FEATURES.get("web_ui", True):
        nav_page.start_flask(port=5001)

    estop_keyboard = EstopKeyboardThread()
    estop_keyboard.start()

    # 9. 点云后台缓存线程
    nav_stop_event = threading.Event()
    pc_cache = PointCloudCache(slam, PC_INTERVAL_DT, nav_stop_event, get_obstacle_points)
    pc_cache.start()

    rate = rospy.Rate(10)

    services = NavServices(
        project_root=project_root,
        slam=slam,
        bot=bot,
        motion_thread=motion_thread,
        vlm_worker=vlm_worker,
        odom_holder=odom_holder,
        pc_cache=pc_cache,
        estop_keyboard=estop_keyboard,
        nav_stop_event=nav_stop_event,
        rate=rate,
        bbox_calibration=bbox_calib,
    )

    # 注册 Ctrl+C 与 atexit 清理钩子
    def _sigint_handler(signum, frame):
        services.stop_all()
        os._exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)
    import atexit
    atexit.register(services.stop_all)

    return services
