#!/usr/bin/env python3
"""
RGBRosConnector — 从 ROS 话题获取当前 RGB 帧 (numpy ndarray)
===============================================================

用法:
    from viewer.rgb_ros_connector import RGBRosConnector

    conn = RGBRosConnector(topic="/camera_f/color/image_raw")
    conn.wait_for_frame(timeout=5)   # 阻塞等首帧，最多5秒
    frame = conn.get_frame()         # numpy ndarray, BGR, (H,W,3)
"""

import threading
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RGBRosConnector:
    """从 ROS Image 话题获取当前 RGB 帧，返回 numpy ndarray (BGR)。"""

    def __init__(self, topic="/camera_f/color/image_raw", node_name="rgb_connector"):
        self._bridge = CvBridge()
        self._frame = None
        self._lock = threading.Lock()
        self._first_frame_event = threading.Event()

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        rospy.Subscriber(topic, Image, self._cb, queue_size=1)

    def _cb(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._frame = img.copy()
            self._first_frame_event.set()
        except Exception:
            pass

    def wait_for_frame(self, timeout=10.0):
        """阻塞等待收到第一帧。超时返回 False，成功返回 True。"""
        return self._first_frame_event.wait(timeout)

    def get_frame(self):
        """返回当前最新 RGB 帧 (numpy ndarray, BGR, dtype=uint8)。未收到帧时返回 None。"""
        with self._lock:
            return self._frame


if __name__ == "__main__":
    conn = RGBRosConnector(topic="/camera_f/color/image_raw")
    print("等待首帧...")
    ok = conn.wait_for_frame(timeout=5)
    if not ok:
        print("超时，5秒内没收到帧。确认摄像头节点已启动。")
    else:
        f = conn.get_frame()
        print(f"OK  shape={f.shape}  dtype={f.dtype}  range=[{f.min()}, {f.max()}]")
