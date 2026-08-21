"""
ROS 桥接层 —— 封装 ROS 话题的发布与订阅缓存

发布话题:
    /cmd_vel               (geometry_msgs/Twist)         运动控制
    /tracer_light_control  (tracer_msgs/TracerLightCmd)   灯光控制

订阅话题 (缓存最新值):
    /tracer_status         (tracer_msgs/TracerStatus)     CAN 通道底盘状态
    /uart_tracer_status    (tracer_msgs/UartTracerStatus) UART 通道底盘状态
    /odom                  (nav_msgs/Odometry)            里程计
"""
import math
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tracer_msgs.msg import TracerStatus, UartTracerStatus, TracerLightCmd


class ROSBridge:
    def __init__(self):
        self._lock = threading.Lock()

        # ---- 缓存最新数据 ----
        self._latest_status = None
        self._latest_status_time = None
        self._latest_uart_status = None
        self._latest_uart_status_time = None
        self._latest_odom = None
        self._latest_odom_time = None

        # ---- 发布者 ----
        self._cmd_vel_pub = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=1
        )
        self._light_cmd_pub = rospy.Publisher(
            "/tracer_light_control", TracerLightCmd, queue_size=1
        )

        # ---- 订阅者 ----
        rospy.Subscriber("/tracer_status", TracerStatus, self._on_status)
        rospy.Subscriber("/uart_tracer_status", UartTracerStatus, self._on_uart_status)
        rospy.Subscriber("/odom", Odometry, self._on_odom)

        rospy.loginfo("[TracerHTTP] ROS bridge initialized")

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def _on_status(self, msg):
        with self._lock:
            self._latest_status = msg
            self._latest_status_time = time.time()

    def _on_uart_status(self, msg):
        with self._lock:
            self._latest_uart_status = msg
            self._latest_uart_status_time = time.time()

    def _on_odom(self, msg):
        with self._lock:
            self._latest_odom = msg
            self._latest_odom_time = time.time()

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    def publish_cmd_vel(self, linear_x, angular_z):
        """发布速度命令"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)

    def publish_light_cmd(self, enable, front_mode, front_custom,
                          rear_mode, rear_custom):
        """发布灯光命令"""
        msg = TracerLightCmd()
        msg.enable_cmd_light_control = enable
        msg.front_mode = front_mode
        msg.front_custom_value = front_custom
        msg.rear_mode = rear_mode
        msg.rear_custom_value = rear_custom
        self._light_cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # 查询 (线程安全)
    # ------------------------------------------------------------------
    def get_status(self):
        """获取最新 TracerStatus (CAN 通道)"""
        with self._lock:
            return self._latest_status, self._latest_status_time

    def get_uart_status(self):
        """获取最新 UartTracerStatus (UART 通道)"""
        with self._lock:
            return self._latest_uart_status, self._latest_uart_status_time

    def get_odom(self):
        """获取最新 Odometry"""
        with self._lock:
            return self._latest_odom, self._latest_odom_time

    @property
    def is_master_connected(self):
        """检查 ROS master 是否可达"""
        try:
            rospy.get_master().getPid()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def quat_to_yaw_deg(x, y, z, w):
        """四元数 -> 偏航角 (度)"""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))
