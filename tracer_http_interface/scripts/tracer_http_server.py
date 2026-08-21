#!/usr/bin/env python3
"""
Tracer HTTP Interface
=====================
FastAPI RESTful 服务, 将 HTTP 请求桥接到 ROS 话题, 实现对 Tracer 底盘的远程控制。

启动方式:
    rosrun tracer_http_interface tracer_http_server.py  _host:=0.0.0.0 _port:=8080

    # 或通过 launch 文件:
    roslaunch tracer_http_interface tracer_http_interface.launch

接口文档:
    启动后访问 http://<host>:<port>/docs 查看 Swagger UI
"""
import asyncio
import sys
import os
import threading

# 确保 scripts 目录在 Python 搜索路径中, 支持 rosrun 和直接运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import rospy
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ros_bridge import ROSBridge
from models import (
    CmdVelRequest,
    TimedMoveRequest,
    LightControlRequest,
    RobotStatus,
    MotorState,
    UartRobotStatus,
    UartMotorState,
    LightState,
    OdometryInfo,
    HealthStatus,
)


# ============================================================
# 初始化 ROS (必须在创建 Publisher / Subscriber 之前)
# ============================================================
rospy.init_node("tracer_http_interface", anonymous=False)

# ---- 读取参数 ----
_HOST = rospy.get_param("~host", "0.0.0.0")
_PORT = rospy.get_param("~port", 8080)
_MAX_LINEAR_VEL = rospy.get_param("~max_linear_vel", 1.5)
_MAX_ANGULAR_VEL = rospy.get_param("~max_angular_vel", 3.0)
# current_time-last_time>3, invoked
_WATCHDOG_TIMEOUT = rospy.get_param("~watchdog_timeout", 2.0)


# ============================================================
# 看门狗定时器 (安全机制: 超时未发指令自动停车)
# ============================================================
class SafetyWatchdog:
    """超时自动发送零速命令, 防止无人操控时持续运动"""

    def __init__(self, bridge, timeout):
        self._bridge = bridge
        self._timeout = timeout
        self._last_cmd_time = rospy.Time.now()
        self._watchdog_triggered = False
        self._timer = rospy.Timer(
            rospy.Duration(0.1), self._check
        )

    def touch(self):
        """更新最后一次收到命令的时间"""
        self._last_cmd_time = rospy.Time.now()
        self._watchdog_triggered = False

    def _check(self, event):
        elapsed = (rospy.Time.now() - self._last_cmd_time).to_sec()
        if elapsed > self._timeout:
            if not self._watchdog_triggered:
                self._bridge.publish_cmd_vel(0.0, 0.0)
                self._watchdog_triggered = True
                rospy.logwarn_throttle(
                    5, "[Safety] Watchdog triggered --- zero velocity sent"
                )

    @property
    def triggered(self):
        return self._watchdog_triggered


# ============================================================
# 创建桥接和看门狗
# ============================================================
bridge = ROSBridge()
watchdog = SafetyWatchdog(bridge, _WATCHDOG_TIMEOUT)


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="Tracer HTTP Interface",
    description="""
RESTful HTTP API for AgileX Tracer mobile robot chassis control.

**控制端点**: `/api/motion/*`, `/api/light`
**状态端点**: `/api/status`, `/api/uart_status`, `/api/odom`
**系统端点**: `/api/health`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 辅助函数
# ============================================================
def _clamp_velocity(linear_x: float, angular_z: float):
    """将速度限制在安全范围内"""
    linear_x = max(-_MAX_LINEAR_VEL, min(_MAX_LINEAR_VEL, linear_x))
    angular_z = max(-_MAX_ANGULAR_VEL, min(_MAX_ANGULAR_VEL, angular_z))
    return linear_x, angular_z


def _build_robot_status(msg, ts):
    """将 TracerStatus ROS 消息转为 RobotStatus API 响应"""
    motors = []
    if hasattr(msg, "motor_states"):
        for m in msg.motor_states:
            motors.append(MotorState(rpm=m.rpm))

    front_light = None
    if hasattr(msg, "front_light_state") and msg.front_light_state is not None:
        front_light = LightState(
            mode=msg.front_light_state.mode,
            custom_value=msg.front_light_state.custom_value,
        )

    return RobotStatus(
        linear_velocity=msg.linear_velocity,
        angular_velocity=msg.angular_velocity,
        base_state=msg.base_state,
        control_mode=msg.control_mode,
        fault_code=msg.fault_code,
        battery_voltage=msg.battery_voltage,
        motor_states=motors,
        light_control_enabled=getattr(msg, "light_control_enabled", False),
        front_light=front_light,
        left_odomter=getattr(msg, "left_odomter", 0.0),
        right_odomter=getattr(msg, "right_odomter", 0.0),
        timestamp=ts,
    )


def _build_uart_status(msg, ts):
    """将 UartTracerStatus ROS 消息转为 UartRobotStatus API 响应"""
    motors = []
    if hasattr(msg, "motor_states"):
        for m in msg.motor_states:
            motors.append(UartMotorState(
                current=m.current,
                rpm=m.rpm,
                temperature=m.temperature,
            ))

    front_light = None
    if hasattr(msg, "front_light_state") and msg.front_light_state is not None:
        front_light = LightState(
            mode=msg.front_light_state.mode,
            custom_value=msg.front_light_state.custom_value,
        )

    rear_light = None
    if hasattr(msg, "rear_light_state") and msg.rear_light_state is not None:
        rear_light = LightState(
            mode=msg.rear_light_state.mode,
            custom_value=msg.rear_light_state.custom_value,
        )

    return UartRobotStatus(
        linear_velocity=msg.linear_velocity,
        angular_velocity=msg.angular_velocity,
        base_state=msg.base_state,
        control_mode=msg.control_mode,
        fault_code=msg.fault_code,
        battery_voltage=msg.battery_voltage,
        motor_states=motors,
        light_control_enabled=getattr(msg, "light_control_enabled", False),
        front_light=front_light,
        rear_light=rear_light,
        timestamp=ts,
    )


def _build_odom_info(msg, ts):
    """将 Odometry ROS 消息转为 OdometryInfo API 响应"""
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    t = msg.twist.twist

    yaw_deg = ROSBridge.quat_to_yaw_deg(o.x, o.y, o.z, o.w)

    return OdometryInfo(
        position_x=p.x,
        position_y=p.y,
        position_z=p.z,
        orientation_w=o.w,
        orientation_x=o.x,
        orientation_y=o.y,
        orientation_z=o.z,
        yaw_deg=yaw_deg,
        linear_velocity=t.linear.x,
        angular_velocity=t.angular.z,
        timestamp=ts,
    )


# ============================================================
# 端点: 运动控制
# ============================================================
@app.post("/api/motion/cmd_vel", summary="发送速度命令")
async def send_cmd_vel(req: CmdVelRequest):
    """
    发送速度指令到 Tracer 底盘。

    注意: 速度会被限制在安全范围内 (max_linear_vel / max_angular_vel)。
    底层默认以 50Hz 发布, 建议客户端按 20-50Hz 调用。
    """
    lx, az = _clamp_velocity(req.linear_x, req.angular_z)
    bridge.publish_cmd_vel(lx, az)
    watchdog.touch()
    return {
        "status": "ok",
        "linear_x": lx,
        "angular_z": az,
        "clamped": (lx != req.linear_x or az != req.angular_z),
    }


@app.post("/api/motion/stop", summary="紧急停止")
async def emergency_stop():
    """立即停止所有运动 (发送零速指令)"""
    bridge.publish_cmd_vel(0.0, 0.0)
    watchdog.touch()
    return {"status": "ok", "message": "Stop command sent"}


@app.post("/api/motion/timed_move", summary="定时移动 (自动停止)")
async def timed_move(req: TimedMoveRequest):
    """
    发送指定速度指令，持续 duration_sec 秒后自动停止。

    适用于 curl 一次性调用场景:
    - 前进 0.5s:  `{"linear_x": 0.3, "angular_z": 0.0, "duration_sec": 0.5}`
    - 后退 1s:   `{"linear_x": -0.3, "angular_z": 0.0, "duration_sec": 1.0}`
    - 左转 0.5s: `{"linear_x": 0.0, "angular_z": 0.5, "duration_sec": 0.5}`
    """
    if req.duration_sec <= 0:
        raise HTTPException(status_code=400, detail="duration_sec must be > 0")
    if req.duration_sec > 60:
        raise HTTPException(status_code=400, detail="duration_sec must be <= 60")

    lx, az = _clamp_velocity(req.linear_x, req.angular_z)

    # duration 期间以 20Hz 持续发布 cmd_vel, 否则底盘收不到后续指令会自己停
    rate = 0.05  # 20Hz
    elapsed = 0.0
    while elapsed < req.duration_sec:
        bridge.publish_cmd_vel(lx, az)
        watchdog.touch()
        step = min(rate, req.duration_sec - elapsed)
        await asyncio.sleep(step)
        elapsed += step

    bridge.publish_cmd_vel(0.0, 0.0)
    return {
        "status": "ok",
        "linear_x": lx,
        "angular_z": az,
        "duration_sec": req.duration_sec,
        "clamped": (lx != req.linear_x or az != req.angular_z),
    }


@app.get("/api/motion/safety", summary="查看安全参数")
async def get_safety_params():
    """返回当前限速和看门狗配置"""
    return {
        "max_linear_vel": _MAX_LINEAR_VEL,
        "max_angular_vel": _MAX_ANGULAR_VEL,
        "watchdog_timeout": _WATCHDOG_TIMEOUT,
        "watchdog_triggered": watchdog.triggered,
    }


# ============================================================
# 端点: 灯光控制
# ============================================================
@app.post("/api/light", summary="灯光控制")
async def control_light(req: LightControlRequest):
    """控制 Tracer 底盘前后灯光"""
    bridge.publish_light_cmd(
        enable=req.enable,
        front_mode=req.front_mode,
        front_custom=req.front_custom_value,
        rear_mode=req.rear_mode,
        rear_custom=req.rear_custom_value,
    )
    return {"status": "ok", "light": req.model_dump()}


# ============================================================
# 端点: 状态查询
# ============================================================
@app.get("/api/status", summary="获取底盘状态 (CAN)")
async def get_status():
    """获取最新 CAN 通道底盘状态 (线速度、角速度、电池、故障码、电机 RPM 等)"""
    msg, ts = bridge.get_status()
    if msg is None:
        raise HTTPException(status_code=503, detail="No CAN status data received yet")
    return _build_robot_status(msg, ts)


@app.get("/api/uart_status", summary="获取底盘状态 (UART)")
async def get_uart_status():
    """获取最新 UART 通道底盘状态 (含电机电流/温度、前后灯光)"""
    msg, ts = bridge.get_uart_status()
    if msg is None:
        raise HTTPException(status_code=503, detail="No UART status data received yet")
    return _build_uart_status(msg, ts)


@app.get("/api/odom", summary="获取里程计")
async def get_odom():
    """获取最新里程计 (位置、朝向、速度)"""
    msg, ts = bridge.get_odom()
    if msg is None:
        raise HTTPException(status_code=503, detail="No odometry data received yet")
    return _build_odom_info(msg, ts)


# ============================================================
# 端点: 健康检查
# ============================================================
@app.get("/api/health", summary="健康检查")
async def health_check():
    """检查服务及 ROS 连接状态"""
    return HealthStatus(
        server_running=True,
        ros_master_connected=bridge.is_master_connected,
        last_status_time=bridge.get_status()[1],
        last_uart_status_time=bridge.get_uart_status()[1],
        last_odom_time=bridge.get_odom()[1],
        watchdog_triggered=watchdog.triggered,
    )


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # rospy 需要一个运行中的事件循环来刷新 TCP 连接和发送队列,
    # 否则 publish() 可能只入队不发出。启动 daemon 线程来 spin。
    _spin_thread = threading.Thread(target=rospy.spin, daemon=True)
    _spin_thread.start()

    rospy.loginfo(
        f"[TracerHTTP] Starting server on {_HOST}:{_PORT}"
    )
    rospy.loginfo(f"[TracerHTTP] Safety limits: "
                   f"linear={_MAX_LINEAR_VEL}m/s, "
                   f"angular={_MAX_ANGULAR_VEL}rad/s, "
                   f"watchdog={_WATCHDOG_TIMEOUT}s")
    rospy.loginfo(f"[TracerHTTP] API docs: http://{_HOST}:{_PORT}/docs")

    uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")
