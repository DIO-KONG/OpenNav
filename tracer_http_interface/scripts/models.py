"""
Pydantic 数据模型 —— HTTP 请求/响应体定义

基于 tracer_msgs 最新 msg 定义:
  TracerStatus, TracerMotorState, TracerLightState,
  UartTracerStatus, UartTracerMotorState, TracerLightCmd
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class CmdVelRequest(BaseModel):
    """运动控制命令请求"""
    linear_x: float = Field(
        default=0.0,
        description="线速度 (m/s), 正值前进, 负值后退",
        ge=-2.0,
        le=2.0,
    )
    angular_z: float = Field(
        default=0.0,
        description="角速度 (rad/s), 正值左转, 负值右转",
        ge=-4.0,
        le=4.0,
    )


class TimedMoveRequest(BaseModel):
    """定时移动命令: 发速度指令, 到时间自动停止"""
    linear_x: float = Field(
        default=0.0,
        description="线速度 (m/s), 正值前进, 负值后退",
        ge=-2.0,
        le=2.0,
    )
    angular_z: float = Field(
        default=0.0,
        description="角速度 (rad/s), 正值左转, 负值右转",
        ge=-4.0,
        le=4.0,
    )
    duration_sec: float = Field(
        default=1.0,
        description="运动持续时间 (秒), 范围 (0, 60]",
    )


# ---- TracerLightCmd 映射 ----
class LightControlRequest(BaseModel):
    """灯光控制命令请求"""
    enable: bool = Field(
        default=False,
        description="是否启用灯光命令控制",
    )
    front_mode: int = Field(
        default=0,
        description="前灯模式: 0=关, 1=常亮, 2=呼吸, 3=自定义",
        ge=0,
        le=3,
    )
    front_custom_value: int = Field(
        default=0,
        description="前灯自定义亮度 (0-255)",
        ge=0,
        le=255,
    )
    rear_mode: int = Field(
        default=0,
        description="后灯模式: 0=关, 1=常亮, 2=呼吸, 3=自定义",
        ge=0,
        le=3,
    )
    rear_custom_value: int = Field(
        default=0,
        description="后灯自定义亮度 (0-255)",
        ge=0,
        le=255,
    )


# ---- TracerMotorState 映射 (CAN 通道) ----
class MotorState(BaseModel):
    """CAN 通道电机状态 (TracerMotorState)"""
    rpm: float = 0.0


# ---- UartTracerMotorState 映射 (UART 通道) ----
class UartMotorState(BaseModel):
    """UART 通道电机状态 (UartTracerMotorState)"""
    current: float = 0.0
    rpm: float = 0.0
    temperature: float = 0.0


# ---- TracerLightState 映射 ----
class LightState(BaseModel):
    """灯光状态"""
    mode: int = 0          # 0=关, 1=常亮, 2=呼吸, 3=自定义
    custom_value: int = 0  # 0-255


# ---- TracerStatus 映射 ----
class RobotStatus(BaseModel):
    """CAN 通道底盘状态 (映射 /tracer_status)"""
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    base_state: int = 0
    control_mode: int = 0
    fault_code: int = 0       # uint8
    battery_voltage: float = 0.0
    motor_states: List[MotorState] = Field(default_factory=list)   # [2]
    light_control_enabled: bool = False
    front_light: Optional[LightState] = None
    left_odomter: float = 0.0
    right_odomter: float = 0.0
    timestamp: Optional[float] = None


# ---- UartTracerStatus 映射 ----
class UartRobotStatus(BaseModel):
    """UART 通道底盘状态 (映射 /uart_tracer_status)"""
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    base_state: int = 0
    control_mode: int = 0
    fault_code: int = 0       # uint16
    battery_voltage: float = 0.0
    motor_states: List[UartMotorState] = Field(default_factory=list)  # [2]
    light_control_enabled: bool = False
    front_light: Optional[LightState] = None
    rear_light: Optional[LightState] = None
    timestamp: Optional[float] = None


# ---- Odometry 映射 ----
class OdometryInfo(BaseModel):
    """里程计信息 (映射 /odom)"""
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    orientation_w: float = 1.0
    orientation_x: float = 0.0
    orientation_y: float = 0.0
    orientation_z: float = 0.0
    yaw_deg: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    timestamp: Optional[float] = None


# ---- 系统端点 ----
class HealthStatus(BaseModel):
    """健康检查"""
    server_running: bool = True
    ros_master_connected: bool = False
    last_status_time: Optional[float] = None
    last_uart_status_time: Optional[float] = None
    last_odom_time: Optional[float] = None
    watchdog_triggered: bool = False
