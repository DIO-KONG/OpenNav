"""
Tracer HTTP API 客户端 —— 基于 requests 封装

提供 TracerRobot 类, 将所有 HTTP 端点封装为 Python 方法,
可直接在脚本或交互式环境中调用, 实现纯 Python 控制。

依赖:
    pip install requests

用法:
    from rw_api import TracerRobot

    bot = TracerRobot(base_url="http://192.168.1.100:8080")

    # 定时移动 (发完自动停)
    bot.timed_move(linear_x=0.3, angular_z=0.0, duration_sec=2.0)

    # 连续速度指令 (需配合 stop 或看门狗)
    bot.cmd_vel(linear_x=0.3, angular_z=0.0)

    # 紧急停止
    bot.stop()

    # 灯光控制
    bot.set_light(front_mode=1, rear_mode=0)

    # 查询状态
    status = bot.get_status()
    odom   = bot.get_odom()
    uart   = bot.get_uart_status()
    health = bot.health_check()
"""
import time
from typing import Optional

import requests


class TracerRobot:
    """
    AgileX Tracer 底盘 HTTP 客户端。

    封装 /api/motion/*, /api/light, /api/status, /api/uart_status,
    /api/odom, /api/health 全部端点。
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 5.0):
        """
        Args:
            base_url: Tracer HTTP 服务器地址, 格式 "http://IP:PORT"
            timeout:  单次 HTTP 请求超时 (秒)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ============================================================
    # 运动控制
    # ============================================================

    def timed_move(
        self,
        linear_x: float,
        angular_z: float = 0.0,
        duration_sec: float = 1.0,
    ) -> dict:
        """
        定时移动: 发送速度指令, duration_sec 秒后自动停止。

        适用场景: 一次性脚本调用、无需持续发送指令的场景。

        Args:
            linear_x:    线速度 (m/s), 正值前进, 负值后退
            angular_z:    角速度 (rad/s), 正值左转, 负值右转
            duration_sec: 持续时间 (秒)

        Returns:
            dict: {"status", "linear_x", "angular_z", "duration_sec", "clamped"}

        Raises:
            RuntimeError: HTTP 请求失败或服务返回错误
        """
        url = f"{self.base_url}/api/motion/timed_move"
        payload = {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration_sec": duration_sec,
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def cmd_vel(self, linear_x: float, angular_z: float = 0.0) -> dict:
        """
        发送连续速度指令。

        注意: 此为持续指令, 底盘会一直保持该速度,
        直到收到下一条指令或看门狗超时自动停车。
        通常需要在一个循环里持续调用, 或配合 stop() 使用。

        Args:
            linear_x: 线速度 (m/s)
            angular_z: 角速度 (rad/s)

        Returns:
            dict: {"status", "linear_x", "angular_z", "clamped"}
        """
        url = f"{self.base_url}/api/motion/cmd_vel"
        payload = {"linear_x": linear_x, "angular_z": angular_z}
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def stop(self) -> dict:
        """
        紧急停止: 立即发送零速指令。

        Returns:
            dict: {"status", "message"}
        """
        url = f"{self.base_url}/api/motion/stop"
        resp = self._session.post(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_safety_params(self) -> dict:
        """
        查询当前安全参数 (限速、看门狗配置)。

        Returns:
            dict: {"max_linear_vel", "max_angular_vel", "watchdog_timeout", "watchdog_triggered"}
        """
        url = f"{self.base_url}/api/motion/safety"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ============================================================
    # 灯光控制
    # ============================================================

    def set_light(
        self,
        enable: bool = True,
        front_mode: int = 0,
        front_custom: int = 0,
        rear_mode: int = 0,
        rear_custom: int = 0,
    ) -> dict:
        """
        控制前后灯光。

        灯光模式:
            0 = 关闭
            1 = 常亮
            2 = 呼吸
            3 = 自定义亮度 (需配合 front_custom / rear_custom)

        Args:
            enable:        是否启用灯光控制
            front_mode:    前灯模式 (0~3)
            front_custom:  前灯自定义亮度 (0~100, 仅 mode=3 时有效)
            rear_mode:     后灯模式 (0~3)
            rear_custom:   后灯自定义亮度 (0~100, 仅 mode=3 时有效)

        Returns:
            dict: {"status", "front_mode", "rear_mode"}
        """
        url = f"{self.base_url}/api/light"
        payload = {
            "enable": enable,
            "front_mode": front_mode,
            "front_custom_value": front_custom,
            "rear_mode": rear_mode,
            "rear_custom_value": rear_custom,
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def light_off(self) -> dict:
        """关闭前后灯光 (快捷方法)"""
        return self.set_light(enable=True, front_mode=0, rear_mode=0)

    def light_on(self, brightness: int = 100) -> dict:
        """前灯常亮 (快捷方法), brightness: 0~100"""
        return self.set_light(
            enable=True,
            front_mode=3,
            front_custom=brightness,
            rear_mode=0,
            rear_custom=0,
        )

    # ============================================================
    # 状态查询
    # ============================================================

    def get_status(self) -> dict:
        """
        查询底盘 CAN 通道状态 (电池电压、速度、故障码、电机 RPM)。

        Returns:
            dict: TracerStatus 完整字段
        """
        url = f"{self.base_url}/api/status"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_uart_status(self) -> dict:
        """
        查询底盘 UART 通道状态 (电机电流、温度、前后灯光状态)。

        Returns:
            dict: UartTracerStatus 完整字段
        """
        url = f"{self.base_url}/api/uart_status"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_odom(self) -> dict:
        """
        查询里程计 (位置 x/y, 朝向 yaw, 线速度)。

        Returns:
            dict: {"position": {"x", "y"}, "orientation": {"yaw"}, "linear_velocity": {"x"}}
        """
        url = f"{self.base_url}/api/odom"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> dict:
        """
        服务健康检查。

        Returns:
            dict: {"server_running", "ros_master_connected",
                   "last_status_time", "last_odom_time",
                   "last_uart_status_time", "watchdog_triggered"}
        """
        url = f"{self.base_url}/api/health"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ============================================================
    # 高级封装: 阻塞式移动
    # ============================================================

    def move_forward(self, speed: float = 0.3, duration: float = 1.0) -> dict:
        """前进 (快捷方法)"""
        return self.timed_move(linear_x=abs(speed), angular_z=0.0, duration_sec=duration)

    def move_backward(self, speed: float = 0.3, duration: float = 1.0) -> dict:
        """后退 (快捷方法)"""
        return self.timed_move(linear_x=-abs(speed), angular_z=0.0, duration_sec=duration)

    def turn_left(self, angular_speed: float = 0.5, duration: float = 1.0) -> dict:
        """原地左转 (快捷方法)"""
        return self.timed_move(linear_x=0.0, angular_z=abs(angular_speed), duration_sec=duration)

    def turn_right(self, angular_speed: float = 0.5, duration: float = 1.0) -> dict:
        """原地右转 (快捷方法)"""
        return self.timed_move(linear_x=0.0, angular_z=-abs(angular_speed), duration_sec=duration)

    def move_arc(self, linear_x: float, angular_z: float, duration: float = 1.0) -> dict:
        """弧线运动 (同时有线速度和角速度)"""
        return self.timed_move(linear_x=linear_x, angular_z=angular_z, duration_sec=duration)


# ============================================================
# CLI 入口 (python rw_api.py)
# ============================================================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Tracer HTTP API CLI")
    parser.add_argument("--host", default="localhost:8080", help="HTTP 服务器地址 (默认 localhost:8080)")
    parser.add_argument("--timeout", type=float, default=5.0, help="请求超时秒数 (默认 5.0)")

    sub = parser.add_subparsers(dest="command", required=True)

    # timed_move
    p = sub.add_parser("timed_move", help="定时移动")
    p.add_argument("--lx", type=float, default=0.0, help="线速度 (m/s)")
    p.add_argument("--az", type=float, default=0.0, help="角速度 (rad/s)")
    p.add_argument("--dur", type=float, default=1.0, help="持续时间 (秒)")

    # cmd_vel
    p = sub.add_parser("cmd_vel", help="发送速度指令")
    p.add_argument("--lx", type=float, default=0.0)
    p.add_argument("--az", type=float, default=0.0)

    # stop
    sub.add_parser("stop", help="紧急停止")

    # light
    p = sub.add_parser("light", help="灯光控制")
    p.add_argument("--front", type=int, default=0, help="前灯模式 (0~3)")
    p.add_argument("--rear", type=int, default=0, help="后灯模式 (0~3)")
    p.add_argument("--fc", type=int, default=0, help="前灯自定义亮度 (0~100)")
    p.add_argument("--rc", type=int, default=0, help="后灯自定义亮度 (0~100)")

    # status
    sub.add_parser("status", help="查询 CAN 状态")

    # uart_status
    sub.add_parser("uart_status", help="查询 UART 状态")

    # odom
    sub.add_parser("odom", help="查询里程计")

    # health
    sub.add_parser("health", help="健康检查")

    # light_on / light_off
    sub.add_parser("light_on", help="前灯常亮")
    sub.add_parser("light_off", help="关闭灯光")

    # 快捷移动
    p = sub.add_parser("forward", help="前进")
    p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--dur", type=float, default=1.0)

    p = sub.add_parser("backward", help="后退")
    p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--dur", type=float, default=1.0)

    p = sub.add_parser("left", help="原地左转")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--dur", type=float, default=1.0)

    p = sub.add_parser("right", help="原地右转")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--dur", type=float, default=1.0)

    args = parser.parse_args()

    bot = TracerRobot(base_url=f"http://{args.host}", timeout=args.timeout)

    try:
        if args.command == "timed_move":
            result = bot.timed_move(args.lx, args.az, args.dur)
        elif args.command == "cmd_vel":
            result = bot.cmd_vel(args.lx, args.az)
        elif args.command == "stop":
            result = bot.stop()
        elif args.command == "light":
            result = bot.set_light(front_mode=args.front, rear_mode=args.rear,
                                  front_custom=args.fc, rear_custom=args.rc)
        elif args.command == "status":
            result = bot.get_status()
        elif args.command == "uart_status":
            result = bot.get_uart_status()
        elif args.command == "odom":
            result = bot.get_odom()
        elif args.command == "health":
            result = bot.health_check()
        elif args.command == "light_on":
            result = bot.light_on()
        elif args.command == "light_off":
            result = bot.light_off()
        elif args.command == "forward":
            result = bot.move_forward(args.speed, args.dur)
        elif args.command == "backward":
            result = bot.move_backward(args.speed, args.dur)
        elif args.command == "left":
            result = bot.turn_left(args.speed, args.dur)
        elif args.command == "right":
            result = bot.turn_right(args.speed, args.dur)
        else:
            parser.print_help()
            raise SystemExit(1)

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] HTTP 请求失败: {e}")
        raise SystemExit(1)
