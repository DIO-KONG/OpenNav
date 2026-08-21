# tracer_http_interface

> **Author**: yinzecheng (yonggiee@qq.com)
> **Date**: 2026-07-01

HTTP RESTful API interface for AgileX Tracer mobile robot chassis control.

---

## Overview

This package provides an HTTP bridge to the Tracer ROS stack, allowing remote control and state queries via standard REST endpoints. It runs a FastAPI server inside a `rospy` node, publishing to `/cmd_vel` and `/tracer_light_control` while caching the latest state from `/tracer_status`, `/uart_tracer_status`, and `/odom`.

Architecture:

```
HTTP Client ──REST──▶ FastAPI (uvicorn) ──▶ ROSBridge ──▶ /cmd_vel
                                                        ──▶ /tracer_light_control
                                                        ◀── /tracer_status     (CAN)
                                                        ◀── /uart_tracer_status (UART)
                                                        ◀── /odom
                     ──▶ SafetyWatchdog (auto-stop on timeout)
```

---

## Prerequisites


### Python Dependencies

```bash
pip3 install -r requirements.txt
```

Or install individually:

```bash
pip3 install fastapi uvicorn pydantic
```

---

## Quick Start

### 1. Build

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 2. Launch

Make sure `tracer_base_node` is already running, by bash
```
sudo modprobe gs_usb
```

* first time use tracer-ros package
    ```
    $rosrun tracer_bringup setup_can2usb.bash
    ```
* If not the first time use tracer-ros package(Run this command every time you turn on the power)
    ```
    $rosrun tracer_bringup bringup_can2usb.bash
    ```
```
roslaunch tracer_bringup tracer_robot_base.launch
```
if succeeded, should be
```
process[tracer_base_node-2]: started with pid [1457142]
Start listening to port: can0
[ INFO] [1782908783.786473055]: Using CAN bus to talk with the robot
```
then:
```
# if needed, conda deactivate multiple times
python tracer_http_interface/scripts/tracer_http_server.py
# can use this python /home/agilex/miniconda3/envs/aloha/bin/python


```
or use ros to run it
```bash
roslaunch tracer_http_interface tracer_http_interface.launch
```

Or with custom parameters:

```bash
roslaunch tracer_http_interface tracer_http_interface.launch \
    host:=localhost port:=9090 \
    max_linear_vel:=1.0 max_angular_vel:=2.0 \
    watchdog_timeout:=0.5
```

### 3. API Documentation

Once running, open the auto-generated Swagger UI:

```
http://<host>:<port>/docs
```

---

## API Endpoints

### Motion Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/motion/cmd_vel` | Send velocity command (continuous, needs watchdog or manual stop) |
| `POST` | `/api/motion/timed_move` | **Timed move — auto-stops after duration** |
| `POST` | `/api/motion/stop` | Emergency stop (zero velocity) |
| `GET`  | `/api/motion/safety` | View safety parameters |

#### POST `/api/motion/cmd_vel`

```json
{
  "linear_x": 0.3,
  "angular_z": 0.0
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `linear_x` | float | [-2.0, 2.0] | Linear velocity (m/s), positive = forward |
| `angular_z` | float | [-4.0, 4.0] | Angular velocity (rad/s), positive = left turn |

> **Note**: Velocities are clamped to `max_linear_vel` / `max_angular_vel` (set via launch args).

#### POST `/api/motion/timed_move`

```json
{
  "linear_x": 0.3,
  "angular_z": 0.0,
  "duration_sec": 0.5
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `linear_x` | float | [-2.0, 2.0] | Linear velocity (m/s), positive = forward |
| `angular_z` | float | [-4.0, 4.0] | Angular velocity (rad/s), positive = left turn |
| `duration_sec` | float | (0, 60] | How long to hold the velocity before auto-stop |

> **Common use**: curl one-shot "move forward a bit". Sends velocity, waits `duration_sec`, then auto-stops. No need to manually send stop.

### Light Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/light` | Control front & rear lights |

```json
{
  "enable": true,
  "front_mode": 1,
  "front_custom_value": 128,
  "rear_mode": 0,
  "rear_custom_value": 0
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `enable` | bool | --- | Enable command light control |
| `front_mode` | int | 0-3 | 0=OFF, 1=ON, 2=Breath, 3=Custom |
| `front_custom_value` | int | 0-255 | Custom brightness for front light |
| `rear_mode` | int | 0-3 | Same as front_mode for rear |
| `rear_custom_value` | int | 0-255 | Custom brightness for rear light |

### State Queries

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | CAN chassis status (speed, battery, fault, motor RPM) |
| `GET` | `/api/uart_status` | UART chassis status (motor current/temp, front + rear lights) |
| `GET` | `/api/odom` | Odometry (position, orientation, velocity) |
| `GET` | `/api/health` | Service health & ROS connectivity |

---

## Safety Features

### Velocity Clamping

Commands exceeding `max_linear_vel` / `max_angular_vel` are automatically clamped. The response includes a `clamped` flag indicating whether clamping occurred.

### Safety Watchdog

If no `/api/motion/cmd_vel` request is received within `watchdog_timeout` seconds (default: 1.0s), the watchdog automatically publishes zero velocity to prevent runaway motion. `watchdog_triggered` is reported in `/api/health` and `/api/motion/safety`.

> **Important**: The watchdog only watches `/api/motion/cmd_vel` requests. Direct ROS `/cmd_vel` publishers (e.g., keyboard teleop) are not covered.

---

## Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `host` | `0.0.0.0` | HTTP server listen address |
| `port` | `8080` | HTTP server port |
| `max_linear_vel` | `1.5` | Maximum linear velocity (m/s) |
| `max_angular_vel` | `3.0` | Maximum angular velocity (rad/s) |
| `watchdog_timeout` | `1.0` | Safety watchdog timeout (seconds) |

---

## Usage Examples

> Replace `localhost:8080` with the actual IP and port of your Tracer HTTP server.

### Motion Control --- curl

```bash
# ---- 前进 1 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move   -H "Content-Type: application/json"   -d '{"linear_x": 0.3, "angular_z": 0.0, "duration_sec": 1.0}'

# ---- 后退 1 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move \
  -H "Content-Type: application/json" \
  -d '{"linear_x": -0.3, "angular_z": 0.0, "duration_sec": 1.0}'

# ---- 左转 0.5 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move \
  -H "Content-Type: application/json" \
  -d '{"linear_x": 0.0, "angular_z": 0.5, "duration_sec": 0.5}'

# ---- 右转 0.5 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move \
  -H "Content-Type: application/json" \
  -d '{"linear_x": 0.0, "angular_z": -0.5, "duration_sec": 0.5}'

# ---- 弧线前进 + 左转 1 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move \
  -H "Content-Type: application/json" \
  -d '{"linear_x": 0.2, "angular_z": 0.3, "duration_sec": 1.0}'

# ---- 弧线前进 + 右转 1 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move \
  -H "Content-Type: application/json" \
  -d '{"linear_x": 0.2, "angular_z": -0.3, "duration_sec": 1.0}'

# ---- 原地旋转 1 秒 ----
curl -X POST http://localhost:8080/api/motion/timed_move \
  -H "Content-Type: application/json" \
  -d '{"linear_x": 0.0, "angular_z": 1.0, "duration_sec": 1.0}'

# ---- 连续速度指令 (不会自动停, 需看门狗或手动 stop) ----
curl -X POST http://localhost:8080/api/motion/cmd_vel \
  -H "Content-Type: application/json" \
  -d '{"linear_x": 0.3, "angular_z": 0.0}'

# ---- 紧急停止 ----
curl -X POST http://localhost:8080/api/motion/stop

# ---- 查看安全参数 ----
curl http://localhost:8080/api/motion/safety
```

### Motion Control --- Python (requests)

```python
import requests

BASE_URL = "http://localhost:8080"

def timed_move(linear_x: float, angular_z: float, duration_sec: float):
    """定时移动: 发速度指令, 持续 duration_sec 秒后自动停止"""
    resp = requests.post(
        f"{BASE_URL}/api/motion/timed_move",
        json={
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration_sec": duration_sec,
        },
    )
    print(resp.json())

def send_cmd_vel(linear_x: float, angular_z: float):
    """连续速度指令 (不会自动停止, 需手动 stop 或等待看门狗)"""
    resp = requests.post(
        f"{BASE_URL}/api/motion/cmd_vel",
        json={"linear_x": linear_x, "angular_z": angular_z},
    )
    print(resp.json())

def stop():
    """紧急停止"""
    resp = requests.post(f"{BASE_URL}/api/motion/stop")
    print(resp.json())

def get_safety():
    """查看安全参数"""
    resp = requests.get(f"{BASE_URL}/api/motion/safety")
    print(resp.json())

# ---- 前进 1 秒 ----
timed_move(linear_x=0.3, angular_z=0.0, duration_sec=1.0)

# ---- 后退 1 秒 ----
timed_move(linear_x=-0.3, angular_z=0.0, duration_sec=1.0)

# ---- 左转 0.5 秒 ----
timed_move(linear_x=0.0, angular_z=0.5, duration_sec=0.5)

# ---- 右转 0.5 秒 ----
timed_move(linear_x=0.0, angular_z=-0.5, duration_sec=0.5)

# ---- 弧线前进 + 左转 1 秒 ----
timed_move(linear_x=0.2, angular_z=0.3, duration_sec=1.0)

# ---- 弧线前进 + 右转 1 秒 ----
timed_move(linear_x=0.2, angular_z=-0.3, duration_sec=1.0)

# ---- 原地旋转 1 秒 ----
timed_move(linear_x=0.0, angular_z=1.0, duration_sec=1.0)

# ---- 紧急停止 ----
stop()

# ---- 查看安全参数 ----
get_safety()
```

### Light Control --- curl

```bash
# ---- 前灯常亮 ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "front_mode": 1, "front_custom_value": 0, "rear_mode": 0, "rear_custom_value": 0}'

# ---- 前灯呼吸 (Breath) ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "front_mode": 2, "front_custom_value": 0, "rear_mode": 0, "rear_custom_value": 0}'

# ---- 前灯自定义亮度 128 (Custom) ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "front_mode": 3, "front_custom_value": 128, "rear_mode": 0, "rear_custom_value": 0}'

# ---- 后灯常亮 ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "front_mode": 0, "front_custom_value": 0, "rear_mode": 1, "rear_custom_value": 0}'

# ---- 前后灯都开 ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "front_mode": 1, "front_custom_value": 0, "rear_mode": 1, "rear_custom_value": 0}'

# ---- 前后灯自定义不同亮度 ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "front_mode": 3, "front_custom_value": 200, "rear_mode": 3, "rear_custom_value": 80}'

# ---- 关闭所有灯 ----
curl -X POST http://localhost:8080/api/light \
  -H "Content-Type: application/json" \
  -d '{"enable": false, "front_mode": 0, "front_custom_value": 0, "rear_mode": 0, "rear_custom_value": 0}'
```

### Light Control --- Python (requests)

```python
def set_light(enable: bool, front_mode: int, front_custom_value: int,
              rear_mode: int, rear_custom_value: int):
    """灯光控制"""
    resp = requests.post(
        f"{BASE_URL}/api/light",
        json={
            "enable": enable,
            "front_mode": front_mode,
            "front_custom_value": front_custom_value,
            "rear_mode": rear_mode,
            "rear_custom_value": rear_custom_value,
        },
    )
    print(resp.json())

# ---- 前灯常亮 ----
set_light(True, front_mode=1, front_custom_value=0, rear_mode=0, rear_custom_value=0)

# ---- 前灯呼吸 ----
set_light(True, front_mode=2, front_custom_value=0, rear_mode=0, rear_custom_value=0)

# ---- 前灯自定义亮度 ----
set_light(True, front_mode=3, front_custom_value=128, rear_mode=0, rear_custom_value=0)

# ---- 后灯常亮 ----
set_light(True, front_mode=0, front_custom_value=0, rear_mode=1, rear_custom_value=0)

# ---- 前后灯都开 ----
set_light(True, front_mode=1, front_custom_value=0, rear_mode=1, rear_custom_value=0)

# ---- 关闭所有灯 ----
set_light(False, front_mode=0, front_custom_value=0, rear_mode=0, rear_custom_value=0)
```

### State Queries --- curl

```bash
# ---- CAN 底盘状态 (速度 / 电池 / 故障码 / 电机 RPM) ----
curl http://localhost:8080/api/status

# ---- UART 底盘状态 (电机电流 / 温度 / 前后灯光) ----
curl http://localhost:8080/api/uart_status

# ---- 里程计 (位置 / 朝向 / 速度) ----
curl http://localhost:8080/api/odom

# ---- 服务健康检查 (ROS 连接 / 各话题最后更新时间 / 看门狗) ----
curl http://localhost:8080/api/health
```

### State Queries --- Python (requests)

```python
def get_status():
    """获取 CAN 通道底盘状态"""
    resp = requests.get(f"{BASE_URL}/api/status")
    data = resp.json()
    print(f"  Battery:   {data['battery_voltage']:.2f} V")
    print(f"  Velocity:  linear={data['linear_velocity']:.3f}, angular={data['angular_velocity']:.3f}")
    print(f"  Fault:     {data['fault_code']}")
    print(f"  Odometer:  L={data['left_odomter']:.3f}, R={data['right_odomter']:.3f}")
    for i, m in enumerate(data.get("motor_states", [])):
        print(f"  Motor[{i}]: rpm={m['rpm']:.1f}")
    return data

def get_uart_status():
    """获取 UART 通道底盘状态 (含电机电流/温度)"""
    resp = requests.get(f"{BASE_URL}/api/uart_status")
    data = resp.json()
    print(f"  Battery:   {data['battery_voltage']:.2f} V")
    print(f"  Velocity:  linear={data['linear_velocity']:.3f}, angular={data['angular_velocity']:.3f}")
    print(f"  Fault:     {data['fault_code']}")
    for i, m in enumerate(data.get("motor_states", [])):
        print(f"  Motor[{i}]: rpm={m['rpm']:.1f}, current={m['current']:.2f}A, temp={m['temperature']:.1f}C")
    return data

def get_odom():
    """获取里程计"""
    resp = requests.get(f"{BASE_URL}/api/odom")
    data = resp.json()
    print(f"  Position:  x={data['position_x']:.3f}, y={data['position_y']:.3f}")
    print(f"  Yaw:       {data['yaw_deg']:.1f} deg")
    print(f"  Velocity:  linear={data['linear_velocity']:.3f}, angular={data['angular_velocity']:.3f}")
    return data

def health_check():
    """健康检查"""
    resp = requests.get(f"{BASE_URL}/api/health")
    print(resp.json())

# ---- 查询示例 ----
get_status()
get_uart_status()
get_odom()
health_check()
```

---

## ROS Topics Interface

### Publishers

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command |
| `/tracer_light_control` | `tracer_msgs/TracerLightCmd` | Light control command |

### Subscribers (cache only)

| Topic | Type | Description |
|-------|------|-------------|
| `/tracer_status` | `tracer_msgs/TracerStatus` | CAN chassis state (50 Hz) |
| `/uart_tracer_status` | `tracer_msgs/UartTracerStatus` | UART chassis state (50 Hz) |
| `/odom` | `nav_msgs/Odometry` | Odometry (50 Hz) |

---

## File Layout

```
tracer_http_interface/
├── package.xml
├── CMakeLists.txt
├── requirements.txt
├── README.md
├── launch/
│   └── tracer_http_interface.launch
└── scripts/
    ├── models.py              # Pydantic request/response models
    ├── ros_bridge.py           # ROS topic pub/sub bridge
    └── tracer_http_server.py   # FastAPI application & main entry
```
