
# both the base and the rgb use aloha conda env!
## to move base(tracer)
```
sudo modprobe gs_usb
rosrun tracer_bringup bringup_can2usb.bash

roslaunch tracer_bringup tracer_robot_base.launch
/home/agilex/miniconda3/envs/aloha/bin/python tracer_http_interface/scripts/tracer_http_server.py

cd yinzecheng/opennav/
/home/agilex/miniconda3/envs/aloha/bin/python keyboard.py
```

## usage
```python
from tracer_http_interface.scripts.rw_api import TracerRobot

# 初始化 (默认 http://localhost:8080)
bot = TracerRobot(base_url="http://192.168.1.100:8080")

# === 运动控制 ===

# 定时移动: 前进 0.3 m/s, 持续 2 秒后自动停止
bot.timed_move(linear_x=0.3, angular_z=0.0, duration_sec=2.0)

# 后退 0.5 秒
bot.timed_move(linear_x=-0.3, angular_z=0.0, duration_sec=0.5)

# 原地左转 1 秒
bot.timed_move(linear_x=0.0, angular_z=0.5, duration_sec=1.0)

# 弧线前进 (同时有线速度和角速度)
bot.move_arc(linear_x=0.2, angular_z=0.3, duration=1.5)

# 快捷方法
bot.move_forward(speed=0.3, duration=2.0)   # 前进
bot.move_backward(speed=0.3, duration=1.0)   # 后退
bot.turn_left(speed=0.5, duration=0.5)      # 左转
bot.turn_right(speed=0.5, duration=0.5)     # 右转

# 连续速度指令 (需手动停止或依赖看门狗)
bot.cmd_vel(linear_x=0.3, angular_z=0.0)
# ... 做其他事 ...
bot.stop()

# 紧急停止
bot.stop()

# === 灯光控制 ===
bot.set_light(front_mode=1, rear_mode=0)             # 前灯常亮
bot.set_light(front_mode=2, rear_mode=0)             # 前灯呼吸
bot.set_light(front_mode=3, front_custom=128, rear_mode=0)  # 前灯自定义亮度
bot.light_on(brightness=100)                          # 前灯常亮 (快捷)
bot.light_off()                                       # 关闭所有灯

# === 状态查询 ===
status = bot.get_status()        # CAN 通道: 电池/速度/故障/RPM
uart   = bot.get_uart_status()  # UART 通道: 电流/温度/灯光
odom   = bot.get_odom()         # 里程计: 位置/朝向/速度
health = bot.health_check()     # 健康检查
```

### 作为 CLI 使用

```bash
# 定时移动
python rw_api.py --host localhost:8080 timed_move --lx 0.3 --az 0.0 --dur 2.0

# 快捷移动
python rw_api.py --host localhost:8080 forward --speed 0.3 --dur 2.0
python rw_api.py --host localhost:8080 backward --speed 0.3 --dur 1.0
python rw_api.py --host localhost:8080 left --speed 0.5 --dur 0.5
python rw_api.py --host localhost:8080 right --speed 0.5 --dur 0.5

# 紧急停止
python rw_api.py --host localhost:8080 stop

# 灯光
python rw_api.py --host localhost:8080 light --front 1 --rear 0
python rw_api.py --host localhost:8080 light_on
python rw_api.py --host localhost:8080 light_off

# 状态查询
python rw_api.py --host localhost:8080 status
python rw_api.py --host localhost:8080 uart_status
python rw_api.py --host localhost:8080 odom
python rw_api.py --host localhost:8080 health
```




## RGB api
### before run, start ros server
```
source ~/cobot_magic/camera_ws/devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

### keyboard control
python keyboard.py


