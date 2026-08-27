#!/usr/bin/env python
"""nav_constants — 统一常量真相源 (Single Source of Truth)

按功能域组织：
  1. 机器人物理尺寸与核心安全净空
  2. 路径规划 (RRT*) 与航路点到达容差
  3. 纯追踪控制器 (Pure Pursuit) 与大角转向预回退
  4. 侵入脱困参数 (Escape)
  5. 视觉目标检测与终调锁定 (Final Adjust / VLM)
  6. 点云高度带与位姿平滑
  7. 底盘控制与建图扫描 (Motion / Scan / Reloc)
  8. 语义拓扑记忆 (nav_memory)
  9. 可视化与调试配置 (Web UI / Debug)
"""

# ============================================================
#  1. 物理尺寸与安全净空 (基础参数，联动全局)
# ============================================================
ROBOT_RADIUS = 0.35                       # 机器人本体半径 (m)
RRT_OBSTACLE_INFLATE = ROBOT_RADIUS + 0.05  # RRT 规划障碍物膨胀半径 (≈0.40m)
NAV_CLEARANCE = RRT_OBSTACLE_INFLATE      # 全局唯一安全净空：导航点/连接段/剩余路径合法性检验基准


# ============================================================
#  2. 路径规划 (RRT*) 与到达判定
# ============================================================
PLAN_BOX_PAD = 3.0                        # 局部规划窗口余量 (起点到终点包围盒双向外扩)
RRT_PLAN_TIME = 0.5                       # 单次 RRT* 规划最大超时时间 (s)
GOAL_RETRIEVE_MAX = 2.0                   # 语义目标点周围候选安全站位的最大检索半径 (m)

# 目标到达容差判定 (m)
PATROL_ARRIVE_EPS = ROBOT_RADIUS * 0.3    # 普通巡逻/航路点到达容差
FRONTIER_ARRIVE_EPS = ROBOT_RADIUS * 0.8  # Frontier 探索点到达容差


# ============================================================
#  3. 纯追踪控制 (Pure Pursuit) 与大角转向预回退 (Preturn)
# ============================================================
YAW_TURN_SIGN = -1.0                      # 底盘左转为正(+omega)，SLAM世界系右转为正，故系数为 -1.0
FOLLOW_LIN_MAX = 0.10                     # 巡航最大前进线速度 (m/s)
FOLLOW_ANG_MAX = 0.10                     # 巡航最大转向角速度 (rad/s)
FOLLOW_LIN_FLOOR = 0.05                   # 接近路点时的线速度下限 (m/s)，防线速度归零卡死转向
FOLLOW_LIN_GAIN = 0.30                    # 目标逼近线速度减速比例增益

FOLLOW_LOOKAHEAD = 0.50                   # 纯追踪前视距离 (m)
FOLLOW_WAYPOINT_THRESHOLD = 0.30          # 航路点切换阈值 (m)：当前距路点小于此值时推进到下一点
FOLLOW_NEAR_DIST = 0.60                   # 近距离直行阈值 (m)：小于此距离跳过前视追踪直接对准航向

FOLLOW_YAW_EMA = 0.30                     # 位姿航向输入平滑系数 (EMA)
FOLLOW_OUT_EMA = 0.40                     # 转向角速度输出平滑系数 (EMA)
FOLLOW_DEADBAND = 0.087                   # 角速度输出死区 (rad, ≈5°)，消除微小航向抖动
FOLLOW_STRAIGHT_ALPHA = 0.175             # 直行/转弯模式判定阈值 (rad, ≈10°)
FOLLOW_SLEW = 0.08                        # 角速度变化率限幅 (rad/帧)，平抑跳变阶跃
FOLLOW_INPLACE_ALPHA = 0.70               # 原地自旋阈值 (rad, ≈40°)：航向偏差过大时先自旋对正
FOLLOW_TURN_GAIN = 0.50                   # 原地自旋比例增益

# 近墙大角度转向前直线回退参数
PRETURN_MIN_TURN_DEG = 5.0                # 触发预转向评估的最小航向转角偏差 (°)
PRETURN_WALL_RAY_MIN_DEG = -35            # 墙体探测射线起始扇区角 (°)
PRETURN_WALL_RAY_MAX_DEG = 35             # 墙体探测射线结束扇区角 (°)
PRETURN_WALL_RAY_STEP_DEG = 1             # 射线角分辨率 (°)
PRETURN_WALL_RAY_MAX_DIST = 0.80          # 射线最大探测距离 (m)
PRETURN_WALL_MIN_HIT_RATIO = 0.65         # 判定为墙体的最小射线击中率 (65%)
PRETURN_WALL_MAX_MEDIAN_DIST = 0.60       # 墙体距离中位数上限 (m)
PRETURN_WALL_MAX_LINE_RESIDUAL = 0.10     # 墙体拟合直线残差容限 (m)
PRETURN_RETREAT_DISTANCE = 0.30           # 触发预转向后的直线后退距离 (m)
PRETURN_RETREAT_SPEED = FOLLOW_LIN_MAX    # 预转向后退速度 (m/s)
PRETURN_RETREAT_TIMEOUT = 4.0             # 预转向最大动作超时时间 (s)


# ============================================================
#  4. 碰撞侵入与受控脱困 (Escape)
# ============================================================
ESCAPE_PATH0_REVERSE_MIN_ANGLE_DEG = 90.0 # 目标位于后半球时倒车脱困切入角 (°)
ESCAPE_PATH0_ALIGN_EPS_DEG = 5.0          # 脱困直线原地对准容差角 (°)
ESCAPE_PATH0_ALIGN_ANG_MAX = 0.12         # 脱困对准最大自旋角速度 (rad/s)
ESCAPE_PATH0_REVERSE_SPEED = 0.05         # 脱困倒车巡航速度 (m/s)
ESCAPE_PATH0_REVERSE_MIN_SPEED = 0.03     # 脱困倒车最小速度 (m/s)
ESCAPE_PATH0_ARRIVE_EPS = 0.08            # 脱困第一航路点到达判定容差 (m)

# 脱困后 Odom 保持窗口 (防 SLAM 位姿回弹)
POST_ESCAPE_HOLD = 1.5                    # Odom 保持窗口上限超时时间 (s)
POST_ESCAPE_CONVERGE_DIST = 0.10          # SLAM 位姿与 Odom 追平距离阈值 (m)
POST_ESCAPE_CONVERGE_YAW = 0.10           # SLAM 航向与 Odom 追平角度阈值 (rad)


# ============================================================
#  5. 视觉目标检测与终调锁定 (VLM / Final Adjust)
# ============================================================
OBJ_DET_PERIOD = 2.0                      # 巡逻期间目标检测周期 (s)
AUTO_VLM_RETRY_PERIOD = 3.0               # VLM 方向问询失败时的重试间隔 (s)
VLM_URL_DEFAULT = "http://localhost:8222/v1"
VLM_DETECT_TARGET_DEFAULT = "white plastic stool" # 默认视觉目标检测对象

# 终调视觉伺服 (Final Adjust)
FINAL_ADJUST_TIME = 20.0                  # 终调最小持续观察时间 (s)
FINAL_ADJUST_MIN_FRAMES = 6               # 终调锁定最少累积有效检测帧数
FINAL_ADJUST_DET_PERIOD = 1.0             # 终调期间高频目标检测周期 (s)
FINAL_ADJUST_SMOOTH_ALPHA = 0.50          # EMA 目标坐标平滑系数
FINAL_ADJUST_REJECT_DIST = 5.0            # EMA 离群跳变测量剔除阈值 (m)
FINAL_ADJUST_LOCK_DIST = 1.50             # 允许锁定并进入最终跟踪的距离阈值 (m)
FINAL_ADJUST_FORCE_TIME = 60.0            # 终调最大强制锁定超时时间 (s)

# VLM Prompt 模板
AUTO_VLM_DIR_PROMPT = f"""You are controlling a mobile robot exploring an indoor space to find {VLM_DETECT_TARGET_DEFAULT}.
Based on the current camera view, decide which horizontal direction the robot should explore next.
You must select eangularword from: left, right, front.
- "left": the promising area is in the left part of the view
- "right": the promising area is in the right part of the view
- "front": the promising area is straight ahead / center
Reply with ONLY the single word (left/right/front), no explanation."""

VLM_DETECTION_PROMPT = f"""Detect {VLM_DETECT_TARGET_DEFAULT} and identify their reference designators (reference numbers), and output the results in the following JSON format:
```json
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "type_of_component", "sub_label": "Reference_designator"}},
  ...
]
```"""

VLM_PRESENCE_PROMPT = f"""Look at the current camera image. Is there {VLM_DETECT_TARGET_DEFAULT} present in the scene?
Answer with only the single word 'yes' or 'no', no explanation."""

MOBILE_SAM_CHECKPOINT_PATH = '/home/agilex/yinzecheng/opennav/mobile_sam/mobile_sam.pt'


# ============================================================
#  6. 点云高度带与位姿平滑
# ============================================================
# 障碍物点云垂直截面截取范围 (相对于当前导航平面 Y)
OBS_Y_ABOVE = 0.20                        # 平面上方截取高度 (m)
OBS_Y_BELOW = 0.50                        # 平面下方截取深度 (m)

# 关键帧锚定位姿平滑系数
NAV_SMOOTH_POS = 0.30                     # 位置 (x, z) 关键帧修正软收敛系数
NAV_SMOOTH_YAW = 0.30                     # 偏航 (yaw) 关键帧修正软收敛系数
ODOM_YAW_OFFSET = 0.0                     # Odom 坐标系到 SLAM 世界系的固定角度偏置 (rad)


# ============================================================
#  7. 底盘控制与建图扫描 (Motion / Scan / Reloc)
# ============================================================
MOTION_HZ = 20                            # 底盘速度控制发布频率 (Hz)
SCAN_ANGULAR = 0.10                       # 原地建图扫描与重定位自旋角速度 (rad/s)
BOOT_SCAN_ENABLED = True                  # 是否允许开局/重定位恢复后执行 360° 建图扫描

RELOC_STOP_TIME = 0.5                     # 重定位恢复机 L0 停车稳定时长 (s)
RELOC_BACK_TIME = 3.0                     # 重定位恢复机 L1 直线后退时长 (s)


# ============================================================
#  8. 语义拓扑记忆 (nav_memory)
# ============================================================
MEM_ENABLE = True                         # 记忆系统总开关
MEM_PERSIST_ENABLE = True                 # 记忆数据持久化保存开关
MEM_DEBUG_EVERY = 30                      # 记忆状态调试日志输出周期 (帧)

# 行走足迹记忆 (Walked Memory)
WALK_MIN_TRANS = 0.05                     # 足迹点记录最小位移增量 (m)
assert WALK_MIN_TRANS < PATROL_ARRIVE_EPS, "WALK_MIN_TRANS 必须小于 PATROL_ARRIVE_EPS"

MEM_WALKED_ENABLE = True
MEM_WALKED_RADIUS = ROBOT_RADIUS + 0.40   # 单个足迹高斯核覆盖半径 (m)
MEM_WALKED_MERGE_RADIUS = 0.08            # 相邻足迹圆合并距离阈值 (m)
MEM_WALKED_MIN_DT = 1.0                   # 足迹最小采样时间间隔 (s)
MEM_WALKED_WEIGHT = 0.0                   # 足迹高斯权重
MEM_WALKED_EDGE_MARGIN = ROBOT_RADIUS     # 足迹边缘安全裕度 (m)
MEM_WALKED_RADIUS_MIN = 0.05              # 足迹世界坐标半径下限 (m)
MEM_WALKED_RADIUS_MAX = 1.50              # 足迹世界坐标半径上限 (m)

MEM_ANCHOR_SCALE_RATIO_MIN = 0.50         # 锚定尺度变化合法下限比率
MEM_ANCHOR_SCALE_RATIO_MAX = 2.00         # 锚定尺度变化合法上限比率

# 物体语义记忆 (Object Memory，预留)
MEM_OBJECT_RADIUS = ROBOT_RADIUS * 0.10
MEM_OBJECT_MERGE_RADIUS = ROBOT_RADIUS * 1.10
MEM_OBJECT_CLUSTER_EPS = ROBOT_RADIUS * 1.40
MEM_OBJECT_WEIGHT = 1.0

# 记忆检索 (Query)
MEM_QUERY_RADIUS = ROBOT_RADIUS * 6.0
MEM_QUERY_N_SAMPLES = 512
MEM_QUERY_KAPPA = 1.0

# 边界栅格拓扑探索 (Frontier Grid)
MEM_FRONTIER_ROBOT_RADIUS = RRT_OBSTACLE_INFLATE
MEM_FRONTIER_RESOLUTION = 0.05            # 局部栅格分辨率 (m)
MEM_FRONTIER_WIN_RADIUS = 20.0            # 局部栅格探索窗口半径 (m)
MEM_FRONTIER_OCC_MIN_POINTS = 3           # 栅格判定为障碍的最小命中点数
MEM_FRONTIER_KAPPA = MEM_QUERY_KAPPA
MEM_FRONTIER_SHAPE = "disk"               # 记忆栅格化形状 (disk 扇形圆盘)
MEM_FRONTIER_INFLATE_TYPE = "circle"      # 障碍膨胀方式 (circle 圆形膨胀)
MEM_FRONTIER_SIGMA_WALKED = 0.40          # 局部检索足迹高斯外扩容限
MEM_FRONTIER_GLOBAL_RESOLUTION = 0.20     # 全局栅格分辨率 (m)
MEM_FRONTIER_MAX_OBSTACLE_POINTS = 100000 # 最大参与计算障碍点数

# 语义词表字典
MEM_VOCAB_ENTRIES = [
    {"id": 1, "name": "walked", "aliases": ["walk", "footprint"]},
]


# ============================================================
#  9. 可视化与调试配置 (Web UI / Debug)
# ============================================================
MAP_VIEW_SCALE = 1.25                     # 顶视地图画布视野缩放倍数 (1.25 -> 约 4.8m 视野)

DEBUG_ENABLED = True                      # 结构化调试总开关
DEBUG_LOG_TO_FILE = True                  # 是否将 stdout 导出到文件
DEBUG_FRAME_SAVE = True                   # 是否异步保存 Web 渲染帧
DEBUG_FRAME_PERIOD = 0.0
DEBUG_QUEUE_SIZE = 2000                   # 调试事件队列容量
DEBUG_FRAME_QUEUE_SIZE = 64               # 图像帧保存队列容量
DEBUG_DROP_OLD_FRAMES = True              # 队列满时丢弃旧帧
DEBUG_EPISODE_FORMAT = "structured_v2"

# 调试子功能开关
DEBUG_FEATURES = {
    "console_debug":     False,           # 控制台冗余状态打印开关
    "episode_log":       True,            # Episode 文本日志记录
    "console_jsonl":     True,            # 时间轴 JSONL 记录
    "pose_history":      True,            # 轨迹记录
    "state_events":      True,            # 状态切换事件记录
    "action_events":     True,            # 动作事件记录
    "command_stats":     False,           # 速度统计
    "performance_probe": True,            # 性能监测探针
    "frame_snapshot":    True,            # 帧快照
    "snapshot":          True,            # 帧元数据索引
    "telemetry":         True,            # 遥测状态流
    "frontier_viz":      True,            # Frontier 可视化计算
    "rgb_overlay":       True,            # Web RGB 视图绘制
    "map_view":          True,            # Web 顶视地图绘制
    "vlm_events":        True,            # VLM 问答事件记录
    "planner_events":    False,           # 规划器调试事件
    "coord_debug":       True,            # 坐标转换调试
    "memory_events":     True,            # 记忆事件记录
    "web_ui":            True,            # 5001 端口 Flask 界面总开关
}

# 调试输出节流周期 (s)
DEBUG_PERIODS = {
    "pose_history":      0.0,
    "state_events":      0.0,
    "action_events":     0.0,
    "performance_probe": 1.0,
    "snapshot":          0.0,
    "rgb_overlay":       0.0,
    "map_view":          0.0,
    "frontier_viz":      1.0,
    "telemetry":         0.5,
    "command_stats":     0.5,
}
