#!/usr/bin/env python
"""nav_constants — 导航 + 语义记忆的单一常量真相源 (single source of truth).
"""

# ============================================================
#  物理尺寸 (基): 改这一处即联动全局
# ============================================================
ROBOT_RADIUS = 0.35          # 机器人体积半径 (m): RRT 碰撞膨胀 + 顶视 footprint


# 顶视地图视野线性倍数 (nav_page.draw_map_view):
#   基准 SCALE0=80 px/m, RANGE0=3.0 m (画布 480px → 半视野 3m)。
#   生效: SCALE = SCALE0 / MAP_VIEW_EXTENT,  RANGE = RANGE0 * MAP_VIEW_EXTENT
#   1.0 = 默认; 2.0 = 半视野 6m、面积×4; 网页窗口仍 480×480 不变。
MAP_VIEW_SCALE = 1.25

# ===== FINAL_ADJUST 终调阶段 =====
# 检测命中进入 FINAL_ADJUST: 单帧点云噪声大, 此阶段边走边持续检测,
# 用 EMA 把多次检测到的目标位置互相平滑, 计时/帧数达标后锁定 -> FINAL_FOLLOW。
FINAL_ADJUST_TIME = 20        # adjust至少x秒  30 if object
FINAL_ADJUST_MIN_FRAMES = 6    # 终调最少检测帧数: 未达标帧数即使到时也不锁定 (防只测到 1 帧就锁)
FINAL_ADJUST_DET_PERIOD = 1  # 终调阶段检测周期 (s): 比常规 2.0s 更快, 以积累足够样本做 EMA 平滑
FINAL_ADJUST_SMOOTH_ALPHA = 0.5 #0.35  # EMA 增益: 新帧权重, 越大越信任新检测; 0.35 约 4~5 帧收敛
FINAL_ADJUST_REJECT_DIST = 5.0

FINAL_ADJUST_LOCK_DIST = 1.5
FINAL_ADJUST_FORCE_TIME = 60.0


# ===== 脱困后 odom 保持窗口 (方案A: 防 stale SLAM 把 odom 进度拉回) =====
# 位姿入侵脱困(_escaping)结束瞬间, SLAM 位姿往往仍 stale (障碍近旁视觉退化),
# 若立刻 revert 回 SLAM, 会把 odom 在脱困期间挣脱出来的进度又拉回障碍方向。
# 故脱困结束后继续用 odom 位姿(锚点冻结+实时 odom 增量), 直到 SLAM 追上 odom
# 修正位姿(距离+航向都够近)或超时, 才切回 SLAM —— 即"冻结"在 odom 上等 SLAM 自愈。
POST_ESCAPE_HOLD = 1.5          # odom 保持窗口上限 (s): 超时强制回退(避免 odom 长期漂移)
POST_ESCAPE_CONVERGE_DIST = 0.10  # 收敛判定: SLAM 与 odom 修正位姿水平距离 < 此值(m)视为追上
POST_ESCAPE_CONVERGE_YAW = 0.10   # 收敛判定: 两者航向差 < 此值(rad, ~5.7°)视为追上

# ============================================================
#  规划 / 到达
# ============================================================
SUBOPT_GOAL_EPS = ROBOT_RADIUS*0.8   # sub-opt-goal 自由点容差: goal 距障>=R, 避免二次回退
PLAN_BOX_PAD = 3.0               # 规划窗口余量 (start<->goal 包围盒+pad)
REPLAN_PERIOD = 1.0              # 周期重规划周期 (s): 1Hz 跟随最新局部地图
RRT_PLAN_TIME = 0.5              # 单次 RRT 超时 (s): 无解最多阻塞这么久即返
RRT_OBSTACLE_INFLATE = ROBOT_RADIUS + 0.05   # ≈0.40m
# 唯一导航安全净空。ROBOT_RADIUS 只表达物理尺寸；所有导航位置、连接段、
# RRT root/goal、FOLLOW 剩余路径和 ESCAPE 完成判定均使用 NAV_CLEARANCE。
NAV_CLEARANCE = RRT_OBSTACLE_INFLATE
GOAL_RETRIEVE_MAX = 2.0         # 语义 GOAL 周围导航站位最大搜索半径 (m)

# 到达容差 (m): 用 robot 中心判达 (goal 已保证距障>=R; 越大越早判到)
PATROL_ARRIVE_EPS = ROBOT_RADIUS * 0.3
FRONTIER_ARRIVE_EPS = ROBOT_RADIUS * 0.8


# ============================================================
#  跟随 (纯追踪 PurePursuit)
# ============================================================
YAW_TURN_SIGN = -1.0     # 顶视 +Z 前 +X 右, 目标在右(alpha>0)应右转; 底盘 +angular_z=左转故=-1
FOLLOW_LIN_MAX = 0.10    # 前进线速度上限 (m/s) 11f:  0.3
FOLLOW_ANG_MAX = 0.10    # 跟随转向角速度上限 (rad/s) 11f:  0.3
FOLLOW_WAYPOINT_THRESHOLD = 0.3  # 中间 waypoint 小于此距离直接推进到下一点
FOLLOW_LIN_GAIN = 0.3    # 近距减速比例增益
FOLLOW_LOOKAHEAD = 0.5   # 纯追踪 lookahead 距离 (m): 越大越稳, 跟踪精度略降
FOLLOW_NEAR_DIST = 0.6    # 近距离直行阈值(m): 距 waypoint < 此值跳过 lookahead 纯追踪, 直接朝目标走过去
                           #   (避免近距离 look 落到身后/原地转乒乓抖动; 需 > follow_path_step threshold=0.3)

FOLLOW_YAW_EMA = 0.3     # yaw 输入平滑 EMA 系数 (10Hz 下延迟 ~0.23s)
FOLLOW_OUT_EMA = 0.4     # 角速度输出平滑 EMA 系数
FOLLOW_DEADBAND = 0.087  # 角速度死区 (rad, ~5°): |omega| 小于此不转向, 防微抖画龙
FOLLOW_LIN_FLOOR = 0.05  # 线速度下限 (m/s): 防接近 waypoint 时 v 衰减到 0 压死 omega (不死锁)
FOLLOW_STRAIGHT_ALPHA = 0.175  # 直行/转弯阈值(~10°): <=此值直行用死区; >此值转弯跳过死区
FOLLOW_SLEW = 0.08       # 角速度变化率上限 (rad/帧): 防 yaw 阶跃跳变时的大脉冲
# 原地转 (in-place turn): 航向误差过大时先原地转正(线速度=0), 不再走弧线
FOLLOW_INPLACE_ALPHA = 0.70  # 原地转阈值(rad, ~40°): |alpha| 超过此值 -> 原地转; 否则走 PurePursuit 弧线
FOLLOW_TURN_GAIN = 0.50    # 原地转角速度比例增益: omega=GAIN*alpha (饱和于 FOLLOW_ANG_MAX)

# 近墙大角转向前回退：规划成功后，仅在前方原始 OCC 呈墙状且下一航段需大角转向时触发。
PRETURN_MIN_TURN_DEG = 5.0
PRETURN_WALL_RAY_MIN_DEG = -35
PRETURN_WALL_RAY_MAX_DEG = 35
PRETURN_WALL_RAY_STEP_DEG = 1
PRETURN_WALL_RAY_MAX_DIST = 0.80
PRETURN_WALL_MIN_HIT_RATIO = 0.65
PRETURN_WALL_MAX_MEDIAN_DIST = 0.60
PRETURN_WALL_MAX_LINE_RESIDUAL = 0.10
PRETURN_RETREAT_DISTANCE = 0.30
PRETURN_RETREAT_SPEED = FOLLOW_LIN_MAX
PRETURN_RETREAT_TIMEOUT = 4.0

# 位姿侵入时返回 path[0]：相对方位 [-90°, 90°] 前进，其余后半球倒车。
# 进入 path0 时冻结直线航向；先原地对准，再以 angular=0 直线行驶。
ESCAPE_PATH0_REVERSE_MIN_ANGLE_DEG = 90.0
ESCAPE_PATH0_ALIGN_EPS_DEG = 5.0
ESCAPE_PATH0_ALIGN_ANG_MAX = 0.12
ESCAPE_PATH0_REVERSE_SPEED = 0.05
ESCAPE_PATH0_REVERSE_MIN_SPEED = 0.03
# 保留旧名以兼容外部调用。
ESCAPE_PATH0_REVERSE_ANG_MAX = ESCAPE_PATH0_ALIGN_ANG_MAX
ESCAPE_PATH0_ARRIVE_EPS = 0.08

# ============================================================
#  点云横截面 (障碍高度带, get_obstacle_points)
# ============================================================
# 从全图点云切出导航平面附近的横截面作为障碍点:
#   mask = nav_y - OBS_Y_ABOVE <= points[:,1] <= nav_y + OBS_Y_BELOW
# SLAM 世界系 Y 轴向下为正, 故 -OBS_Y_ABOVE 是"平面上方", +OBS_Y_BELOW 是"平面下方"。
OBS_Y_ABOVE = 0.2        # 横截面上方相对范围 (m): 取到 nav 平面上方多高
OBS_Y_BELOW = 0.5        # 横截面下方相对范围 (m): 取到 nav 平面下方多深

# ============================================================
#  位姿融合与平滑 (PoseContext)
# ============================================================
NAV_SMOOTH_POS = 0.30       # 位置 (x,z) 关键帧修正软收敛系数
NAV_SMOOTH_YAW = 0.30       # 偏航 (yaw) 关键帧修正软收敛系数
ODOM_YAW_OFFSET = 0.0       # odom 系 -> SLAM 世界系固定安装偏置 (rad)

# ============================================================
#  底盘控制 (运动线程)
# ============================================================
MOTION_HZ = 20           # 移动线程发 cmd_vel 的频率 (Hz), 与主循环解耦防抖
SCAN_ANGULAR = 0.10      # 扫描/重定位自旋角速度 (rad/s), 慢速平滑防抖
BOOT_SCAN_ENABLED = True  # 开局/reloc 恢复是否先扫描一圈建图; False=跳过扫描直接进 VLM_INQUIRY
RELOC_STOP_TIME = 0.5    # RELOC L0 停车稳定时长 (s), 给 SLAM 自愈窗口
RELOC_BACK_TIME = 3.0    # RELOC L1 直线后退时长 (s), 约退 0.10*3≈0.3m 离墙

# ============================================================
#  语义记忆 (nav_memory) — 全部从 ROBOT_RADIUS 派生
# ============================================================
MEM_ENABLE = True
MEM_PERSIST_ENABLE = True
MEM_DEBUG_EVERY = 30

# walked: 写入阈值 (关键联动点)
#   WALK_MIN_TRANS(b) 是"每走过多少距离才新增一个 walked 记录点", 应越小越好(记录越密)。
#   约束公式: WALK_MIN_TRANS < PATROL_ARRIVE_EPS
WALK_MIN_TRANS = 0.05
assert WALK_MIN_TRANS < PATROL_ARRIVE_EPS

MEM_WALKED_ENABLE = True
MEM_WALKED_RADIUS = ROBOT_RADIUS+0.4        
MEM_WALKED_MERGE_RADIUS = 0.08   #  两个 walked 事件多近才合并成一个圆。
                                 #  取 ~2.4*WALK_MIN_TRANS(0.05): 合并后圆中心间距~0.12-0.24m,
                                 #  任意脚印到最近中心 <=0.08m (始终"踩在中心附近"),
                                 #  且 kappa 取 1.0(严格) 也零洞, 不再依赖 kappa 兜底。
                                 #  比关合并(每采样点一圆)省 ~4x 圆数量, 内存可控。
MEM_WALKED_MIN_DT = 1.0
MEM_WALKED_WEIGHT = .0

# walked memory 世界半径合法范围。radius 已固定为世界单位，不再随 Sim3 scale 缩放。
MEM_WALKED_RADIUS_MIN = 0.05
MEM_WALKED_RADIUS_MAX = 1.50
# 关键帧 scale 仅做诊断，不钳制位姿。相对写入时 scale 超界时记录 warning。
MEM_ANCHOR_SCALE_RATIO_MIN = 0.50
MEM_ANCHOR_SCALE_RATIO_MAX = 2.00
# 查询"是否走过"时, 点到 walked 圆边界的剩余余量必须 > 此值才算走过。
# = ROBOT_RADIUS 表示: 机器人体积半径内的边沿不算走过, 只认可圆内心足够深的点,
# 避免目标点贴着记忆圆边沿就被误判为已走过 (机器人本体根本到不了边界)。
MEM_WALKED_EDGE_MARGIN = ROBOT_RADIUS

# object
MEM_OBJECT_RADIUS = ROBOT_RADIUS * 0.1         # 0.245 (物体圆: 世界半径 = kappa*scale*此值)
MEM_OBJECT_MERGE_RADIUS = ROBOT_RADIUS * 1.1    # ~0.39
MEM_OBJECT_CLUSTER_EPS = ROBOT_RADIUS * 1.4     # ~0.49
MEM_OBJECT_WEIGHT = 1.0

# query
MEM_QUERY_RADIUS = ROBOT_RADIUS * 6.0          # 2.1
MEM_QUERY_N_SAMPLES = 512
MEM_QUERY_KAPPA = 1.0

# frontier_grid
MEM_FRONTIER_ROBOT_RADIUS = RRT_OBSTACLE_INFLATE       # frontier dilation
MEM_FRONTIER_RESOLUTION = 0.05                 # 网格分辨率 (m)
MEM_FRONTIER_WIN_RADIUS = 20.0                  # 查询窗口半径 (m)
MEM_FRONTIER_OCC_MIN_POINTS = 3                 # 原始 OCC: 每格至少命中点数
MEM_FRONTIER_KAPPA = MEM_QUERY_KAPPA
MEM_FRONTIER_SHAPE = "disk"                    # memory的栅格化方式，disk表示以云盘标记walked
MEM_FRONTIER_INFLATE_TYPE = "circle"           # 障碍物膨胀方式， circle 表示occ以半径ROBOT_RADIUS膨胀标记， square 表示occ以半边长ROBOT_RADIUS膨胀标记
MEM_FRONTIER_SIGMA_WALKED = 0.40               # local frontier 查找时筛取memory的扩大范围，MEM_FRONTIER_WIN_RADIUS + 3 MEM_FRONTIER_SIGMA_WALKED = 实际筛选区域
MEM_FRONTIER_GLOBAL_RESOLUTION = 0.20          # global 模式网格分辨率
MEM_FRONTIER_MAX_OBSTACLE_POINTS = 100000

# 语义词表 (sid<->name<->aliases): 与 nav_memory/types.py 的 SID_* 一致。
# 统一放此处作为单一真相源 (不再依赖 nav_memory/config_mem.yaml).
MEM_VOCAB_ENTRIES = [
    {"id": 1, "name": "walked", "aliases": ["walk", "footprint"]},
    # object memory 尚未启用；保留 SID/type/query 实现，暂不暴露 vocabulary 入口。
    # {"id": 2, "name": "dust_bin", "aliases": ["dustbin", "trash_can", "垃圾桶"]},
]

# ============================================================
#  自动导航 (nav_auto.py) — 周期 / 自动决策
# ============================================================
# 自动版 nav: 不再等网页/终端按键, 由主循环自动驱动 j/k/l/f/o 逻辑。
OBJ_DET_PERIOD = 2.0        # 自动目标检测周期 (s): 每这么久在主循环跑一次 o 键检测逻辑
AUTO_VLM_RETRY_PERIOD = 3.0 # VLM 方向问询失败/无明确答案时的重试间隔 (s), 防刷屏

# ============================================================
#  调试 / Episode 结构化记录 (nav_debug.NavDebugger)
# ============================================================
# 每次启动 nav_auto.py 会在 ./episodes/ep_YYYYMMDD_HHMMSS/ 下建结构化 episode 文件夹:
#   metadata.json / summary.json / config/debug_config.json
#   telemetry/{pose,status,control,performance}.jsonl
#   vlm/vlm_events.jsonl  events/{events,planner_events}.jsonl
#   frames/step_*.jpg (+ frames/index.jsonl)
# 总开关 DEBUG_ENABLED=False 关闭一切落盘; 各功能由 DEBUG_FEATURES / DEBUG_PERIODS 驱动。
DEBUG_ENABLED          = True    # 总开关: False 关闭所有调试/落盘
DEBUG_LOG_TO_FILE      = True    # 接管 stdout -> episode/nav_auto.log
DEBUG_FRAME_SAVE       = True    # 是否保存 RGB+顶视地图 横排快照
DEBUG_FRAME_PERIOD     = 0.0     # (保留兼容) 旧式节流, 实际快照节流见 DEBUG_PERIODS["snapshot"]
DEBUG_QUEUE_SIZE       = 2000    # 结构化事件队列上限
DEBUG_FRAME_QUEUE_SIZE = 64      # 异步帧保存队列上限
DEBUG_DROP_OLD_FRAMES  = True    # 帧队列满时丢弃最旧帧 (否则丢弃最新)
DEBUG_EPISODE_FORMAT   = "structured_v2"  # v2: step+seq 外壳 + console.jsonl + vlm.stage

# 各调试功能开关 (NavDebugger 按名读取; 默认全开便于 debug)
DEBUG_FEATURES = {
    "console_debug":     False,  # [nav_2d][STATE] 等控制台调试 print
    "episode_log":       True,   # 接管 stdout -> nav_auto.log
    "console_jsonl":     True,   # console/console.jsonl 分段存储 (Web 时间轴)
    "pose_history":      True,   # telemetry/pose.jsonl (轨迹)
    "state_events":      True,   # events/state
    "action_events":     True,   # events/action
    "command_stats":     False,  # telemetry/control (当前未使用, 保留)
    "performance_probe": True,   # performance.jsonl + [NAVSTEP]/[RELOC] 探针
    "frame_snapshot":    True,   # frames/step_*.jpg
    "snapshot":          True,   # 帧索引记录 (与 frame_snapshot 同开)
    "telemetry":         True,   # telemetry/status.jsonl
    "frontier_viz":      True,   # frontier 可视化缓存
    "rgb_overlay":       True,   # 每帧绘制 debug overlay 到网页
    "map_view":          True,   # 每帧绘制顶视地图
    "vlm_events":        True,   # vlm/vlm_events.jsonl
    "planner_events":    False,  # planner 事件 (按需开启)
    "coord_debug":       True,   # coord/coord_events.jsonl (投影/点云调试, 供 Web 回放)
    "memory_events":     True,   # memory/memory_events.jsonl (半径/scale/删除诊断)
    "web_ui":            True,   # 启动 Flask 网页 (localhost:5001)
}
# 各功能输出节流周期 (秒); 0 = 每帧/每次都写
DEBUG_PERIODS = {
    "pose_history":      0.0,
    "state_events":      0.0,
    "action_events":     0.0,
    "performance_probe": 1.0,
    "snapshot":          0.0,    # 快照保存节流 (0=每帧都存)
    "rgb_overlay":       0.0,
    "map_view":          0.0,
    "frontier_viz":      1.0,
    "telemetry":         0.5,
    "command_stats":     0.5,
}



# ============================================================
#  VLM 目标检测 (nav_mock o 键触发)
# ============================================================
# o 键检测「什么」: 优先用网页输入框设的目标, 没设就回退此默认词。
# VLM_DETECT_TARGET_DEFAULT = "emergency white exit double door with green sign"
# VLM_DETECT_TARGET_DEFAULT ="fire-hydrant cabinet with vertical fire-protection pipes"
# VLM_DETECT_TARGET_DEFAULT ="fire telephone mounted on the white wall"
# VLM_DETECT_TARGET_DEFAULT = "long bright office corridor"
# VLM_DETECT_TARGET_DEFAULT = "Air conditioner adjuster on the wall"
# VLM_DETECT_TARGET_DEFAULT = "microwave oven on the kitchen counter"

# VLM_DETECT_TARGET_DEFAULT = "White round plate plastic stool in corridor"
# VLM_DETECT_TARGET_DEFAULT = "White round plate plastic stool"
# VLM_DETECT_TARGET_DEFAULT = "middle of corridor"
VLM_DETECT_TARGET_DEFAULT = "rest area"
# VLM_DETECT_TARGET_DEFAULT = "washing area"
# VLM_DETECT_TARGET_DEFAULT = "coffee corner"

# VLM 方向问询 prompt (nav_auto 在 VLM_INQUIRY 自动问, 取代人工按 j/k/l)。
# 要求模型只回一个词 left/right/front; 解析不到这三个词即视为 F(最终目标)。
AUTO_VLM_DIR_PROMPT = f"""You are controlling a mobile robot exploring an indoor space to find {VLM_DETECT_TARGET_DEFAULT}.
Based on the current camera view, decide which horizontal direction the robot should explore next.
You must select eangularword from: left, right, front.
- "left": the promising area is in the left part of the view
- "right": the promising area is in the right part of the view
- "front": the promising area is straight ahead / center
Reply with ONLY the single word (left/right/front), no explanation."""

VLM_URL_DEFAULT = "http://localhost:8222/v1"   # Qwen VLM 服务 (OpenAI 兼容)
VLM_MODEL_DEFAULT = ""
# object detection 的 guided-JSON prompt 模板 ({} 处填检测目标词)。
VLM_DETECTION_PROMPT = f"Detect {VLM_DETECT_TARGET_DEFAULT}" +"""and identify their reference designators (reference numbers), and output the results in the following JSON format:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "type_of_component", "sub_label": "Reference_designator"},
  ...
]
```"""
# h 键自由问答的「自设」prompt (纯多模态问答, 不强制 JSON schema, 回答文本显示在网页)。
# 这是用户自设值: 想要问什么就改这一行 (无需任何代码改动)。
VLM_H_PROMPT = f"In order to find {VLM_DETECT_TARGET_DEFAULT} , in current observation, which direction of the area is most likely to go? You have to select a choice from left, right, front,  and explain"
# 周期自动检测前的「存在性问询」: 先 cheap ask 当前画面是否含目标,
# 仅当确认含 (yes) 才跑昂贵的 detect (±SAM)。要求单字 yes/no 便于解析。
# VLM_PRESENCE_PROMPT = f"""Look at the current camera image. Is there full whole {VLM_DETECT_TARGET_DEFAULT} present in the scene?
# Answer with only the single word 'yes' or 'no', no explanation."""
VLM_PRESENCE_PROMPT = f"""Look at the current camera image. Is there {VLM_DETECT_TARGET_DEFAULT} present in the scene?
Answer with only the single word 'yes' or 'no', no explanation."""

# Mobile SAM checkpoint (机器人副本路径; 开发机无 GPU/ckpt 时 SAM 会懒加载失败,
# 但 bbox 检测不受影响, 仅缺 mask 叠加)。
MOBILE_SAM_CHECKPOINT_PATH = '/home/agilex/yinzecheng/opennav/mobile_sam/mobile_sam.pt'
# VLM (o 检测 / h 问答) 的 (img, prompt, answer) 保存目录; 每次调用存一对 png+txt。
VLM_PAIRS_DIR = "vlm_pairs"
