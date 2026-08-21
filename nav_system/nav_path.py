#!/usr/bin/env python
"""nav_path — 路径规划 (RRT* 避障) 与路径跟随 (纯追踪)。

由 nav_mock.py 拆分而来, 自包含: 仅依赖 numpy + nav_rrt (函数内延迟 import),
不反向 import nav_mock / nav_control, 故无循环依赖。
常量 (机器人体积 / 跟随参数) 与运行标记 (路径状态 / 节流时间戳) 均在此模块级定义;
nav_mock 主循环通过 `nav_path.<name>` 访问可变运行标记。
"""

from dataclasses import dataclass
import math
import time
import numpy as np


# ============================================================
#  路径规划 / 跟随 常量  (单一真相源见 nav_constants.py)
# ============================================================
# 以下常量统一在 nav_constants.py 定义并从 ROBOT_RADIUS 派生; 此处 re-export 以兼容既有 import。
from nav_constants import (
    ROBOT_RADIUS,
    RRT_OBSTACLE_INFLATE,
    NAV_CLEARANCE,
    PLAN_BOX_PAD,
    RRT_PLAN_TIME,
    GOAL_RETRIEVE_MAX,
    YAW_TURN_SIGN,
    PATROL_ARRIVE_EPS,
    FRONTIER_ARRIVE_EPS,
    FOLLOW_LIN_MAX,
    FOLLOW_ANG_MAX,
    FOLLOW_LIN_GAIN,
    FOLLOW_LOOKAHEAD,
    FOLLOW_WAYPOINT_THRESHOLD,
    FOLLOW_NEAR_DIST,
    FOLLOW_YAW_EMA,
    FOLLOW_OUT_EMA,
    FOLLOW_DEADBAND,
    FOLLOW_LIN_FLOOR,
    FOLLOW_STRAIGHT_ALPHA,
    FOLLOW_SLEW,
    FOLLOW_INPLACE_ALPHA,
    FOLLOW_TURN_GAIN,
    ESCAPE_PATH0_REVERSE_MIN_ANGLE_DEG,
    ESCAPE_PATH0_ALIGN_EPS_DEG,
    ESCAPE_PATH0_ALIGN_ANG_MAX,
    ESCAPE_PATH0_REVERSE_SPEED,
    ESCAPE_PATH0_REVERSE_MIN_SPEED,
    ESCAPE_PATH0_ARRIVE_EPS,
)

from nav_helpers import ObstacleSnapshot, build_obstacle_snapshot


# ============================================================
#  模块级运行标记 (nav_mock 主循环经 nav_path.<name> 访问)
# ============================================================
_path_is_avoiding = False   # 当前 path 是否由 RRT 避障生成 (否则不查动态阻挡)
_last_replan_t = 0.0        # 上次因"阻挡"触发 replan 的时间 (冷却用, 防刷屏/抖动)
_last_periodic_replan_t = 0.0  # 上次周期重规划时间
_last_plan_fail_t = 0.0     # 上次打印"路径规划失败"的时间 (节流, 防刷屏)
_last_intrusion_warn_t = 0.0  # 上次打印"位姿侵入障碍"警告的时间 (节流, 防刷屏)


# ============================================================
#  路径规划: RRT* 避障 (带机器人体积)
# ============================================================
def crop_to_box(obstacle_pts, a, b, pad=PLAN_BOX_PAD):
    """把障碍点云裁到 a<->b 包围盒 + pad 内 (X,Z 平面)。

    保留 a,b 附近 (含目标) 的点, 丢弃远处点。返回 (M,3) 数组或空数组。
    a,b 为 (x,z)。
    """
    if obstacle_pts is None or len(obstacle_pts) == 0:
        return obstacle_pts
    pts = np.asarray(obstacle_pts, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lo = np.minimum(a, b) - pad
    hi = np.maximum(a, b) + pad
    col_x = pts[:, 0]
    col_z = pts[:, 2] if pts.shape[1] >= 3 else pts[:, 1]
    mask = ((col_x >= lo[0]) & (col_x <= hi[0]) &
            (col_z >= lo[1]) & (col_z <= hi[1]))
    return pts[mask]


@dataclass(frozen=True)
class PathCheckResult:
    safe: bool
    reason: str = "ok"
    point: tuple | None = None
    segment_index: int | None = None
    clearance: float = float("inf")
    nearest_distance: float = float("inf")
    second_nearest_distance: float = float("inf")


@dataclass(frozen=True)
class GoalResolution:
    goal: tuple | None
    reason: str
    clearance: float
    shifted_distance: float = 0.0
    candidates: tuple = ()


@dataclass(frozen=True)
class GoalCandidate:
    goal: tuple
    reason: str
    clearance: float
    shifted_distance: float


def _as_point2(point):
    try:
        value = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
    except (TypeError, ValueError):
        return None
    if value.size < 2 or not np.all(np.isfinite(value)):
        return None
    return value


def check_nav_point(point, obstacle_snapshot, clearance=NAV_CLEARANCE):
    """权威点净空检查。所有导航位置使用同一 ``NAV_CLEARANCE``。"""
    value = _as_point2(point)
    if value is None or obstacle_snapshot is None:
        return PathCheckResult(False, "invalid_point", None, None, 0.0)
    if obstacle_snapshot.tree is None:
        return PathCheckResult(True, point=(float(value[0]), float(value[1])))
    from nav_rrt import tree_nearest_dists
    nearest, second_nearest = tree_nearest_dists(
        obstacle_snapshot.tree,
        obstacle_snapshot.fixed_y,
        float(value[0]),
        float(value[1]),
        k=2,
    )
    # 只有两个最近障碍点都严格位于净空阈值内才否决；单个离群点不触发。
    safe = not (nearest < float(clearance) and
                second_nearest < float(clearance))
    return PathCheckResult(
        safe,
        "ok" if safe else "point_clearance",
        (float(value[0]), float(value[1])),
        None,
        float(second_nearest),
        float(nearest),
        float(second_nearest),
    )


def check_nav_segment(a, b, obstacle_snapshot, clearance=NAV_CLEARANCE,
                      sample_step=0.05, segment_index=None):
    """权威连接段检查，包含起点、终点和全部中间采样点。"""
    p0, p1 = _as_point2(a), _as_point2(b)
    if p0 is None or p1 is None or obstacle_snapshot is None:
        return PathCheckResult(False, "invalid_segment", None,
                               segment_index, 0.0)
    if obstacle_snapshot.tree is None:
        return PathCheckResult(True, segment_index=segment_index)
    seg = p1 - p0
    seg_len = float(np.linalg.norm(seg))
    step = max(float(sample_step), 1e-3)
    n_steps = max(1, int(np.ceil(seg_len / step)))
    min_dist = float("inf")
    min_nearest = float("inf")
    min_point = None
    from nav_rrt import tree_nearest_dists
    for t in np.linspace(0.0, 1.0, n_steps + 1):
        point = p0 + seg * float(t)
        nearest, second_nearest = tree_nearest_dists(
            obstacle_snapshot.tree,
            obstacle_snapshot.fixed_y,
            float(point[0]),
            float(point[1]),
            k=2,
        )
        if second_nearest < min_dist:
            min_dist = float(second_nearest)
            min_nearest = float(nearest)
            min_point = (float(point[0]), float(point[1]))
        if (nearest < float(clearance) and
                second_nearest < float(clearance)):
            print(
                f"[nav_2d][DBG-SEGMENT] safe=false reason=segment_clearance "
                f"segment={segment_index} point=({point[0]:.3f},{point[1]:.3f}) "
                f"d1={nearest:.3f}m d2={second_nearest:.3f}m "
                f"threshold={float(clearance):.3f}m"
            )
            return PathCheckResult(
                False, "segment_clearance", min_point, segment_index,
                min_dist, min_nearest, min_dist)
    return PathCheckResult(True, point=min_point, segment_index=segment_index,
                           clearance=min_dist,
                           nearest_distance=min_nearest,
                           second_nearest_distance=min_dist)


def check_nav_path(path, obstacle_snapshot, start_idx=0, current_pose=None,
                   clearance=NAV_CLEARANCE, sample_step=0.05):
    """权威路径检查；路径复检固定允许 5% 净空容差。"""
    if path is None or obstacle_snapshot is None:
        return PathCheckResult(False, "invalid_path", None, None, 0.0)
    # RRT root/goal 与独立点、段检查仍使用传入的名义净空；这里只对完整
    # 候选路径及 FOLLOW 剩余路径复检放宽 5%，0.40m 对应拒绝阈值 0.38m。
    path_clearance = float(clearance) * 0.95
    try:
        idx = max(0, int(start_idx))
        remaining = list(path[idx:])
    except (TypeError, ValueError):
        return PathCheckResult(False, "invalid_path", None, None, 0.0)
    if not remaining:
        return PathCheckResult(False, "empty_path", None, None, 0.0)
    points = ([current_pose] + remaining) if current_pose is not None else remaining
    min_result = PathCheckResult(True)
    for offset, point in enumerate(points):
        # current_pose 作为起点仅受物理体积 ROBOT_RADIUS(0.35m) 约束(由外部 _robot_check 把关),
        # 对其点检查使用 ROBOT_RADIUS 避免在 0.35~0.38m 临界区死锁；路径航点仍严格使用 path_clearance。
        pt_clearance = ROBOT_RADIUS if (offset == 0 and current_pose is not None) else path_clearance
        result = check_nav_point(point, obstacle_snapshot, pt_clearance)
        if not result.safe:
            segment_index = idx + max(0, offset - (1 if current_pose is not None else 0))
            return PathCheckResult(
                False, result.reason, result.point, segment_index,
                result.clearance, result.nearest_distance,
                result.second_nearest_distance)
        if result.clearance < min_result.clearance:
            min_result = result
    for offset in range(len(points) - 1):
        segment_index = idx + max(0, offset - (1 if current_pose is not None else 0))
        result = check_nav_segment(
            points[offset], points[offset + 1], obstacle_snapshot,
            clearance=path_clearance, sample_step=sample_step,
            segment_index=segment_index,
        )
        if not result.safe:
            return result
        if result.clearance < min_result.clearance:
            min_result = result
    return PathCheckResult(True, clearance=min_result.clearance)


def resolve_nav_goal(semantic_goal, robot_point, obstacle_snapshot,
                     max_retrieve=GOAL_RETRIEVE_MAX, step=0.05):
    """生成按距离 GOAL 从近到远排序的安全导航站位候选。

    GOAL 不安全时，只保留位于「以 GOAL 为原点、机器人所在第一象限」
    内的径向候选。象限只限制方向，不使用 GOAL-机器人矩形的坐标上界；
    因此候选可以沿机器人方向越过当前机器人位置。不使用容差。
    """
    goal = _as_point2(semantic_goal)
    robot = _as_point2(robot_point)
    if goal is None or robot is None or obstacle_snapshot is None:
        return GoalResolution(None, "invalid_input", 0.0)
    candidates = []
    direct = check_nav_point(goal, obstacle_snapshot)
    if direct.safe:
        candidates.append(GoalCandidate(
            (float(goal[0]), float(goal[1])), "direct",
            direct.clearance, 0.0))
    if obstacle_snapshot.tree is None:
        candidate = GoalCandidate(
            (float(goal[0]), float(goal[1])), "no_obstacles",
            float("inf"), 0.0)
        return GoalResolution(candidate.goal, candidate.reason,
                              candidate.clearance, candidate.shifted_distance,
                              (candidate,))

    direction = robot - goal
    for radius in np.arange(float(step), float(max_retrieve) + step * 0.5,
                            float(step)):
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            candidate = goal + radius * np.array(
                [math.cos(angle), math.sin(angle)], dtype=np.float64)
            offset = candidate - goal
            if ((direction[0] > 0.0 and offset[0] < 0.0) or
                    (direction[0] < 0.0 and offset[0] > 0.0) or
                    (direction[1] > 0.0 and offset[1] < 0.0) or
                    (direction[1] < 0.0 and offset[1] > 0.0)):
                continue
            result = check_nav_point(candidate, obstacle_snapshot)
            if result.safe:
                candidates.append(GoalCandidate(
                    (float(candidate[0]), float(candidate[1])),
                    "radial", result.clearance, float(radius)))
    if candidates:
        first = candidates[0]
        return GoalResolution(first.goal, first.reason, first.clearance,
                              first.shifted_distance, tuple(candidates))
    return GoalResolution(None, "no_safe_station", direct.clearance)


def plan_path(start_2d, goal_2d, obstacle_points, fixed_y=0.0,
              plan_start_2d=None, obstacle_snapshot=None,
              max_time=RRT_PLAN_TIME):
    """规划到已经解析好的导航站位；规划器自身不得移动目标。"""
    del plan_start_2d  # 兼容旧调用；目标解析已与 RRT 分离。
    global _path_is_avoiding
    snapshot = obstacle_snapshot or build_obstacle_snapshot(
        obstacle_points, fixed_y=fixed_y)
    goal_check = check_nav_point(goal_2d, snapshot, clearance=NAV_CLEARANCE)
    if not goal_check.safe:
        _path_is_avoiding = False
        return None, None
    planning_points = crop_to_box(
        snapshot.points, start_2d, goal_2d, PLAN_BOX_PAD)
    path = None
    try:
        from nav_rrt import plan_with_rrt
        path = plan_with_rrt(
            start_2d, goal_2d, planning_points,
            NAV_CLEARANCE, fixed_y=fixed_y,
            validation_tree=snapshot.tree,
            max_time=max_time,
        )
    except Exception as exc:
        print(f"[nav_2d] RRT 规划异常 {exc!r}, 返回 None 停等")
    validation = check_nav_path(path, snapshot) if path is not None else \
        PathCheckResult(False, "rrt_failed")
    if path is None or len(path) < 2 or not validation.safe:
        global _last_plan_fail_t
        now = time.time()
        if now - _last_plan_fail_t >= 2.0:
            print(f"[nav_2d] 路径规划失败 reason={validation.reason} "
                  f"clearance={validation.clearance:.3f}m -> 停等重试")
            _last_plan_fail_t = now
        _path_is_avoiding = False
        return None, None
    _path_is_avoiding = True
    return path, (float(goal_2d[0]), float(goal_2d[1]))


def plan_goal_resolution(start_2d, resolution, obstacle_points, fixed_y=0.0,
                         plan_start_2d=None, obstacle_snapshot=None,
                         total_time=RRT_PLAN_TIME):
    """按距离 GOAL 从近到远对候选做路径可达性检验。

    直线连接已经通过权威障碍检查时直接返回；否则在一次常规
    RRT 时间预算内依次尝试最近候选，避免为大量候选各阻塞一个完整超时。
    """
    global _path_is_avoiding
    if resolution is None or not resolution.candidates:
        return None, None, GoalResolution(None, "no_safe_station", 0.0)
    snapshot = obstacle_snapshot or build_obstacle_snapshot(
        obstacle_points, fixed_y=fixed_y)
    start = _as_point2(start_2d)
    if start is None:
        return None, None, GoalResolution(None, "invalid_start", 0.0)

    deadline = time.time() + max(0.0, float(total_time))
    for candidate in resolution.candidates:
        segment = check_nav_segment(start, candidate.goal, snapshot)
        if segment.safe:
            path = [(float(start[0]), float(start[1])), candidate.goal]
            _path_is_avoiding = True
            selected = GoalResolution(
                candidate.goal, candidate.reason, candidate.clearance,
                candidate.shifted_distance, (candidate,))
            return path, candidate.goal, selected

        remaining = deadline - time.time()
        if remaining <= 0.0:
            break
        path, actual_goal = plan_path(
            start, candidate.goal, obstacle_points, fixed_y=fixed_y,
            plan_start_2d=plan_start_2d, obstacle_snapshot=snapshot,
            max_time=min(0.15, remaining))
        if path is not None:
            selected = GoalResolution(
                candidate.goal, candidate.reason, candidate.clearance,
                candidate.shifted_distance, (candidate,))
            return path, actual_goal, selected

    _path_is_avoiding = False
    return None, None, GoalResolution(
        None, "no_reachable_station", resolution.clearance)


# ============================================================
#  路径跟随: 纯追踪 (Pure Pursuit)
# ============================================================
def follow_path_step(path, current_idx, current_pose,
                     threshold=FOLLOW_WAYPOINT_THRESHOLD,
                     arrival_eps=PATROL_ARRIVE_EPS):
    """纯追踪(Pure Pursuit)跟随: 取 lookahead 处目标点走圆弧, 对 pose 噪声鲁棒(不画龙)。
    omega = 2*v*sin(alpha)/L_d (alpha=lookahead 点偏角, L_d=lookahead 距离)。
    当 |alpha| > FOLLOW_INPLACE_ALPHA 时改为原地转(linear=0, omega=GAIN*alpha 饱和):
    先原地转正再前进, 避免带 FOLLOW_LIN_FLOOR 下限走弧线、永远无法原地转。
    返回 ((linear,angular), next_idx, (lkx,lkz), alpha); next_idx>=len(path) 表示完成。"""
    if current_idx >= len(path):
        return (0.0, 0.0), current_idx, None, 0.0

    wp = path[current_idx]
    dx = wp[0] - current_pose[0]
    dz = wp[1] - current_pose[1]
    dist = (dx * dx + dz * dz) ** 0.5

    # 中间 waypoint: 足够近即推进到下一航点 (避免纯追踪在 L_d 极小时抖动/画龙)
    if current_idx != len(path) - 1 and dist < threshold:
        return (0.0, 0.0), current_idx + 1, (wp[0], wp[1]), 0.0

    # 末 waypoint: 仅在"真实到达容差"内才判完成。否则继续前进(落入下方近距离直行/
    # 纯追踪分支), 不再提前返回 (0,0)。修复: 旧逻辑在距末点 <0.3m 即返回 (0,0), 与
    # 状态机到达容差(PATROL_ARRIVE_EPS≈0.0875m)不匹配 -> 机器人停在 0.3m 处,
    # path_idx>=len(path) 误判到达 -> 立刻重规划, 形成 PLAN<->FOLLOW 死循环。
    if current_idx == len(path) - 1 and dist < arrival_eps:
        return (0.0, 0.0), len(path), (wp[0], wp[1]), 0.0

    # ===== 近距离直行: 跳过 lookahead 纯追踪与原地转, 直接朝当前 waypoint 走过去 =====
    # 目标很近时 lookahead 会直接落到 waypoint(甚至被算到身后), pure pursuit 在 L_d 很
    # 小时对位姿噪声极度敏感 -> 抖动/乒乓。近距离保持前进 + 比例转向即可平稳到达。
    if dist < FOLLOW_NEAR_DIST:
        _tgt_ang = np.arctan2(dx, dz)
        alpha = (_tgt_ang - current_pose[2] + np.pi) % (2 * np.pi) - np.pi
        linear = max(FOLLOW_LIN_FLOOR, min(FOLLOW_LIN_MAX, dist * FOLLOW_LIN_GAIN))
        omega = YAW_TURN_SIGN * np.clip(FOLLOW_TURN_GAIN * alpha,
                                        -FOLLOW_ANG_MAX, FOLLOW_ANG_MAX)
        return (linear, omega), current_idx, (wp[0], wp[1]), alpha

    # lookahead 点: 朝当前 waypoint 方向 FOLLOW_LOOKAHEAD 处
    # (path 通常只有 2 点=直线; 距 waypoint < lookahead 时取 waypoint 本身)
    if dist > FOLLOW_LOOKAHEAD:
        lkx = current_pose[0] + dx / dist * FOLLOW_LOOKAHEAD
        lkz = current_pose[1] + dz / dist * FOLLOW_LOOKAHEAD
        L_d = FOLLOW_LOOKAHEAD
    else:
        lkx, lkz = wp[0], wp[1]
        L_d = max(dist, 0.1)

    # lookahead 点相对当前朝向的偏角 (世界系, 与 yaw 同参考系)
    target_angle = np.arctan2(lkx - current_pose[0], lkz - current_pose[1])
    alpha = target_angle - current_pose[2]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi   # 归一化 [-pi, pi]

    # F: 原地转 (in-place turn) — 航向误差过大时先原地转正, 不前进。
    #    否则 PurePursuit 在 FOLLOW_LIN_FLOOR 下限下永远带 0.05 线速度走弧线, 无法原地转。
    if abs(alpha) > FOLLOW_INPLACE_ALPHA:
        omega_drive = FOLLOW_TURN_GAIN * alpha
        omega = YAW_TURN_SIGN * np.clip(omega_drive, -FOLLOW_ANG_MAX, FOLLOW_ANG_MAX)
        return (0.0, omega), current_idx, (lkx, lkz), alpha

    # E: 线速度下限, 防止接近 waypoint 时 v 衰减到 0 压死 omega (低速大角转不动)
    linear = max(min(FOLLOW_LIN_MAX, dist * FOLLOW_LIN_GAIN), FOLLOW_LIN_FLOOR)

    # 纯追踪角速度; YAW_TURN_SIGN 把世界系转成 cmd_vel angular_z 约定
    omega_drive = 2.0 * linear * np.sin(alpha) / L_d
    omega = YAW_TURN_SIGN * np.clip(omega_drive, -FOLLOW_ANG_MAX, FOLLOW_ANG_MAX)

    return (linear, omega), current_idx, (lkx, lkz), alpha


# ============================================================
#  位姿入侵脱困: 沿"最近障碍反方向"稳定后退 (不再径向搜最近自由点,
#  避免该点在障碍边界左右/前后跳变 -> 消除侧向剧烈抖动)
# ============================================================
def find_escape_dir(pose_2d, obstacle_tree, fixed_y=0.0):
    """位姿入侵时, 取最近障碍点, 返回"远离该障碍"的单位方向 (dx, dz)。

    脱困方向 = normalize(pose - 最近障碍点)。该方向只由最近障碍点决定,
    随机器人微移变化平缓, 不会像"径向最近自由点"那样在障碍边界上
    左右/前后跳变 -> 消除脱困侧向剧烈抖动。
    依赖已构建的 bo_tree (KDTreeFlann)。返回 None 时无最近点(无障碍)。
    """
    if obstacle_tree is None:
        return None
    try:
        from nav_rrt import nearest_obstacle_point
        ox, oz, dist = nearest_obstacle_point(
            obstacle_tree, fixed_y, float(pose_2d[0]), float(pose_2d[1]))
    except Exception:
        return None
    if ox is None:
        return None
    dirx = float(pose_2d[0]) - ox
    dirz = float(pose_2d[1]) - oz
    n = math.hypot(dirx, dirz)
    if n < 1e-4:
        return None
    return (dirx / n, dirz / n)


def escape_velocity(cur_pose, escape_dir):
    """脱困: 沿 escape_dir(远离障碍的单位向量) 后退离开。

    后退位移方向 = -heading, 需使其对准 escape_dir ->
      目标 heading = atan2(-dirx, -dirz)
      omega = YAW_TURN_SIGN * clip(TURN_GAIN*alpha) 转向该朝向
      linear 固定后退 (linear < 0)
    稳定不抖: escape_dir 由最近障碍点决定, 随 robot 微移变化平缓,
    且全程后退、不做"前/后切换", 故消除左右剧烈翻转。
    返回 (linear, angular)。
    """
    px, pz, yaw = (float(cur_pose[0]), float(cur_pose[1]), float(cur_pose[2]))
    if escape_dir is None:
        return (-FOLLOW_LIN_MAX * 0.5, 0.0)
    dirx, dirz = float(escape_dir[0]), float(escape_dir[1])
    # 期望 heading: 后退位移(-heading)对准 escape_dir -> heading = atan2(-dirx,-dirz)
    desired_yaw = math.atan2(-dirx, -dirz)
    alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    omega = YAW_TURN_SIGN * np.clip(FOLLOW_TURN_GAIN * alpha,
                                    -FOLLOW_ANG_MAX, FOLLOW_ANG_MAX)
    linear = -FOLLOW_LIN_MAX * 0.5
    return (linear, omega)


def make_escape_path_start_line(start_pose, path_start_2d):
    """冻结从当前被困点到 path[0] 的直线控制参数。

    path[0] 在车头相对方位 [-90°, 90°] 内时前进，否则倒车。
    fixed_yaw 为对准直线后的车头航向，行驶期间不再根据实时位置
    重算目标角，从而避免边走边转的弧线回退。
    """
    if start_pose is None or path_start_2d is None:
        return None
    try:
        sx, sz, syaw = (float(start_pose[0]), float(start_pose[1]),
                        float(start_pose[2]))
        tx, tz = float(path_start_2d[0]), float(path_start_2d[1])
    except (TypeError, ValueError, IndexError):
        return None
    values = (sx, sz, syaw, tx, tz)
    if not all(math.isfinite(v) for v in values):
        return None
    dx = tx - sx
    dz = tz - sz
    total_dist = math.hypot(dx, dz)
    if total_dist < 1e-6:
        return None
    bearing = math.atan2(dx, dz)
    relative = (bearing - syaw + math.pi) % (2.0 * math.pi) - math.pi
    forward_limit = math.radians(ESCAPE_PATH0_REVERSE_MIN_ANGLE_DEG)
    drive_mode = "forward" if abs(relative) <= forward_limit else "reverse"
    fixed_yaw = bearing if drive_mode == "forward" else bearing + math.pi
    fixed_yaw = (fixed_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return {
        "start": (sx, sz),
        "unit": (dx / total_dist, dz / total_dist),
        "distance": total_dist,
        "fixed_yaw": fixed_yaw,
        "drive_mode": drive_mode,
        "relative_angle": relative,
    }


def escape_align_to_yaw_velocity(cur_pose, target_yaw):
    """脱困子阶段原地对准指定世界航向。

    返回 (linear=0, angular, yaw_error, aligned)。
    """
    if cur_pose is None:
        return (0.0, 0.0, 0.0, False)
    try:
        yaw = float(cur_pose[2])
        target_yaw = float(target_yaw)
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0, 0.0, False)
    if not math.isfinite(yaw) or not math.isfinite(target_yaw):
        return (0.0, 0.0, 0.0, False)
    alpha = (target_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    align_eps = math.radians(ESCAPE_PATH0_ALIGN_EPS_DEG)
    if abs(alpha) <= align_eps:
        return (0.0, 0.0, alpha, True)
    omega = YAW_TURN_SIGN * np.clip(
        FOLLOW_TURN_GAIN * alpha,
        -ESCAPE_PATH0_ALIGN_ANG_MAX,
        ESCAPE_PATH0_ALIGN_ANG_MAX)
    return (0.0, float(omega), alpha, False)


def escape_to_path_start_velocity(cur_pose, line, remaining, aligned=False):
    """沿已冻结的 path[0] 直线返回速度命令。

    未对准时只原地转向；对准后只输出正/负线速度，angular 始终为 0。
    航向和前进/倒车模式均来自进入 path0 时的冻结结果。
    返回 (linear, angular, yaw_error, aligned)。
    """
    if cur_pose is None or not isinstance(line, dict):
        return (0.0, 0.0, 0.0, bool(aligned))
    try:
        yaw = float(cur_pose[2])
        fixed_yaw = float(line["fixed_yaw"])
        drive_mode = str(line["drive_mode"])
        remaining = max(0.0, float(remaining))
    except (TypeError, ValueError, IndexError, KeyError):
        return (0.0, 0.0, 0.0, bool(aligned))
    if (not math.isfinite(yaw) or not math.isfinite(fixed_yaw)
            or drive_mode not in ("forward", "reverse")):
        return (0.0, 0.0, 0.0, bool(aligned))

    alpha = (fixed_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    if not aligned:
        _, omega, alpha, aligned = escape_align_to_yaw_velocity(
            cur_pose, fixed_yaw)
        if not aligned:
            return (0.0, omega, alpha, False)

    if drive_mode == "reverse":
        speed = max(ESCAPE_PATH0_REVERSE_MIN_SPEED,
                    min(ESCAPE_PATH0_REVERSE_SPEED,
                        remaining * FOLLOW_LIN_GAIN))
        return (-speed, 0.0, alpha, True)

    speed = max(FOLLOW_LIN_FLOOR,
                min(FOLLOW_LIN_MAX, remaining * FOLLOW_LIN_GAIN))
    return (speed, 0.0, alpha, True)
