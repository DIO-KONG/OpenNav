#!/usr/bin/env python
"""nav_mock 的辅助函数集合 (检测 / VLM / 位姿 / 障碍 / frontier)。

从 nav_mock.py 抽出, 主状态机只 `from nav_helpers import ...` 即可,
main() 逻辑不动。所有函数均为纯函数或轻量类, 仅依赖 nav_control
(mock / _angle_diff) 与标准库/numpy/open3d, 不反向依赖 nav_mock,
因此不会引入循环 import。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import numpy as np
from scipy import ndimage
import open3d as o3d  # 点云半径密度滤波

from nav_control import mock, _angle_diff, NavState
from nav_constants import OBS_Y_ABOVE, OBS_Y_BELOW, AUTO_VLM_DIR_PROMPT, ROBOT_RADIUS,NAV_CLEARANCE


@dataclass(frozen=True)
class ObstacleSnapshot:
    """一次不可变的导航障碍快照。

    ``points`` 始终保存完整障碍点；RRT 可另行降采样用于搜索，但所有安全
    结论必须使用这里的 ``tree``。``revision`` 由调用方在障碍缓存更新时递增，
    用于避免对同一地图重复验证路径。
    """

    points: np.ndarray
    fixed_y: float
    tree: object
    revision: int = 0


@dataclass(frozen=True)
class CameraCalibration:
    """原始 RGB 图像的针孔内参与畸变参数。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple


def load_camera_calibration(path):
    """读取 MASt3R-SLAM calibration YAML，供 Debug bbox 解析投影使用。"""
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    values = data.get("calibration") or []
    if len(values) < 4:
        raise ValueError(f"invalid camera calibration: {path}")
    return CameraCalibration(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(values[0]),
        fy=float(values[1]),
        cx=float(values[2]),
        cy=float(values[3]),
        distortion=tuple(float(v) for v in values[4:]),
    )


def _project_world_point(point_world, camera_pose_world, image_shape,
                         calibration):
    """将一个世界点解析投影到当前原始 RGB 帧，返回 (u, v, distance)。"""
    if point_world is None or camera_pose_world is None or calibration is None:
        return None
    point = np.asarray(point_world, dtype=np.float64).reshape(-1)
    pose = np.asarray(camera_pose_world, dtype=np.float64).reshape(-1)
    if point.size < 3 or pose.size < 7 or not np.all(np.isfinite(point[:3])) \
            or not np.all(np.isfinite(pose[:7])):
        return None

    qx, qy, qz, qw = pose[3:7]
    q_norm = float(np.linalg.norm((qx, qy, qz, qw)))
    if q_norm <= 1e-12 or not np.isfinite(q_norm):
        return None
    qx, qy, qz, qw = (qx / q_norm, qy / q_norm,
                      qz / q_norm, qw / q_norm)
    # T_WC 的旋转矩阵；世界点进入相机系需使用其转置。
    rotation_wc = np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz),
         2.0 * (qx * qy - qz * qw),
         2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),
         1.0 - 2.0 * (qx * qx + qz * qz),
         2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),
         2.0 * (qy * qz + qx * qw),
         1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    delta_world = point[:3] - pose[:3]
    point_camera = rotation_wc.T @ delta_world
    depth = float(point_camera[2])
    if depth <= 1e-4 or not np.isfinite(depth):
        return None

    xn = float(point_camera[0] / depth)
    yn = float(point_camera[1] / depth)
    coeffs = list(calibration.distortion) + [0.0] * 5
    k1, k2, p1, p2, k3 = coeffs[:5]
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn

    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    scale_x = float(image_w) / float(calibration.width)
    scale_y = float(image_h) / float(calibration.height)
    u = (calibration.fx * xd + calibration.cx) * scale_x
    v = (calibration.fy * yd + calibration.cy) * scale_y
    distance = float(np.linalg.norm(delta_world))
    if not np.all(np.isfinite((u, v, distance))) or distance <= 1e-4:
        return None
    return float(u), float(v), distance


def project_locked_target_bbox(target_world, camera_pose_world, image_shape,
                               calibration, reference_bbox_px,
                               reference_image_shape,
                               reference_camera_pose):
    """按锁定世界点投影 bbox；宽高按相机到目标距离的反比缩放。"""
    current = _project_world_point(
        target_world, camera_pose_world, image_shape, calibration)
    reference = _project_world_point(
        target_world, reference_camera_pose, reference_image_shape,
        calibration)
    if current is None or reference is None:
        return None

    bbox = np.asarray(reference_bbox_px, dtype=np.float64).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        return None
    ref_h, ref_w = (int(reference_image_shape[0]),
                    int(reference_image_shape[1]))
    cur_h, cur_w = int(image_shape[0]), int(image_shape[1])
    if min(ref_h, ref_w, cur_h, cur_w) <= 0:
        return None

    ref_box_cx = 0.5 * (bbox[0] + bbox[2])
    ref_box_cy = 0.5 * (bbox[1] + bbox[3])
    distance_scale = reference[2] / current[2]
    resolution_x = float(cur_w) / float(ref_w)
    resolution_y = float(cur_h) / float(ref_h)
    center_x = current[0] + (ref_box_cx - reference[0]) * \
        resolution_x * distance_scale
    center_y = current[1] + (ref_box_cy - reference[1]) * \
        resolution_y * distance_scale
    box_w = max(1.0, (bbox[2] - bbox[0]) * resolution_x * distance_scale)
    box_h = max(1.0, (bbox[3] - bbox[1]) * resolution_y * distance_scale)

    x1, y1 = center_x - 0.5 * box_w, center_y - 0.5 * box_h
    x2, y2 = center_x + 0.5 * box_w, center_y + 0.5 * box_h
    if x2 <= 0.0 or y2 <= 0.0 or x1 >= cur_w or y1 >= cur_h:
        return None
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(cur_w - 1), x2), min(float(cur_h - 1), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # draw_detections 使用 Qwen 的 0..1000 bbox 约定。
    return [x1 / cur_w * 1000.0, y1 / cur_h * 1000.0,
            x2 / cur_w * 1000.0, y2 / cur_h * 1000.0]


def build_obstacle_snapshot(obstacle_points, fixed_y=0.0, revision=0):
    """从完整障碍点构建权威快照；空地图使用 ``tree=None``。"""
    if obstacle_points is None:
        points = np.empty((0, 3), dtype=np.float64)
    else:
        points = np.asarray(obstacle_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            points = np.empty((0, 3), dtype=np.float64)
    tree = None
    if len(points) > 0:
        from nav_rrt import build_obstacle_tree
        tree = build_obstacle_tree(points, fixed_y)
    return ObstacleSnapshot(
        points=points,
        fixed_y=float(fixed_y),
        tree=tree,
        revision=int(revision),
    )


def _cmd_str(linear, angular):
    """速度 → 可读动作描述。"""
    if abs(linear) < 0.01 and abs(angular) < 0.01:
        return "stop"
    if abs(angular) < 0.05:
        return f"forward({linear:.2f})"
    if abs(linear) < 0.05:
        d = "L" if angular > 0 else "R"
        return f"turn_{d}({abs(angular):.2f})"
    d = "L" if angular > 0 else "R"
    return f"arc(v={linear:.2f},{d}{abs(angular):.2f})"

# ============================================================
#  Mock 检测 / VLM (键盘驱动)
# ============================================================
def key_to_bbox(key, h, w):
    """按键 → 图片三分之一的 bbox。"""
    third = w // 3
    if key == "j":
        return (0, 0, third, h)            # 左 1/3
    elif key == "k":
        return (third, 0, 2 * third, h)    # 中 1/3 (前方)
    elif key == "l":
        return (2 * third, 0, w, h)        # 右 1/3
    elif key == "o":
        return (third, 0, 2 * third, h)    # 中 1/3 (前方)
    return None


def run_object_detection(img):
    """每步调用的目标检测 (mock: 按 o 触发)。"""
    key = mock.take_det_key()
    if key is None or img is None:
        return None
    h, w = img.shape[:2]
    return key_to_bbox(key, h, w)


def run_vlm_inquiry(img, pose):
    """到达 patrol 目标后的 VLM 问询 (mock: 按 j/k/l 触发)。"""
    key = mock.take_vlm_key()
    if key is None or img is None:
        return None
    h, w = img.shape[:2]
    return key_to_bbox(key, h, w)


def get_keyframe_pose(slam: "Mast3rSlamWrapper"):
    """获取 2D 导航位姿 (x, z, yaw) 或 None。

    位置用最后一个被 BA 校正的关键帧位姿 (get_pose_keyframe /
    get_pose_full_keyframe), 与地图点云 (get_map) 同源同准, 旋转/运动
    后不漂, 用作地图/路径坐标基准。

    yaw 用"当前帧"连续位姿 (get_pose), 而非关键帧 yaw:
    关键帧 yaw 只在新建关键帧时更新, 运动中稀疏/阶跃跳变, 是
    follow 画龙的主因; 当前帧 yaw 每帧连续更新, 转向不抖。
    (SCANNING 的旋转累计也已用当前帧, 见主循环 cur_pose)
    """
    pose = slam.get_pose_keyframe()
    if pose is None:
        return None
    pose_full = slam.get_pose_full_keyframe()
    if pose_full is None:
        return None
    cur = slam.get_pose()          # 当前帧连续位姿 (x, z, yaw)
    yaw = cur[2] if (cur is not None) else pose[2]
    return (float(pose_full[0]), float(pose_full[2]), yaw)


# ============================================================
#  轮式里程计回退 (仅 RELOC 时用 /odom 死推算 nav_pose)
# ============================================================
class OdomHolder:
    """线程安全地保存最新的 /odom 位姿。

    odom 固定坐标系约定 (标准 ROS 移动底盘): x=前, y=左, z=上,
    yaw 绕竖直 z 轴。与 SLAM 世界系 (X=右, Z=前, Y=上) 不同, 因此
    相对位移需经 _odom_nav_pose 换算后才能与规划/障碍同源使用。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None       # (x, y, yaw) 或 None

    def cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # yaw 绕竖直 z 轴 (标准 ROS 四元数 -> yaw)
        yaw = math.atan2(
            2.0 * (q.z * q.w + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        with self._lock:
            self._latest = (float(p.x), float(p.y), float(yaw))

    def get(self):
        with self._lock:
            return self._latest


# odom 系 -> SLAM 世界系的固定安装偏置 (rad).
# 相机/底盘坐标系相对 SLAM 世界系存在约 90° 固定旋转时设为 math.pi/2;
# 若方向反了改 -math.pi/2; 若实测无偏置则设 0.0.
ODOM_YAW_OFFSET = 0* math.pi / 2


def _odom_nav_pose(slam_anchor, odom_anchor, odom_now, yaw_sign=1.0):
    """把 odom 相对位移换算到 SLAM 世界系 (X=右, Z=前), 锚在最后可信 SLAM 位姿。

    参数:
        slam_anchor: (sx, sz, syaw)  SLAM 世界系最后可信位姿
        odom_anchor: (ox, oy, oyaw)  锚定时刻 odom 位姿 (x=前, y=左)
        odom_now:    (oxn, oyn, oywn) 当前 odom 位姿
        yaw_sign:    odom->SLAM 偏航符号 (±1), 由主循环在可靠且旋转时自标定。
    返回 nav_pose (nx, nz, nyaw), 与 cur_pose 同源同全局系, 可直接用于
    路径跟随 / 障碍距离 / 绘制。

    注意: odom 系(x=前,y=左,Y=上, +angular_z=左转, 正odom yaw=左) 与 SLAM 世界系
    (X=右,Z=前, 航向=atan2(fx,fz), 正yaw=右) 旋向相反, odom 偏航不能直接叠加,
    需乘 yaw_sign 校正。
    航向用 odom 偏航积分 (轮式里程计 yaw 稳定不漂), 不用 SLAM 位移法 ——
    SLAM pose 本身漂移大, 用其位置位移方向反推航向会持续偏一边 (纯追踪一直转圈)。
    yaw_sign 手动硬编码 (SLAM 漂移导致自标定不可靠), 实测反向则翻号即可。
    """
    ddx = odom_now[0] - odom_anchor[0]
    ddy = odom_now[1] - odom_anchor[1]
    c = math.cos(odom_anchor[2])
    s = math.sin(odom_anchor[2])
    # 把 odom 固定系位移分解成"锚定朝向"下的 前(f)/左(l) 分量
    f = c * ddx + s * ddy
    l = -s * ddx + c * ddy
    # 映射到 SLAM 世界系: 前(f) / 左(l) 是机器人局部坐标, 需按锚定 SLAM 航向
    # syaw 旋转到世界系。原实现漏了旋转, 仅在机器人正对 SLAM +Z 时正确, 否则
    # odom 修正的位置会偏向侧边。
    #   局部前: (sin(syaw),  cos(syaw))  -> (X, Z)
    #   局部左: (-cos(syaw), sin(syaw))  -> (X, Z)
    # syaw = slam_anchor 航向 + ODOM_YAW_OFFSET (相机/底盘相对 SLAM 世界系固定偏置).
    syaw = slam_anchor[2] + ODOM_YAW_OFFSET
    cs = math.cos(syaw)
    sn = math.sin(syaw)
    nx = slam_anchor[0] + f * sn - l * cs
    nz = slam_anchor[1] + f * cs + l * sn
    # 航向: 锚定 SLAM yaw + 安装偏置 + odom 相对偏航积分 (乘 yaw_sign 校正旋向)。
    # odom yaw 稳定不漂, 积分可靠; yaw_sign 手动指定 (默认 -1, 反向改 +1)。
    nyaw = slam_anchor[2] + ODOM_YAW_OFFSET + yaw_sign * _angle_diff(odom_now[2], odom_anchor[2])
    return (nx, nz, nyaw)


# 累积地图增量缓存 (按 slam 实例区分), 避免每次 get_map 全量重算
# (地图点随时间膨胀到百万级, 全量重算 + 拼接很慢)。搜索回退时只取新增关键帧。
_MAP_CACHE = {}


def _get_accumulated_map(slam, conf_threshold=1.5):
    """增量获取累积地图点云 (世界系, (N,3) float64), 供 extract_target_3d 回退搜索。

    用 get_keyframe_points_since 仅取新增关键帧并缓存拼接; 不触碰 SLAM 后端。
    返回 None 表示尚无关键帧。
    """
    global _MAP_CACHE
    key = id(slam)
    entry = _MAP_CACHE.get(key)
    if entry is None:
        entry = {"idx": 0, "pts": None}
        _MAP_CACHE[key] = entry
    new_pts, n = slam.get_keyframe_points_since(entry["idx"], conf_threshold=conf_threshold)
    if new_pts is not None and len(new_pts) > 0:
        if entry["pts"] is None:
            entry["pts"] = new_pts
        else:
            entry["pts"] = np.concatenate([entry["pts"], new_pts], axis=0)
    entry["idx"] = n
    return entry["pts"]


def extract_target_3d_from_snapshot(
        bbox, pointcloud_2d, image_shape, camera_pos_3d, nav_y=None,
        max_dist=5.0, map_max_dist=8.0, map_lateral=0.5,
        allow_map_fallback=True, map_points=None,
        debugger=None, debug_step=None):
    """从提交时快照提取目标 3D 世界坐标。

    路径1 (免搜索, 目标在当前帧时): 直接切片 get_pointcloud_2d (已是世界系,
    且与图像逐像素对齐), 按距离 0.3~max_dist / 高度 nav_y±0.8 过滤取 median。
    这就是"目标在当前帧 -> 直接用世界坐标"。

    路径2 (需搜索, 目标在当前帧无有效深度时: 遮挡/超距): 在当前帧 bbox 中心
    附近找有效点作为"视线方向"锚点 (这些点已是世界坐标, 完全免内参/坐标约定),
    再到累积地图里沿该视线 ±map_lateral 容差找最近表面点 (取视线近端一半,
    避免目标背后墙体被当作目标)。

    allow_map_fallback=False (由调用方 FINAL_ADJUST 持续检测阶段传入): 禁用路径2,
    仅用当前帧路径1。目标近处时路径1 通常已足够且更准; 路径2 在深度缺失时会搜到
    背景墙/远处, 反而污染 EMA —— 此时宁可返回 None (调用方保持旧终点), 也不引入猜测。
    本函数不访问 SLAM，供异步 VLM 返回后在主线程使用；bbox、逐像素点云、
    图像尺寸和相机位置必须来自同一次任务提交快照。map_points 可使用结果消费时
    主线程取得的最新累积地图，但射线原点与方向仍由提交快照确定。
    """
    if pointcloud_2d is None:
        return None
    img_pc = np.asarray(pointcloud_2d)
    if img_pc.ndim != 3 or img_pc.shape[2] != 3:
        return None
    try:
        cam_pos = (None if camera_pos_3d is None else
                   np.asarray(camera_pos_3d, dtype=np.float32).reshape(3))
    except (TypeError, ValueError):
        return None
    h_pc, w_pc, _ = img_pc.shape
    if h_pc <= 0 or w_pc <= 0:
        return None

    # bbox 缩放到 SLAM 处理分辨率 (供两条路径共用)
    if image_shape is not None and len(image_shape) >= 2:
        h_img, w_img = int(image_shape[0]), int(image_shape[1])
        if h_img > 0 and w_img > 0:
            sx = w_pc / w_img
            sy = h_pc / h_img
            bbox = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)

    # --- 路径1: 当前帧直接切片 ---
    x1 = max(0, min(int(bbox[0]), w_pc - 1))
    y1 = max(0, min(int(bbox[1]), h_pc - 1))
    x2 = max(0, min(int(bbox[2]), w_pc - 1))
    y2 = max(0, min(int(bbox[3]), h_pc - 1))
    region = img_pc[y1:y2 + 1, x1:x2 + 1]
    pts = region.reshape(-1, 3)
    valid = np.any(pts != 0, axis=1)
    pts = pts[valid]
    if len(pts) > 0:
        if cam_pos is not None:
            dists = np.linalg.norm(pts - cam_pos, axis=1)
            mask = (dists > 0.3) & (dists < max_dist)
            if nav_y is not None:
                mask = mask & (np.abs(pts[:, 1] - nav_y) < 0.8)
            pts = pts[mask]
            # 严格过滤后没点 -> 放宽高度再试一次
            if len(pts) == 0 and nav_y is not None:
                dists2 = np.linalg.norm(region.reshape(-1, 3)[valid] - cam_pos, axis=1)
                mask2 = (dists2 > 0.3) & (dists2 < max_dist)
                pts = region.reshape(-1, 3)[valid][mask2]
        if len(pts) > 0:
            if debugger is not None:
                try:
                    debugger.record_coord({
                        "stage": "pc_region",
                        "bbox": [float(bbox[0]), float(bbox[1]),
                                 float(bbox[2]), float(bbox[3])],
                        "cam_pos": ([float(v) for v in cam_pos]
                                    if cam_pos is not None else None),
                        "n_pts": int(len(pts)),
                        "X": [float(pts[:, 0].min()), float(pts[:, 0].max()),
                              float(np.median(pts[:, 0]))],
                        "Z": [float(pts[:, 2].min()), float(pts[:, 2].max()),
                              float(np.median(pts[:, 2]))],
                        "Y": [float(pts[:, 1].min()), float(pts[:, 1].max()),
                              float(np.median(pts[:, 1]))],
                    }, step=debug_step)
                except Exception:
                    pass
            target = np.median(pts, axis=0)
            return tuple(float(v) for v in target)

    # --- 路径2: 地图回退 (当前帧该区域无有效深度) ---
    if not allow_map_fallback:
        # 终调持续检测阶段: 不用路径2, 直接返回 None (调用方保持旧终点)
        return None
    bcx = 0.5 * (bbox[0] + bbox[2])
    bcy = 0.5 * (bbox[1] + bbox[3])
    u0 = int(np.clip(bcx, 0, w_pc - 1))
    v0 = int(np.clip(bcy, 0, h_pc - 1))
    R_s = max(15, int(0.10 * min(h_pc, w_pc)))
    r0, r1 = max(0, v0 - R_s), min(h_pc, v0 + R_s + 1)
    c0, c1 = max(0, u0 - R_s), min(w_pc, u0 + R_s + 1)
    patch = img_pc[r0:r1, c0:c1].reshape(-1, 3)
    pvalid = np.any(patch != 0, axis=1)
    if np.any(pvalid):
        p_anchor = np.median(patch[pvalid], axis=0)
        if cam_pos is not None:
            raydir = p_anchor - cam_pos
            nrm = float(np.linalg.norm(raydir))
            if nrm > 1e-6:
                raydir = raydir / nrm
                map_pts = map_points
                if map_pts is not None and len(map_pts) > 0:
                    map_pts = np.asarray(map_pts)
                    vec = map_pts - cam_pos
                    t = vec @ raydir                       # 沿视线深度
                    perp = np.linalg.norm(vec - t[:, None] * raydir, axis=1)
                    mask = (t > 0.3) & (t < map_max_dist) & (perp < map_lateral)
                    if np.count_nonzero(mask) >= 5:
                        sel = map_pts[mask]
                        tt = t[mask]
                        # 取视线近端一半 (前表面), 避免目标背后墙体被选中
                        order = np.argsort(tt)
                        half = order[: max(5, len(order) // 2)]
                        target = np.median(sel[half], axis=0)
                        return tuple(float(v) for v in target)
    return None


def extract_target_3d(slam, bbox, nav_y=None, max_dist=5.0,
                      map_max_dist=8.0, map_lateral=0.5,
                      allow_map_fallback=True,
                      camera_pos_3d=None, debugger=None):
    """同步兼容包装；新异步调用应使用 extract_target_3d_from_snapshot。"""
    img_pc = slam.get_pointcloud_2d()
    if img_pc is None:
        return None
    img = slam.get_img()
    image_shape = None if img is None else tuple(img.shape[:2])
    if camera_pos_3d is None:
        pose_full = slam.get_pose_full()
        camera_pos_3d = None if pose_full is None else tuple(pose_full[:3])
    result = extract_target_3d_from_snapshot(
        bbox=bbox,
        pointcloud_2d=img_pc,
        image_shape=image_shape,
        camera_pos_3d=camera_pos_3d,
        nav_y=nav_y,
        max_dist=max_dist,
        map_max_dist=map_max_dist,
        map_lateral=map_lateral,
        allow_map_fallback=allow_map_fallback,
        map_points=None,
        debugger=debugger,
    )
    if result is not None or not allow_map_fallback:
        return result
    return extract_target_3d_from_snapshot(
        bbox=bbox,
        pointcloud_2d=img_pc,
        image_shape=image_shape,
        camera_pos_3d=camera_pos_3d,
        nav_y=nav_y,
        max_dist=max_dist,
        map_max_dist=map_max_dist,
        map_lateral=map_lateral,
        allow_map_fallback=True,
        map_points=_get_accumulated_map(slam),
        debugger=debugger,
    )


# ============================================================
#  丝状障碍滤波 (占据栅格开运算 + Open3D 半径密度)
# ============================================================
FILAMENT_CELL = 0.05            # 占据栅格分辨率 (m)
FILAMENT_SE_RADIUS = 0.06       # 形态学结构元半径 (m) -> 直径 ~12cm
FILAMENT_DENSITY_RADIUS = 0.10  # Open3D 半径密度滤波半径 (m)
FILAMENT_DENSITY_NB = 4         # 半径内最少邻居数, 低于则视为孤立丝状点
FILAMENT_DOWNSAMPLE_VOXEL = 0.04  # 半径滤波前的体素降采样尺寸 (m); 0=禁用


def filter_filament_obstacles(points, cell=FILAMENT_CELL,
                              se_radius=FILAMENT_SE_RADIUS,
                              density_radius=FILAMENT_DENSITY_RADIUS,
                              density_nb=FILAMENT_DENSITY_NB,
                              downsample_voxel=FILAMENT_DOWNSAMPLE_VOXEL):
    """去除细小丝状 / 孤立障碍点, 供寻路与可视化共用。

    两步: (1) Open3D 半径密度滤波剔除孤立散点/长线缆; (2) X-Z 占据栅格 +
    形态学开运算 (scipy) 剔除比结构元更细的连续细丝。
    仅作用于导出的 obstacle_points, 不触碰 SLAM 后端 (X_canon/T_WC)。
    坐标约定 points[:,0]=X, points[:,1]=Y(高度), points[:,2]=Z。
    """
    if points is None or len(points) == 0:
        return points

    # 步骤1: 半径密度滤波 (Open3D, 去孤立散丝 / 稀疏长线缆)
    if density_radius > 0 and density_nb > 0:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                np.ascontiguousarray(points[:, :3], dtype=np.float64))
            # 体素降采样前置: get_map() 拼所有关键帧, 点数随时间单调增长,
            # 而 remove_radius_outlier = 建 KD-tree O(NlogN) + 逐点半径查询,
            # N 越大越慢且整段独占 GIL -> 饿死 MotionThread -> cmd_vel 断流抖动.
            # 先降采样把点数砍一个数量级, 单次耗时随之从 ~300ms 压到 ~20-40ms.
            if downsample_voxel > 0:
                pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
            pcd, _ = pcd.remove_radius_outlier(nb_points=density_nb,
                                               radius=density_radius)
            pts = np.asarray(pcd.points, dtype=np.float64)
            if len(pts) == 0:
                return pts
            points = pts
        except Exception:
            pass  # Open3D 异常时退化到仅栅格开运算

    # 步骤2: X-Z 占据栅格 + 形态学开运算 (scipy, 去细丝)
    px = points[:, 0]
    pz = points[:, 2]
    minx, maxx = px.min(), px.max()
    minz, maxz = pz.min(), pz.max()
    nx = int(np.ceil((maxx - minx) / cell)) + 1
    nz = int(np.ceil((maxz - minz) / cell)) + 1
    ix = np.floor((px - minx) / cell).astype(np.int64)
    iz = np.floor((pz - minz) / cell).astype(np.int64)
    occ = np.zeros((nz, nx), dtype=bool)
    occ[iz, ix] = True

    k = max(1, int(round(se_radius / cell)))
    structure = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
    opened = ndimage.binary_opening(occ, structure=structure)

    keep = opened[iz, ix]
    return points[keep]


def get_obstacle_points(slam, nav_y, y_above=OBS_Y_ABOVE, y_below=OBS_Y_BELOW,
                        map_pts=None):
    """高度带投影获取障碍物点 (横截面相对范围默认取 nav_constants.OBS_Y_ABOVE/BELOW)。

    map_pts: 可选, 传入已取出的全图点 (N,3) 以复用, 避免重复调用
    slam.get_map() 重建整图。主循环每帧统一取一次后, 所有调用点
    传同一个 map_pts, 显著减少 10Hz 下的点云重建开销。
    """
    if nav_y is None:
        return np.empty((0, 3))
    if map_pts is None:
        points, _ = slam.get_map()
    else:
        points = map_pts
    if points is None or len(points) == 0:
        return np.empty((0, 3))
    y_low = nav_y - y_above
    y_high = nav_y + y_below
    mask = (points[:, 1] >= y_low) & (points[:, 1] <= y_high)
    pts = points[mask]
    # return filter_filament_obstacles(pts)
    return pts


def query_frontier_for_viz(memory, pose_xz, nav_y, obstacle_points,
                           robot_radius=None, yaw=None):
    """调用 memory.get_nearest_frontier_grid, 仅供绘制。

    yaw: 航向 (rad), 传入后 BFS 优先扩展前方邻格 (同距离前向偏好)。
    返回 dict 含结果 + 分步耗时 (ms)。
    """
    empty = {
        "frontier_xz": None,
        "frontier_dist_m": None,
        "frontier_ms": None,
        "n_walked": 0,
        "n_cells": 0,
        "message": "skip",
        "t_occ_ms": 0.0,
        "t_walked_ms": 0.0,
        "t_cand_ms": 0.0,
        "t_paint_ms": 0.0,
        "t_bfs_ms": 0.0,
    }
    if memory is None:
        empty["message"] = "memory is None"
        return empty
    if pose_xz is None:
        empty["message"] = "pose_xz is None"
        return empty
    if not hasattr(memory, "get_nearest_frontier_grid"):
        empty["message"] = "no get_nearest_frontier_grid API (sync memory module?)"
        return empty
    try:
        kwargs = {
            "pose_xz": (float(pose_xz[0]), float(pose_xz[1])),
            "nav_y": float(nav_y) if nav_y is not None else 0.0,
            "obstacle_points": obstacle_points,
        }
        if yaw is not None:
            kwargs["yaw"] = float(yaw)
        if robot_radius is not None:
            kwargs["robot_radius"] = float(robot_radius)
        res = memory.get_nearest_frontier_grid(**kwargs)
        if res is None:
            empty["message"] = "API returned None"
            return empty
        found = bool(getattr(res, "found", False))
        out = {
            "frontier_xz": res.xz if found else None,
            "frontier_dist_m": (float(res.dist_m) if found else None),
            "frontier_ms": float(getattr(res, "elapsed_ms", 0.0) or 0.0),
            "n_walked": int(getattr(res, "n_walked", 0) or 0),
            "n_cells": int(getattr(res, "n_cells", 0) or 0),
            "message": str(getattr(res, "message", "") or
                           ("ok" if found else "not found")),
            "t_occ_ms": float(getattr(res, "t_occ_ms", 0.0) or 0.0),
            "t_walked_ms": float(getattr(res, "t_walked_ms", 0.0) or 0.0),
            "t_cand_ms": float(getattr(res, "t_cand_ms", 0.0) or 0.0),
            "t_paint_ms": float(getattr(res, "t_paint_ms", 0.0) or 0.0),
            "t_bfs_ms": float(getattr(res, "t_bfs_ms", 0.0) or 0.0),
            # Grid debug overlay
            "grid_occ_inf": getattr(res, "occ_inf", None),
            "grid_walked": getattr(res, "walked_grid", None),
            "grid_n": int(getattr(res, "grid_n", 0) or 0),
            "grid_origin_x": float(getattr(res, "grid_origin_x", 0.0) or 0.0),
            "grid_origin_z": float(getattr(res, "grid_origin_z", 0.0) or 0.0),
            "grid_res": float(getattr(res, "grid_res", 0.10) or 0.10),
        }
        return out
    except Exception as e:
        print(f"[frontier] get_nearest_frontier_grid failed: {e}")
        empty["message"] = f"exception: {e}"
        return empty


# ============================================================
#  自动决策辅助 (nav_auto / nav_auto1 / nav_mock / nav_depth_mock 共用)
#  原分散在多个 nav 脚本各抄一份, 现统一收归此处去重。
# ============================================================
def _goal_walked(mem, x, z, nav_y):
    """查询目标点 (x,z) 是否已走过 (在 walked 记忆圆内且离边界 > ROBOT_RADIUS).

    返回 True=已走过(不应作为 goal/patrol/sub-opt);
          False=未走过(可作为 goal);
    mem 为 None 或 memory 关闭时返回 False(允许作为 goal)。
    """
    if mem is None or not getattr(mem, "enable", True):
        return False
    try:
        p = np.array([x, (nav_y if nav_y is not None else 0.0), z],
                     dtype=np.float32)
        return mem.query_walked(p).score > 0
    except Exception:
        return False


def parse_vlm_direction(answer):
    """把 VLM 方向回答解析为 j/k/l/f/none。"""
    if answer is None or str(answer).startswith("error"):
        return "none"
    a = str(answer).lower()
    if "left" in a:
        return "j"
    if "right" in a:
        return "l"
    if "front" in a or "forward" in a or "center" in a or "middle" in a:
        return "k"
    return "f"


def auto_ask_direction(vlm_detector, img):
    """同步兼容包装：在 VLM_INQUIRY 请求并解析方向。

    返回 'j' / 'k' / 'l' (分别对应 left/right/front, 2D 左/中/右),
    或 'f' (VLM 回答非三选项 -> 视为最终目标 F),
    或 'none' (VLM 出错/无回答 -> 调用方应超时重试)。
    """
    if img is None or vlm_detector is None:
        return "none"
    return parse_vlm_direction(
        vlm_detector.ask(img, AUTO_VLM_DIR_PROMPT))


def _presence_is_yes(ans):
    """解析 VLM 存在性问询 (VLM_PRESENCE_PROMPT) 的 yes/no 回答。

    返回 True 表示画面中确认含目标 (应继续检测), False 表示无/未知。
    """
    if not isinstance(ans, str):
        return False
    s = ans.strip().lower()
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    # 退化容错: 含 'yes' 字样也算 (模型偶尔多写说明)
    return "yes" in s


def auto_trigger_patrol(slam, cur_pose, nav_y, obs_current):
    """F 点逻辑: 取最近 frontier 作为 **patrol 目标** (PATROL_PLAN)。

    F 永远是 frontier 巡逻目标, 不是最终目标 —— 最终目标只来自 object detection
    检测到并走过去 (周期检测块设 final_target -> FINAL_PLAN -> 到达 DONE)。

    返回 (goal_xz, NavState.PATROL_PLAN); 无 frontier / memory 未就绪 -> (None, None)。
    """
    mem = slam.memory
    if mem is None or cur_pose is None:
        return (None, None)
    fr = query_frontier_for_viz(
        mem,
        pose_xz=(cur_pose[0], cur_pose[1]),
        nav_y=nav_y if nav_y is not None else 0.0,
        obstacle_points=obs_current,
        robot_radius=NAV_CLEARANCE,
        yaw=float(cur_pose[2]),
    )
    fxz = fr.get("frontier_xz")
    if fxz is None:
        return (None, None)
    fx, fz = float(fxz[0]), float(fxz[1])
    return ((fx, fz), NavState.PATROL_PLAN)
