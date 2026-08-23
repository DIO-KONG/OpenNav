#!/usr/bin/env python
"""状态机数据流契约：单帧不可变环境输入 (FrameSnapshot) 与控制输出指令 (StateDecision)。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nav_helpers import ObstacleSnapshot
    from fsm.base_state import BaseNavState


@dataclass(frozen=True)
class FrameSnapshot:
    """单帧环境不可变快照 (每轮主循环开始时统一采集一次)。"""
    step: int
    now: float
    dt: float
    cur_pose: Optional[tuple]          # (x, z, yaw) 地面坐标
    raw_pose: Optional[tuple]          # SLAM 当前帧原始位姿
    kf_pose: Optional[tuple]           # SLAM 关键帧位姿
    slam_mode: Any                     # Mode 枚举 (INIT / TRACKING / RELOC / TERMINATED)
    odom_pose: Optional[tuple]         # 里程计推算位姿 (x, z, yaw)
    nav_y: float                       # 导航高度平面 Y
    obstacle_snapshot: Any             # ObstacleSnapshot (完整障碍点 + KD-Tree)
    obs_current: np.ndarray            # 当前障碍物 numpy (N, 3)
    n_map_points: int                  # 累积点云总数
    halted: bool                       # 急停状态 (Space 键触发)
    img: Optional[np.ndarray] = None   # 原始 RGB 图像


@dataclass(frozen=True)
class StateDecision:
    """状态单步演化输出的控制决策。"""
    next_state: Optional[Type[BaseNavState]] = None   # 目标跳转状态类 (None 表示保持当前状态)
    linear: float = 0.0                               # 期望线速度 (m/s)
    angular: float = 0.0                              # 期望角速度 (rad/s)
    command_desc: str = "idle"                        # 终端与 HUD 显示动作描述
    reset_motion: bool = False                        # 是否立即停止底盘运动
