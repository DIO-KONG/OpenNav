#!/usr/bin/env python
"""导航状态机的类型化运行时上下文。

本模块只管理状态和状态转换所需的数据，不访问 SLAM、VLM 或运动线程。
外部 I/O 仍由 nav_auto 主循环拥有。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nav_control import NavState
from nav_path import PathCheckResult, check_nav_path


@dataclass
class RouteContext:
    semantic_target: Optional[tuple] = None
    navigation_goal: Optional[tuple] = None
    path: Optional[list] = None
    path_idx: int = 0
    source: Optional[str] = None
    plan_fail_count: int = 0
    retry_at: float = 0.0

    def clear_path(self, clear_goal=False):
        self.path = None
        self.path_idx = 0
        if clear_goal:
            self.navigation_goal = None

    def install_path(self, path, navigation_goal):
        self.path = path
        self.path_idx = 0
        self.navigation_goal = navigation_goal
        self.plan_fail_count = 0
        self.retry_at = 0.0


@dataclass
class EscapeContext:
    phase: Optional[str] = None
    resume_state: Optional[NavState] = None
    semantic_target: Optional[tuple] = None
    navigation_goal: Optional[tuple] = None
    fail_count: int = 0
    direction: Optional[tuple] = None
    path0_anchor: Optional[tuple] = None
    path0_line: Any = None
    path1_yaw: Optional[float] = None

    def reset(self):
        self.phase = None
        self.resume_state = None
        self.semantic_target = None
        self.navigation_goal = None
        self.fail_count = 0
        self.direction = None
        self.path0_anchor = None
        self.path0_line = None
        self.path1_yaw = None


@dataclass
class PreturnContext:
    active: bool = False
    start_pose: Optional[tuple] = None
    slam_anchor: Optional[tuple] = None
    odom_anchor: Optional[tuple] = None
    active_time: float = 0.0
    last_tick: float = 0.0
    resume_plan_state: Optional[NavState] = None


@dataclass
class RelocContext:
    phase: Optional[str] = None
    started_at: float = 0.0
    previous_state: Optional[NavState] = None
    give_up: bool = False


@dataclass
class VlmContext:
    epoch: int = 0
    request_id: int = 0
    latest: tuple = (None, None)
    presence_confirmed: bool = False
    auto_asked: bool = False
    reask_pending: bool = False


@dataclass(frozen=True)
class FrameSnapshot:
    step: int
    timestamp: float
    pose: Optional[tuple]
    odom_pose: Optional[tuple]
    slam_mode: Any
    nav_y: Optional[float]
    obstacle_snapshot: Any
    halted: bool = False
    image: Any = None


@dataclass(frozen=True)
class NavigationDecision:
    next_state: Optional[NavState] = None
    linear: float = 0.0
    angular: float = 0.0
    start_motion: bool = False
    command: str = "stop"
    reason: str = ""


@dataclass
class _PathValidationCache:
    key: Optional[tuple] = None
    result: PathCheckResult = field(
        default_factory=lambda: PathCheckResult(False, "not_validated"))


class NavigationRuntime:
    """集中管理导航上下文和可缓存的无副作用判断。"""

    def __init__(self, state=NavState.WAITING):
        self.state = state
        self.patrol = RouteContext()
        self.final = RouteContext()
        self.escape = EscapeContext()
        self.preturn = PreturnContext()
        self.reloc = RelocContext()
        self.vlm = VlmContext()
        self._path_validation = _PathValidationCache()

    def transition(self, state):
        self.state = state

    def invalidate_path_validation(self):
        self._path_validation = _PathValidationCache()

    def validate_remaining_path(self, path, path_idx, current_pose,
                                obstacle_snapshot):
        """同一路径、索引和障碍版本只验证一次。

        障碍未更新时，机器人沿已验证航段前进只会缩短该航段，不需要在
        10Hz 主循环中重复扫描；地图版本或 waypoint 改变时自动复检。
        """
        key = (id(path), int(path_idx), int(obstacle_snapshot.revision))
        if self._path_validation.key != key:
            self._path_validation.key = key
            self._path_validation.result = check_nav_path(
                path,
                obstacle_snapshot,
                start_idx=path_idx,
                current_pose=current_pose,
            )
        return self._path_validation.result

