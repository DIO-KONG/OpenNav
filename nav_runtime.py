#!/usr/bin/env python
"""导航状态机的类型化运行时上下文。

本模块集中保存三个入口 (navvlm_open / navvlm_object / nav_frontier) 共用的
可变状态。主循环仍拥有 SLAM、VLM worker、底盘线程等 I/O；这里只做状态
读写和可缓存的无副作用判断。

路径在 PATROL/FINAL 之间共用一份 (同时只有一条在跑的路)，与当前入口行为一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nav_control import NavState
from nav_path import PathCheckResult, check_nav_path


@dataclass
class RouteContext:
    """巡逻或终局目标。path 本身在 NavigationRuntime 顶层共享。"""

    semantic_target: Optional[tuple] = None
    navigation_goal: Optional[tuple] = None
    navigation_goal_target: Optional[tuple] = None
    source: Optional[str] = None
    plan_fail_count: int = 0
    retry_at: float = 0.0
    frontier_retry_at: float = 0.0
    blocked_diverted: bool = False
    adjust_plan_fail_count: int = 0
    adjust_started_at: float = 0.0
    adjust_frames: int = 0
    from_reloc: bool = False
    bbox_anchor: Any = None
    last_goal_shift_t: float = 0.0

    def clear_goal(self):
        self.semantic_target = None
        self.navigation_goal = None
        self.navigation_goal_target = None
        self.source = None
        self.plan_fail_count = 0
        self.retry_at = 0.0
        self.frontier_retry_at = 0.0
        self.blocked_diverted = False
        self.adjust_plan_fail_count = 0
        self.adjust_started_at = 0.0
        self.adjust_frames = 0
        self.from_reloc = False
        self.bbox_anchor = None
        self.last_goal_shift_t = 0.0


@dataclass
class EscapeContext:
    phase: Optional[str] = None
    resume_state: Optional[NavState] = None
    semantic_target: Optional[tuple] = None
    navigation_goal: Optional[tuple] = None
    fail_count: int = 0
    direction: Optional[tuple] = None
    via_path0: bool = False
    path0_drive_mode: Optional[str] = None
    path0_anchor: Optional[tuple] = None
    path0_path: Any = None
    path0_line: Any = None
    path0_aligned: bool = False
    path1_yaw: Optional[float] = None
    replan_from_semantic: bool = False

    def reset(self, clear_path0=True):
        self.phase = None
        self.resume_state = None
        self.semantic_target = None
        self.navigation_goal = None
        self.fail_count = 0
        self.direction = None
        self.replan_from_semantic = False
        if clear_path0:
            self.via_path0 = False
            self.path0_drive_mode = None
            self.path0_anchor = None
            self.path0_path = None
            self.path0_line = None
            self.path0_aligned = False
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

    def reset(self):
        self.active = False
        self.start_pose = None
        self.slam_anchor = None
        self.odom_anchor = None
        self.active_time = 0.0
        self.last_tick = 0.0
        self.resume_plan_state = None


@dataclass
class RelocContext:
    phase: Optional[str] = None
    started_at: float = 0.0
    t: float = 0.0
    previous_state: Optional[NavState] = None
    give_up: bool = False
    probe_t: float = 0.0
    max_time: float = 120.0


@dataclass
class VlmContext:
    epoch: int = 0
    request_id: int = 0
    latest: tuple = (None, None)
    presence_confirmed: bool = False
    auto_asked: bool = False
    ask_t: float = 0.0
    det_t: float = 0.0
    reask_pending: bool = False
    frontier_query_t: float = 0.0
    PRESENCE_WINDOW: int = 10
    PRESENCE_SUSTAIN: float = 5.0
    PRESENCE_RATIO: float = 0.8
    presence_samples: list = field(default_factory=list)

    def invalidate(self):
        self.epoch += 1

    def register_presence(self, is_present, now):
        self.presence_samples.append((now, bool(is_present)))
        over = len(self.presence_samples) - (self.PRESENCE_WINDOW + 1)
        if over > 0:
            del self.presence_samples[:over]

    def presence_ratio(self, now):
        n = len(self.presence_samples)
        if n < self.PRESENCE_WINDOW:
            return None
        win = self.presence_samples[-self.PRESENCE_WINDOW:]
        yes = sum(1 for _, present in win if present)
        return yes / self.PRESENCE_WINDOW

    def presence_span(self, now):
        if len(self.presence_samples) >= self.PRESENCE_WINDOW:
            return now - self.presence_samples[-self.PRESENCE_WINDOW][0]
        if self.presence_samples:
            return now - self.presence_samples[0][0]
        return 0.0

    def presence_locked(self, now):
        ratio = self.presence_ratio(now)
        if ratio is None or ratio < self.PRESENCE_RATIO:
            return False
        win = self.presence_samples[-self.PRESENCE_WINDOW:]
        return (now - win[0][0]) >= self.PRESENCE_SUSTAIN

    def reset_presence(self):
        self.presence_samples.clear()


@dataclass
class PoseFusionContext:
    source: str = "slam"
    use_odom: bool = False
    yaw_sign: float = -1.0
    smooth_pos: float = 0.30
    smooth_yaw: float = 0.30
    nav_smooth: Optional[tuple] = None
    anchor_kf: Any = None
    anchor_cur: Any = None
    kf_prev_pos: Optional[tuple] = None
    odom_anchor: Any = None
    slam_anchor: Any = None
    plan_slam_anchor: Any = None
    plan_odom_anchor: Any = None
    plan_pose: Any = None
    escaping: bool = False
    prev_escaping: bool = False
    post_escape: bool = False
    post_escape_t: float = 0.0


@dataclass
class ScanContext:
    boot_done: bool = False
    accumulated: float = 0.0
    prev_yaw: Optional[float] = None


@dataclass
class ObstacleContext:
    snapshot: Any = None
    points_id: Any = None
    revision: int = 0


@dataclass
class LoopDebug:
    prev_t: float = 0.0
    fps: float = 0.0
    nav_step_t: float = 0.0
    grab_ms: float = 0.0
    draw_ms: float = 0.0
    work_ms: float = 0.0
    vlog_t: float = 0.0
    last_goal_src: Any = None
    last_patrol_src: Any = None
    last_goal_tgt_key: Any = None
    invalid_replan_pending: bool = False
    invalid_replan_reason: Optional[str] = None
    prev_halt: bool = False


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


_FINAL_STATES = (
    NavState.FINAL_FOLLOW, NavState.FINAL_ADJUST, NavState.FINAL_PLAN)
_PATROL_STATES = (NavState.PATROL_FOLLOW, NavState.PATROL_PLAN)


class NavigationRuntime:
    """三个导航入口共用的可变状态。"""

    def __init__(self, state=NavState.WAITING, boot_scan_done=False):
        self.state = state
        self.path = None
        self.path_idx = 0
        self.goal_source = None
        self.nav_y = None
        self.step = 0
        self.last_cmd = "idle"
        self.d_tgt = None
        self.d_robot = None
        self.lookahead_target = None
        self.patrol = RouteContext()
        self.final = RouteContext()
        self.escape = EscapeContext()
        self.preturn = PreturnContext()
        self.reloc = RelocContext()
        self.vlm = VlmContext()
        self.pose = PoseFusionContext()
        self.scan = ScanContext(boot_done=boot_scan_done)
        self.obstacle = ObstacleContext()
        self.debug = LoopDebug()
        self._path_validation = _PathValidationCache()

    # -- 路径 -------------------------------------------------------
    def set_path(self, path, idx=0):
        self.path = path
        self.path_idx = idx
        self.invalidate_path_validation()

    def clear_path(self):
        self.set_path(None, 0)

    def transition(self, state):
        self.state = state

    def invalidate_path_validation(self):
        self._path_validation = _PathValidationCache()

    def validate_remaining_path(self, path, path_idx, current_pose,
                                obstacle_snapshot):
        """同一路径、索引和障碍版本只验证一次。"""
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

    # -- VLM --------------------------------------------------------
    def invalidate_vlm(self):
        self.vlm.invalidate()

    # -- 巡逻目标 ---------------------------------------------------
    def set_patrol_goal(self, ft, src="frontier"):
        self.invalidate_vlm()
        self.escape.reset(clear_path0=True)
        self.vlm.auto_asked = False
        self.patrol.semantic_target = ft
        self.patrol.source = src
        self.goal_source = src
        self.patrol.plan_fail_count = 0
        self.patrol.blocked_diverted = False
        self.patrol.navigation_goal = None
        self.patrol.navigation_goal_target = None
        self.clear_path()
        self.preturn.active = False
        self.state = NavState.PATROL_PLAN

    def abandon_patrol_to_inquiry(self):
        self.patrol.semantic_target = None
        self.patrol.source = None
        self.patrol.navigation_goal = None
        self.patrol.navigation_goal_target = None
        self.vlm.auto_asked = False
        self.vlm.reask_pending = False
        self.state = NavState.VLM_INQUIRY

    # -- 脱困 -------------------------------------------------------
    def reset_intrusion_escape(self, clear_path0=True):
        self.escape.reset(clear_path0=clear_path0)

    def begin_intrusion_escape(self, resume_state, target):
        if target is None:
            return False
        self.invalidate_vlm()
        self.escape.reset(clear_path0=True)
        self.escape.phase = "plan"
        self.escape.resume_state = resume_state
        self.escape.navigation_goal = (float(target[0]), float(target[1]))
        if resume_state in _FINAL_STATES and self.final.semantic_target is not None:
            self.escape.semantic_target = (
                float(self.final.semantic_target[0]),
                float(self.final.semantic_target[1]))
        elif (resume_state in _PATROL_STATES
              and self.patrol.semantic_target is not None):
            self.escape.semantic_target = (
                float(self.patrol.semantic_target[0]),
                float(self.patrol.semantic_target[1]))
        else:
            self.escape.semantic_target = None
        self.escape.replan_from_semantic = False
        self.clear_path()
        self.state = NavState.ESCAPE
        return True

    def start_preturn_odom_hold(self):
        if self.preturn.slam_anchor is None or self.preturn.odom_anchor is None:
            return False
        self.pose.slam_anchor = self.preturn.slam_anchor
        self.pose.odom_anchor = self.preturn.odom_anchor
        self.pose.post_escape = True
        import time
        self.pose.post_escape_t = time.time()
        return True

    def mark_invalid_nav_replan(self, reason):
        self.debug.invalid_replan_pending = True
        self.debug.invalid_replan_reason = str(reason)

    def consume_invalid_nav_replan(self):
        pending = self.debug.invalid_replan_pending
        reason = self.debug.invalid_replan_reason
        self.debug.invalid_replan_pending = False
        self.debug.invalid_replan_reason = None
        return pending, reason
