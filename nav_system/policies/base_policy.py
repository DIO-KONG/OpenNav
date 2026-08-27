#!/usr/bin/env python
"""导航策略抽象基类：隔离 Object / Open / Frontier 三种导航任务的特化行为。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fsm.context import NavContext
    from fsm.decision import FrameSnapshot, StateDecision


class BaseNavigationPolicy(ABC):
    """导航策略基类。"""
    name: str = "base"
    default_boot_scan_done: bool = False

    @abstractmethod
    def handle_inquiry_step(self, ctx: NavContext, snapshot: FrameSnapshot) -> StateDecision:
        """INQUIRY 状态下如何决策巡逻目标 (问询 VLM 方向 vs 直接选 Frontier)。"""
        raise NotImplementedError

    @abstractmethod
    def should_accept_final_adjust(self, ctx: NavContext, snapshot: FrameSnapshot, target_3d: tuple) -> bool:
        """检测到有效 3D 目标后，是否允许进入 FINAL_ADJUST 状态。"""
        raise NotImplementedError

    @abstractmethod
    def should_query_presence(self, ctx: NavContext, in_adjust: bool) -> bool:
        """当前帧目标检测是否需要携带 presence 存在性前置确认。"""
        raise NotImplementedError

    def register_presence(self, is_present: bool, now: float) -> None:
        """注册一次 presence 测量结果 (供滑动窗口策略统计)。"""
        pass

    def presence_locked(self, now: float) -> bool:
        """RELOC spin 是否允许因 presence 进入 detect_hold。

        Object/Frontier: 单次 yes 即可 (默认 True，由调用方再与 present 合取)。
        Open: 覆盖为滑动窗口锁定。
        """
        return True

    def reset_presence_window(self, reason: str = "") -> None:
        """重置存在性滑动窗口 (切换状态时调用)。"""
        pass
