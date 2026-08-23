#!/usr/bin/env python
"""导航状态抽象基类：规范状态生命周期接口 (on_enter, on_update, on_exit)。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsm.context import NavContext
    from fsm.decision import FrameSnapshot, StateDecision


class BaseNavState(ABC):
    """所有导航具体状态的抽象基类。"""
    name: str = "BASE"

    def on_enter(self, ctx: NavContext, snapshot: FrameSnapshot) -> None:
        """状态切入钩子：负责初始化本状态私有成员、重置计数器/计时器/平滑器。"""
        pass

    @abstractmethod
    def on_update(self, ctx: NavContext, snapshot: FrameSnapshot) -> StateDecision:
        """单步演化核心逻辑：执行算法计算、检测环境、输出底盘速度及状态转移目标。"""
        raise NotImplementedError

    def on_exit(self, ctx: NavContext, snapshot: FrameSnapshot) -> None:
        """状态退出钩子：负责清理临时计算资源、注销未完成的短效事务。"""
        pass

    def __repr__(self) -> str:
        return f"<NavState:{self.name}>"
