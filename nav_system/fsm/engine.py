#!/usr/bin/env python
"""导航状态机引擎：负责状态调度、生命周期钩子（on_exit / on_enter）触发与受控转移。"""
from __future__ import annotations

from typing import Type, Optional
from dataclasses import replace
from fsm.base_state import BaseNavState
from fsm.decision import FrameSnapshot, StateDecision
from fsm.context import NavContext


class NavigationStateMachine:
    """有限状态机调度执行器。"""

    def __init__(self, context: NavContext, initial_state_cls: Type[BaseNavState]):
        self.context = context
        self.current_state: BaseNavState = initial_state_cls()
        self._states_cache: dict[Type[BaseNavState], BaseNavState] = {
            initial_state_cls: self.current_state
        }
        self.step_count: int = 0

    def _get_or_create_state(self, state_cls: Type[BaseNavState]) -> BaseNavState:
        """获取或复用具体状态实例，避免高频创建对象。"""
        if state_cls not in self._states_cache:
            self._states_cache[state_cls] = state_cls()
        return self._states_cache[state_cls]

    def change_state(self, next_state_cls: Type[BaseNavState], snapshot: FrameSnapshot) -> None:
        """执行受控的状态转移，严格触发前态退出与新态进入生命周期钩子。"""
        if type(self.current_state) is next_state_cls:
            return

        prev_state = self.current_state
        next_state = self._get_or_create_state(next_state_cls)

        print(f"\033[92m[FSM] State Changed: {prev_state.name} -> {next_state.name}\033[0m")

        # 1. 触发前置状态退出清理
        prev_state.on_exit(self.context, snapshot)

        # 2. 状态指针切换
        self.current_state = next_state

        # 3. 触发后置状态进入初始化
        next_state.on_enter(self.context, snapshot)

    def step(self, snapshot: FrameSnapshot) -> StateDecision:
        """单步执行当前活跃状态的逻辑，并根据返回值处理转移。"""
        self.step_count += 1
        source_state = self.current_state
        decision = source_state.on_update(self.context, snapshot)

        # on_update 可能已经修改 path/path_idx，随后 change_state 的
        # on_exit/on_enter 还可能清理它们。先保存本帧命令对应的调试数据，
        # 供主循环与 overlay 使用，避免显示新状态却对应旧状态命令。
        decision = replace(
            decision,
            source_state_name=source_state.name,
            debug_path=(list(self.context.path) if self.context.path is not None else None),
            debug_path_idx=(
                int(self.context.path_idx) if self.context.path is not None else None
            ),
            debug_lookahead=self.context.lookahead_target,
        )

        # 若状态请求了跳转，则立刻执行生命周期转移
        if decision.next_state is not None:
            self.change_state(decision.next_state, snapshot)

        return decision
