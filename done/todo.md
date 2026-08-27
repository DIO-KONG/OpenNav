# 导航系统重构修复计划

## 任务背景

项目原始实现位于 `legacy/`，重构实现位于 `nav_system/`。重构将约 3000 行导航代码拆分为 FSM、Policy、路径规划和服务模块，但当前三种模式（Object、Open、Frontier）的状态机行为尚未与 legacy 完全对齐。

本计划基于对 `legacy/navvlm_object.py`、`legacy/navvlm_open.py`、`legacy/nav_frontier.py` 及 `nav_system/` 的静态对照，覆盖目标检测、开局扫描、VLM 问询、PATROL/FINAL 路径流程、ESCAPE、RELOC 和状态生命周期。修复目标是恢复 legacy 的状态转移、目标保留、失败重试和运动时序语义。

说明：`segment=None` 主要来自 `plan_goal_resolution()` 调用 `check_nav_segment()` 时未传 `segment_index`。这不是函数签名错位；应优先修复导致规划反复执行的状态逻辑，日志字段是否补充索引可作为独立低优先级事项处理。

## 修复子任务

### P0（已完成）

#### 1. 限制全局 auto_detect 的 FINAL_ADJUST 切入状态 ✅

- **修复指导**：只有 `SCANNING`、`VLM_INQUIRY`、`PATROL_PLAN`、`PATROL_FOLLOW` 可以被有效自动检测切入 `FINAL_ADJUST`。`WAITING`、`ESCAPE`、`FINAL_PLAN`、`FINAL_FOLLOW`、`DONE`、`RELOC` 中的迟到结果不得改变当前状态。
- **legacy 正确逻辑**：`legacy/navvlm_object.py` / `legacy/navvlm_open.py` / `legacy/nav_frontier.py` 关键字 `if state in (NavState.SCANNING, NavState.VLM_INQUIRY, NavState.PATROL_PLAN, NavState.PATROL_FOLLOW)`。
- **重构错误位置**：`nav_system/main.py` 关键字 `handle_global_auto_detection`、`should_accept_final_adjust`；当前在未检查当前状态的情况下调用 `fsm.change_state(FinalAdjustState, snapshot)`。

#### 2. RELOC 目标截获后设置 give-up/接管标志 ✅

- **修复指导**：RELOC 中检测到有效目标并切入 `FINAL_ADJUST` 后，必须设置 `ctx.reloc_giveup = True`（或等价接管状态），避免下一帧 SLAM 仍为 RELOC 时再次进入 RELOC。
- **legacy 正确逻辑**：关键字 `_reloc_giveup = True`，位于 `reloc_detect` 成功得到 3D 目标并切换 `NavState.FINAL_ADJUST` 的分支。
- **重构错误位置**：`nav_system/fsm/states/reloc.py` 关键字 `reloc_detect`、`FinalAdjustState`；当前设置 `ctx.final_from_reloc = True`，但没有设置 `ctx.reloc_giveup`。

#### 3. RELOC 恢复时正确保存 ESCAPE 的 resume 状态 ✅

- **修复指导**：进入 RELOC 前若当前状态是 `ESCAPE`，保存 `ctx.resume_state_cls`，清理脱困上下文；恢复时回到原本要恢复的 `PATROL_FOLLOW`、`FINAL_FOLLOW` 或对应规划状态，而不是重新进入 `EscapeState`。
- **legacy 正确逻辑**：关键字 `_reloc_saved_state = (_escape_resume_state if _escape_resume_state is not None else NavState.WAITING)`、`_reloc_prev_state = _reloc_saved_state`。
- **重构错误位置**：`nav_system/main.py` 关键字 `reloc_prev_state_cls = type(fsm.current_state)`；`nav_system/fsm/states/reloc.py` 关键字 `prev_cls`、`RELOC recovered`。

#### 4. RELOC 恢复时补齐 SCANNING 特殊分支 ✅

- **修复指导**：从 `SCANNING` 发生 RELOC 后，恢复 TRACKING 时按目标选择 `FINAL_PLAN`、`PATROL_PLAN` 或 `VLM_INQUIRY`，不要简单恢复为 `ScanningState`。
- **legacy 正确逻辑**：关键字 `elif _ps == NavState.SCANNING`，随后判断 `final_target`、`patrol_target`，否则进入 `VLM_INQUIRY`。
- **重构错误位置**：`nav_system/fsm/states/reloc.py` 关键字 `else: target_cls = prev_cls`；当前缺少针对 `ScanningState` 的目标分支。

#### 5. RELOC 超时/放弃后的正常流程清理 ✅

- **修复指导**：RELOC 超时或主动放弃后，清除失效的导航站位和路径；若原状态是 PATROL/FOLLOW 或 FINAL/FOLLOW/ADJUST，切回对应规划状态；复位 give-up 标志。
- **legacy 正确逻辑**：关键字 `if _reloc_giveup and (mode is None or mode.name != "RELOC")`、`patrol_goal_nav = None`、`final_goal_nav = None`、`state = NavState.PATROL_PLAN/FINAL_PLAN`。
- **重构错误位置**：`nav_system/fsm/states/reloc.py` 关键字 `RELOC timeout`；`nav_system/main.py` 仅检查 `context.reloc_giveup`，没有对应的放弃后清理分支。

### P1（已完成）

#### 6. Open 模式 RELOC presence 使用滑动窗口门控 ✅

- **修复指导**：Open 模式中单次 `present=True` 不得立即进入 `detect_hold`；必须满足 10 个样本、yes 比例至少 80%、时间跨度至少 5 秒后才停车并检测 bbox。Object/Frontier 保持 legacy 的单次 presence 语义。
- **legacy 正确逻辑**：`legacy/navvlm_open.py` 关键字 `_present_register`、`_present_locked`、`presence_yes_locked`；Object/Frontier 对应 `reloc_presence` 分支为单次 yes 即 `detect_hold`。
- **重构错误位置**：`nav_system/fsm/states/reloc.py` 关键字 `if present: self.phase = "detect_hold"`；当前没有按 policy 区分 Open 窗口锁定。

#### 7. FINAL_PLAN 连续失败时保留 final_target ✅

- **修复指导**：达到失败阈值时只清除失效的 `final_goal_nav`，保留 `final_target`，等待地图更新后重新解析规划；不得直接 `clear_final(clear_target=True)` 回到 WAITING。
- **legacy 正确逻辑**：`legacy/navvlm_object.py` / `legacy/navvlm_open.py` / `legacy/nav_frontier.py` 关键字 `FINAL 路径连续失败`、`final_goal_nav = None`、`_final_plan_retry_t`。
- **重构错误位置**：`nav_system/fsm/states/final_plan.py` 关键字 `plan_fail_cnt >= PLAN_FAIL_MAX`；当前调用 `ctx.clear_final(clear_target=True)` 并切换 `WaitingState`。

#### 8. FINAL_PLAN 实现失败重试冷却 ✅

- **修复指导**：使用 `retry_t` 或等价字段，失败后按 legacy 等待约 2 秒再重新规划；成功或新目标时清零冷却。
- **legacy 正确逻辑**：关键字 `_final_plan_retry_t`、`final plan retry cooldown`、`time.time() + 2.0`。
- **重构错误位置**：`nav_system/fsm/states/final_plan.py` 定义了 `retry_t` 但未使用，`on_update()` 每帧直接调用规划。

#### 9. FINAL_FOLLOW 路径耗尽但目标未到达时转 FINAL_PLAN ✅

- **修复指导**：`path_idx >= len(path)` 只能表示当前路径耗尽；若最终目标仍超过到达容差，应清路径并进入 `FINAL_PLAN`，只有目标/末点真正到达才进入 DONE。
- **legacy 正确逻辑**：关键字 `if path_idx >= len(path)` 位于 FINAL 跟随后的分支；`d_tgt <= PATROL_ARRIVE_EPS` 才 DONE，否则 `state = NavState.FINAL_PLAN`。
- **重构错误位置**：`nav_system/fsm/states/final_follow.py` 关键字 `if d_tgt <= ... or ... or ctx.path_idx >= len(path)`；当前直接返回 `DoneState`。

#### 10. PATROL_FOLLOW 路径耗尽时区分到达与重新规划 ✅

- **修复指导**：路径耗尽但 patrol 目标仍远时进入 `PatrolPlanState`；只有安全站位/末点到达才清除 patrol 目标并进入 `InquiryState`。
- **legacy 正确逻辑**：`legacy/*` 关键字 `if path_idx >= len(path)`，后续判断 `d_tgt <= _arrive_eps`，否则 `state = NavState.PATROL_PLAN`。
- **重构错误位置**：`nav_system/fsm/states/patrol_follow.py` 关键字 `d_tgt <= arrive_eps or d_end <= arrive_eps or ctx.path_idx >= len(path)`；当前将三者统一判为已到达。

#### 11. FOLLOW 路径不安全 cooldown 期间停车 ✅

- **修复指导**：路径检查不安全且尚未达到重规划冷却时间时，必须立即返回停车决策，禁止继续调用 `follow_path_step()`。
- **legacy 正确逻辑**：`legacy/navvlm_object.py` / `legacy/navvlm_open.py` / `legacy/nav_frontier.py` 关键字 `stop (replan cooldown)`。
- **重构错误位置**：`nav_system/fsm/states/patrol_follow.py`、`nav_system/fsm/states/final_follow.py` 关键字 `if not path_check.safe`；当前 cooldown 分支结束后仍落入跟随控制。

#### 12. ESCAPE 成功恢复后清理脱困上下文 ✅

- **修复指导**：path0 到达或 path1 对准后，除设置路径索引和恢复状态外，还应等价执行 legacy 的 `_reset_intrusion_escape(clear_path0=False)`，清除 resume/target/phase 等一次性上下文，同时保留必要的 path0 锚点信息。
- **legacy 正确逻辑**：关键字 `_reset_intrusion_escape(clear_path0=False)`、`_escaping = False`、`state = _resume_state`。
- **重构错误位置**：`nav_system/fsm/states/escape.py` 关键字 `path0 reached -> resume path`、`path1 aligned -> resume follow`；当前直接返回 `next_state`，未清理 `ctx.resume_state_cls` 等字段。

### P2（已完成）

#### 13. 新 patrol 目标时失效旧 VLM 结果 ✅

- **修复指导**：所有设置新 patrol 目标的入口统一执行 `ctx.invalidate_vlm("new_patrol_goal")`，并清理旧路径、站位和预转向状态，避免旧 auto_detect 结果在新巡逻周期中被应用。
- **legacy 正确逻辑**：关键字 `_set_patrol_goal()`、`_invalidate_vlm("new_patrol_goal")`。
- **重构错误位置**：`nav_system/policies/object_policy.py`、`nav_system/policies/frontier_policy.py`、`nav_system/fsm/states/patrol_plan.py` 设置 `patrol_target` 的分支；当前多数入口只调用 `clear_active_path()`。

#### 14. PATROL_PLAN 失败计数跨状态保留 ✅

- **修复指导**：不要在每次进入 `PatrolPlanState` 时无条件清零失败计数；只有新目标或成功规划时清零，保证 FOLLOW -> PLAN 后连续失败达到阈值时能切换 frontier。
- **legacy 正确逻辑**：关键字 `_patrol_plan_fail_cnt`；计数在 `PATROL_PLAN` 失败分支累加，在新目标/成功规划时清零。
- **重构错误位置**：`nav_system/fsm/states/patrol_plan.py` `on_enter()` 关键字 `self.plan_fail_cnt = 0`。

#### 15. FINAL_PLAN 失败计数跨 FOLLOW/PLAN 保留 ✅

- **修复指导**：与 legacy 一样，FINAL_FOLLOW 因路径失效转 FINAL_PLAN 时不要无条件丢失累计失败次数；仅成功规划、新目标或明确重新开始最终任务时清零。
- **legacy 正确逻辑**：关键字 `_final_plan_fail_cnt`、`final_rrt_failed`、`_final_plan_retry_t`。
- **重构错误位置**：`nav_system/fsm/states/final_plan.py` `on_enter()` 关键字 `self.plan_fail_cnt = 0`。

#### 16. FINAL_ADJUST 规划/跟随首帧时序对齐 ✅

- **修复指导**：对照 legacy 决定路径为空时本帧只规划、下一帧再跟随；同时避免 `on_exit()`、`reset_motion=True` 造成 FINAL_ADJUST -> FINAL_FOLLOW 的额外停车。保留跳过 path[0] 的修复，但不要改变后续状态时序。
- **legacy 正确逻辑**：`legacy/*` 关键字 `elif path is None`、`_final_plan_goal`、`_lock_final_adjust_if_ready`；路径重建后不会在同一分支继续执行跟随。
- **重构错误位置**：`nav_system/fsm/states/final_adjust.py` 关键字 `path is None or len(path) == 0`、`follow_path_step`、`on_exit`；当前规划成功后同帧继续跟随，并在状态切换时停车。

#### 17. FINAL_ADJUST 锁定判断统一到 legacy 时机 ✅

- **修复指导**：将锁定判断整理为与 legacy 一致的单一流程：到达 GOAL_NAV 后先停车再判断；未到达时先跟随再判断；FORCE_TIMEOUT 也遵循同一时序。避免函数开头提前锁定导致漏掉本帧控制或改变状态转换时机。
- **legacy 正确逻辑**：`legacy/*` 关键字 `_lock_final_adjust_if_ready`、`if d_tgt <= PATROL_ARRIVE_EPS or _de <= PATROL_ARRIVE_EPS`、跟随后的 `_lock_final_adjust_if_ready(d_tgt)`。
- **重构错误位置**：`nav_system/fsm/states/final_adjust.py` 关键字 `判定终调锁定条件`、函数开头 `if ... force_timeout`，以及跟随后的重复锁定判断。

