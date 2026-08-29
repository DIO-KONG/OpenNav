# 重构 vs legacy 逻辑不一致清单

对照范围：`legacy/` 三份实机状态机（`navvlm_open.py` / `navvlm_object.py` / `nav_frontier.py`）以及它们依赖的共享模块，对比 `nav_system/` + `nav_memory/`。

只记录**运行时行为会变**的差异。注释、打印文案、纯拆函数且控制流等价的改写不列。

共享模块逐行对比结果：

- 完全一致：`nav_control.py`、`nav_path.py`、`nav_rrt.py`、`nav_page.py`、`RGBRosConnector.py`
- 仅注释：`nav_helpers.py`、`nav_runtime.py`
- 有行为差：`nav_constants.py`、`nav_vlm.py`、`nav_debug.py`、`mast3r_slam_wrapper.py`
- 状态机本体：三份 god file ↔ `nav_system/main.py` + `fsm/` + `policies/` + `nav_services.py`

---

## P0 — 会改导航决策 / 可能卡死或崩

### 1. 默认 VLM 检测目标词变了

- legacy `nav_constants.py`：`VLM_DETECT_TARGET_DEFAULT = "rest area"`
- 重构 `nav_system/nav_constants.py`：`"white plastic stool"`

所有 presence / detect / direction prompt 都从该常量展开。实机看到的目标完全不是原来那套。

### 2. object 模式开局扫描：legacy 卡在 WAITING，重构会去 SCANNING

`legacy/navvlm_object.py` 启动时 `_boot_scan_done = True`。WAITING 只有 `if not _boot_scan_done and TRACKING` 才会切 SCANNING / INQUIRY，因此 **object 脚本永远停在 WAITING**（open / frontier 则是 `False`，会进扫描）。

重构 `ObjectNavPolicy.default_boot_scan_done = False`，WAITING 在 TRACKING 后走 SCANNING。`main.py` 注释还写着 “object 模式保持 legacy 的开局跳过扫描”，与 policy 代码相反。

open / frontier 的 `_boot_scan_done = False` 与重构一致。

### 3. `FinalFollowState.PLAN_FAIL_MAX` 未定义

`nav_system/fsm/states/final_follow.py` 在 `path is None` 且连续规划失败时执行：

```python
if ctx.final_adj_plan_fail_cnt >= self.PLAN_FAIL_MAX:
```

该类没有 `PLAN_FAIL_MAX`（`FinalAdjustState` / `PatrolPlanState` / `EscapeState` 才有 `= 10`）。legacy 用模块级 `PLAN_FAIL_MAX = 10`。重构走到这条路径会 `AttributeError` 炸掉主循环。

### 4. ESCAPE 同语义重规划成功后，不写回 `final_goal_nav` / `patrol_goal_nav`

legacy（三份 god file 相同）在 ESCAPE `plan` 阶段 `plan_goal_resolution` 成功时：

- 更新 `_escape_target`
- 若 resume 是 FINAL_* → `final_goal_nav = _selected.goal`
- 否则 → `patrol_goal_nav = _selected.goal`

重构 `EscapeState` 只写 `ctx.escape_target`，不更新 `ctx.final_goal_nav` / `ctx.patrol_goal_nav`。脱困结束后 FOLLOW 仍用**旧的、已判定不安全的站位**做到达判定和后续重规划。

### 5. 切入 ESCAPE 时不再 `invalidate_vlm`

legacy `_begin_intrusion_escape` 第一件事是 `_invalidate_vlm("intrusion_escape")`，丢掉入侵前已提交的 auto_detect / direction。

重构 FOLLOW/ADJUST → Escape 只设 `resume_state_cls` / `escape_target` 并清 path，**不 bump `vlm_epoch`**。脱困期间主循环虽 suppress auto_detect，但 epoch 未失效的结果会在离开 ESCAPE 后被当成有效检测，可能把目标拽走。

### 6. PATROL_FOLLOW 到达语义被拆开，且到达后清不清目标不一致

legacy（到达判定在脱困之前）：

```text
if d_tgt <= eps or d_end <= eps or path_idx >= len(path):
    → VLM_INQUIRY
    不清除 patrol_target
```

路径走完但还没到站，也会放弃本轮去问询；旧巡逻目标仍留着（SCANNING 结束 / reloc 恢复仍可能按旧目标续走）。

重构：

- `d_tgt/d_end` 到达 → INQUIRY，并且 `clear_patrol(clear_target=True)`
- `path_idx >= len(path)` 且未到站 → **PATROL_PLAN 重规划同一目标**，不问询

两种模式（“走完路但没到” 和 “到了之后目标是否还在”）都变了。

### 7. ESCAPE 恢复 FOLLOW 会重新跑 `on_enter`（再武装 preturn + 重置平滑器）

legacy 的 ESCAPE 是同一 `state` 枚举上的旁路，恢复时 `state = _escape_resume_state`，**不重新执行 FOLLOW 的进入逻辑**；`path_idx` 已是 1，直接续跟。

重构 `change_state(PatrolFollowState/FinalFollowState)` 必跑 `on_enter`：重置 yaw/out smoother，并按当前 `path` 再判一次近墙大角 → 可能**刚脱困完立刻再倒车 preturn**。

### 8. FINAL_ADJUST 路径重建时不再武装 preturn

legacy `FINAL_FOLLOW` 与 `FINAL_ADJUST` 共用同一跟随块；`path is None` 规划成功后都 `_arm_preturn(...)`。

重构：

- `FinalFollowState` 有 `_maybe_arm_preturn`
- `FinalAdjustState` 规划成功只 `path_idx` 处理，**不 preturn**

终调阶段贴墙大角转向时，legacy 会先退再转，重构直接跟。

### 9. PRETURN 超时不转入 odom hold（snap-back）

legacy `_run_preturn` 超时：

```text
_preturn_active = False
if back_progress > 0.01:
    _start_preturn_odom_hold()   # 把 preturn 冻结锚点交给 post_escape 窗口
path_idx = 0
```

成功退够距离时两边都会拷锚点并 `post_escape=True`。超时且已经往后蹭了几厘米时，legacy 继续用 odom 直到 SLAM 追上，避免 stale SLAM 把车拽回墙。

重构 `PatrolFollowState` / `FinalFollowState` 超时只 `preturn_active=False; path_idx=0`，**不拷 `slam_anchor`/`odom_anchor`，不设 `post_escape`**。下一帧 `use_odom` 立刻变 False，位姿弹回 SLAM。

---

## P1 — 时序 / 门控 / 恢复机

### 10. SCANNING 急停时仍累计转角 vs 急停直接 return

legacy：halted 只停 `cmd_vel`，**yaw 仍累计**；松开急停后从已转过的角度接着算 360°。

重构 `ScanningState`：`halted` 立刻 `return`，不累计。急停期间底盘若因惯性/人工动了，重构当没转；legacy 会把这段算进扫描进度。

### 11. RELOC 是旁路还是正式状态（超时后的主循环节拍）

legacy：RELOC **不改 `state`**，主循环 `continue`，并且 `time.sleep(0.066)`（约 15Hz），期间仍刷 overlay。超时只设 `_reloc_giveup`、清 `_reloc_phase`，**原来的 WAITING/FOLLOW 状态从未离开**，下一帧直接按原状态跑。

重构：RELOC 是独立状态，走 10Hz `rospy.Rate`。超时 `change_state(reloc_prev)`，等于“离开再回来”，会触发原状态 `on_enter`（扫描进度清零、FOLLOW 再 arm preturn、ADJUST 重设计时/帧数）。

### 12. RELOC spin 的 presence 节流时钟不同

legacy：`_auto_det_t` 在 **poll 到** `auto_detect/reloc_presence/reloc_detect` 的 `finished_at` 时更新，提交条件是 `now - _auto_det_t >= OBJ_DET_PERIOD`（完成→再等 2s）。

重构 `RelocState`：提交时 `self.last_det_t = now`（提交→再等 2s）。请求一发出就开始冷却，spin 期间 presence 更稀。

### 13. object/open 的 INQUIRY 在 pose/img 丢失时提前 return

legacy：即使 `cur_pose is None` 也会先消费 direction 结果（失败则 `_auto_vlm_asked=False`）；只有 3D 投影那一段 `pass`。

重构 `ObjectNavPolicy.handle_inquiry_step` 开头：

```python
if cur_pose is None or img is None:
    return StateDecision(...)
```

该帧不 poll direction。结果虽可能躺在 `pending_vlm_results`，但若随后主循环又 poll 到别的 kind，busy/时序会偏。丢图一帧就会推迟问询。

### 14. INQUIRY 到达判定用的点不同

legacy 在解析出方向 3D 后**先写入** `patrol_target`，再用

```text
_nav = patrol_goal_nav if patrol_goal_nav else patrol_target
d_tgt = dist(cur, _nav)
```

若上一轮巡逻留下了 `patrol_goal_nav`，距离是到**旧站位**，walked 却用新 `(tx,tz)`。

重构直接 `d_tgt = dist(cur, (tx,tz))`，不经过旧 nav。同一帧可能走到“重问 VLM / 去 frontier / 去 PATROL_PLAN”的不同分支。

### 15. `_set_patrol_goal` 是否清 `vlm_reask_pending`

legacy `_set_patrol_goal` 只把 `_auto_vlm_asked=False`，**不清** `_vlm_reask_pending`。

重构 `NavContext.set_patrol_goal` 两者都清。VLM=F 改走 frontier 后，legacy 仍可能把下一轮方向点当成“重问后的点”直接转 frontier。

### 16. FINAL_ADJUST 锁定后切 FOLLOW 的速度

legacy `_lock_final_adjust_if_ready` 只改 `state`，本帧若已在跟随则 `cmd_vel` 仍按 FOLLOW 块发出（函数在跟随之后还会再调一次）。

重构锁定成功返回 `StateDecision(next_state=FinalFollow, linear=..., angular=...)`；引擎先填 debug 再 `change_state`。`FinalAdjust.on_exit` 故意不停电机，`FinalFollow.on_enter` **也不停**，但 `FinalFollow.on_enter` 会 `reset_smoothers()`。锁定当帧平滑器被清掉，下一帧角速度从 0 再爬，legacy 平滑器连续。

### 17. 急停上升沿是否 `bot.stop()`

legacy：halted 上升沿同时 `bot.stop()` + `motion_thread.stop()`。

重构：只靠 `decision.reset_motion` → `motion_thread.stop()`，**没有**额外的 Tracer `bot.stop()`。若 motion 线程本帧还没把 0 速发出去，底盘会多跑最多一个控制周期。

### 18. SIGINT / 退出路径

legacy：SIGINT 设 stop event、停 motion/VLM/bot、`rospy.signal_shutdown`，**0.5s 后再 `os._exit(0)`**；atexit 里 `slam.shutdown()`；主循环正常结束走 `slam.stop()`。

重构 `nav_services.stop_all`：SIGINT **立刻** `stop_all()` + `os._exit(0)`；SLAM 一律 `shutdown()`。Ctrl+C 时不再给 SLAM/HTTP 0.5s 收尾窗口，地图落盘能否跑完取决于 `shutdown()` 是否阻塞（见下条）。

---

## P2 — 感知 / 工程路径 / 可视化

### 19. MASt3R-SLAM 目录与权重路径

legacy wrapper：

```text
_SLAM_DIR = <legacy>/masterslam
MAST3R_WEIGHT / RETRIEVER_WEIGHT = /home/agilex/.../masterslam/checkpoints/...
DEFAULT_SAVE_DIR = <legacy>/slam_output
```

重构：

```text
_SLAM_DIR = <repo>/MASt3R-SLAM
权重 = _SLAM_DIR/checkpoints/...
无 DEFAULT_SAVE_DIR；stop()/shutdown() 不再带 save_dir
```

实机若仍用 `masterslam` 目录或硬编码权重，重构 import / 加载会失败或加载另一份模型。

### 20. bbox 相机内参路径

legacy 三份脚本：

```text
<project_root>/masterslam/config/intrinsics.yaml
```

重构 `nav_services.py`：

```text
<repo>/MASt3R-SLAM/config/intrinsics.yaml
```

路径不存在则 `_bbox_calibration is None`。这会直接关掉下一条的锁定框投影。

### 21. FINAL_FOLLOW overlay 不再投影锁定 bbox

legacy 在 `FINAL_FOLLOW` 且标定加载成功时，用 `project_locked_target_bbox` 把锁定目标投回当前画面。

重构 `main.py` overlay 只画 `ctx.vlm_latest`，**从不调用** `project_locked_target_bbox`，也从不读 `bbox_calibration`。终调锁定后画面框会停在最后一次检测，或直接消失。不影响控制，影响实机盯画面判断是否跟丢。

### 22. RELOC 期间 overlay / 地图内容变少

legacy RELOC 旁路仍画 raw pose、reloc 阶段名、elapsed；`draw_map_view` 带 path/lookahead。

重构 RELOC 走普通 overlay 两行（state + nav pose）。debug 观感变了，控制逻辑见 #11。

### 23. HUD 信息密度

legacy 每帧 overlay 有 SLAM mode / raw+nav pose / path blocked / patrol&final 坐标 / dist / SCAN 进度。

重构只有 `[STATE] cmd` 和一行 `nav[...]`。同样只影响人看，不影响控制。

### 24. TracerRobot import 路径

legacy：`sys.path` 插入 `<root>/tracer_ros/tracer_http_interface/scripts` 再 `from tracer_http_interface.scripts.rw_api import TracerRobot`。

重构：假定仓库根下就有 `tracer_http_interface.scripts.rw_api`。当前仓库布局是根目录 `tracer_http_interface/`，重构能 import；若实机仍是 `tracer_ros/...` 那套，重构起不来。

### 25. `nav_debug.NavDebugger` 的默认 episode 根目录

legacy：`dirname(__file__)` → 模块所在目录。

重构：再包一层 `dirname`，默认写到仓库根 `episodes/` 而不是 `nav_system/episodes/`。

两边主循环都**没有**构造 `NavDebugger()`，当前主路径不触发；一旦有人按模块默认用法打开 debugger，落盘位置会变。

### 26. SLAM wrapper 去掉 reloc_log / `_probe` / `evaluate` 落盘

legacy `_relocalization(..., reloc_log=)` 成功时 append 时间戳与匹配关键帧；`stop(save_dir=)` 会写地图 / memory ply（`append_memory_to_ply`）。

重构去掉 `reloc_log`、`DEBUG_PROBES`、`mast3r_slam.evaluate` 导入。重定位成功仍 `solve_GN_*`，**跟踪几何应相同**；少的是诊断日志和 stop 时自动存图。若实机流程依赖 `slam.stop()` 落盘，重构 `shutdown()` 行为需再对一下（见 wrapper `stop`/`shutdown` 实现：重构 `stop` 约 990 行、`shutdown` 约 1029 行，不再接收 `save_dir`）。

### 27. `AsyncVlmWorker.poll(kind=)`

重构为多消费者加了 kind 过滤：不匹配的结果放回队列并保持 busy。

这是 FSM 拆分所必需的，主循环先统一 poll 再 stash，**设计上应等价**于 legacy 单消费者。不单独算功能回归，但若有状态漏 poll，worker 会一直 busy（legacy 也会，因为队列长度为 1）。

---

## 已核对、判定为等价（不列入缺陷）

- 控制 / RRT / 路径检验 / Flask 页：文件级 identical。
- 位姿融合（关键帧锚点、odom 保持窗、plan_pose yaw 0.17rad）：已迁到 `NavContext.update_pose_estimation`，公式与阈值一致。
- 纯追踪、NAV_CLEARANCE、入侵判定 `clearance=ROBOT_RADIUS`：一致。
- open presence 窗口 10 / 5s / 0.8：`OpenNavPolicy` 一致；object/frontier 单次 yes 进 ADJUST：一致。
- RELOC 阶段机 stop → back → spin → detect_hold，以及 tracking 恢复时 FINAL→FOLLOW、PATROL→PLAN、SCANNING 按已有目标续：骨架一致（差异见 #11/#12）。
- ESCAPE halt 期间 `escaping_active`：legacy 每帧先清再在 path0/align/fallback 里重标；重构 `on_enter` 置 True、`on_exit` 清，halted 时保持 True。odom 保持窗两边都不丢。
- FINAL_ADJUST EMA：主循环 `handle_global_auto_detection` 的滤波公式与拒收阈值与 legacy 一致；差的只是终调中不刷新 overlay 用的 `final_bbox_anchor`（见 #21）。
- FINAL_PLAN 连续失败后的 2s cooldown：重构在 `>= PLAN_FAIL_MAX` 之前就写了 `final_plan_retry_t`，与 legacy 等价。
- `BOOT_SCAN_ENABLED`、`FINAL_ADJUST_*`、跟随 / 脱困数值常量：除检测目标词外，重构只是重排分组，字面值相同。
- `_patrol_blocked_diverted` 在三份 legacy 里只被置位/清零，**从未被读取**；重构删掉不影响行为。

---

## 建议修复顺序

1. 把 `VLM_DETECT_TARGET_DEFAULT` 改回 `"rest area"`（或确认实机现在就要换凳子）。
2. 给 `FinalFollowState` 补 `PLAN_FAIL_MAX = 10`。
3. ESCAPE 成功重选站位时写回 `final_goal_nav`/`patrol_goal_nav`；进入 ESCAPE 时 `invalidate_vlm`。
4. 恢复 FOLLOW 不要无条件 `on_enter` 再 arm preturn（或 resume 时跳过 arm）。
5. PATROL 到达：对齐 “path 耗尽但未到站” 以及是否 `clear_patrol`。
6. object 开局：先决定 WAITING 该卡住、该跳过扫描、还是该扫一圈，三边（object 脚本 / policy / main 注释）必须同一语义。
7. SCANNING halted 仍累计 yaw；ADJUST 规划成功后 arm preturn；preturn timeout 在 `back_progress>0.01` 时进入 odom hold。
8. SLAM / 标定路径若实机仍是 `masterslam/`，wrapper 与 `nav_services` 不要改指向 `MASt3R-SLAM/`。
