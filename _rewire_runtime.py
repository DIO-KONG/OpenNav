#!/usr/bin/env python
"""One-shot rewriter: bind navvlm_open/object/nav_frontier mains to NavigationRuntime."""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REPLACEMENTS = {
    "patrol_goal_nav_target": "runtime.patrol.navigation_goal_target",
    "patrol_goal_nav": "runtime.patrol.navigation_goal",
    "patrol_target": "runtime.patrol.semantic_target",
    "patrol_source": "runtime.patrol.source",
    "_patrol_plan_fail_cnt": "runtime.patrol.plan_fail_count",
    "_patrol_retry_t": "runtime.patrol.retry_at",
    "_patrol_frontier_t": "runtime.patrol.frontier_retry_at",
    "_patrol_blocked_diverted": "runtime.patrol.blocked_diverted",
    "final_goal_nav": "runtime.final.navigation_goal",
    "final_target": "runtime.final.semantic_target",
    "_final_plan_fail_cnt": "runtime.final.plan_fail_count",
    "_final_plan_retry_t": "runtime.final.retry_at",
    "_final_adj_plan_fail_cnt": "runtime.final.adjust_plan_fail_count",
    "final_adjust_t": "runtime.final.adjust_started_at",
    "final_adjust_frames": "runtime.final.adjust_frames",
    "_final_from_reloc": "runtime.final.from_reloc",
    "_final_bbox_anchor": "runtime.final.bbox_anchor",
    "_last_goal_shift_t": "runtime.final.last_goal_shift_t",
    "goal_source": "runtime.goal_source",
    "_last_goal_src": "runtime.debug.last_goal_src",
    "_last_patrol_src": "runtime.debug.last_patrol_src",
    "_last_goal_tgt_key": "runtime.debug.last_goal_tgt_key",
    "_nav_invalid_replan_pending": "runtime.debug.invalid_replan_pending",
    "_nav_invalid_replan_reason": "runtime.debug.invalid_replan_reason",
    "obstacle_snapshot": "runtime.obstacle.snapshot",
    "_obstacle_points_id": "runtime.obstacle.points_id",
    "_obstacle_revision": "runtime.obstacle.revision",
    "scan_accumulated": "runtime.scan.accumulated",
    "scan_prev_yaw": "runtime.scan.prev_yaw",
    "_boot_scan_done": "runtime.scan.boot_done",
    "_nav_step_t": "runtime.debug.nav_step_t",
    "_nav_grab_ms": "runtime.debug.grab_ms",
    "_nav_draw_ms": "runtime.debug.draw_ms",
    "_nav_work_ms": "runtime.debug.work_ms",
    "last_cmd": "runtime.last_cmd",
    "vlm_latest": "runtime.vlm.latest",
    "prev_halt": "runtime.debug.prev_halt",
    "lookahead_target": "runtime.lookahead_target",
    "d_tgt": "runtime.d_tgt",
    "d_robot": "runtime.d_robot",
    "nav_y": "runtime.nav_y",
    "prev_t": "runtime.debug.prev_t",
    "fps": "runtime.debug.fps",
    "_vlog_t": "runtime.debug.vlog_t",
    "_reloc_phase": "runtime.reloc.phase",
    "_reloc_t": "runtime.reloc.t",
    "_reloc_giveup": "runtime.reloc.give_up",
    "_reloc_prev_state": "runtime.reloc.previous_state",
    "_reloc_start_t": "runtime.reloc.started_at",
    "_reloc_probe_t": "runtime.reloc.probe_t",
    "RELOC_MAX_TIME": "runtime.reloc.max_time",
    "_anchor_kf": "runtime.pose.anchor_kf",
    "_anchor_cur": "runtime.pose.anchor_cur",
    "_kf_prev_pos": "runtime.pose.kf_prev_pos",
    "_nav_smooth": "runtime.pose.nav_smooth",
    "_odom_anchor": "runtime.pose.odom_anchor",
    "_slam_anchor": "runtime.pose.slam_anchor",
    "_plan_slam_anchor": "runtime.pose.plan_slam_anchor",
    "_plan_odom_anchor": "runtime.pose.plan_odom_anchor",
    "_plan_pose": "runtime.pose.plan_pose",
    "_nav_source": "runtime.pose.source",
    "_use_odom": "runtime.pose.use_odom",
    "_escaping": "runtime.pose.escaping",
    "_prev_escaping": "runtime.pose.prev_escaping",
    "_post_escape": "runtime.pose.post_escape",
    "_post_escape_t": "runtime.pose.post_escape_t",
    "_yaw_sign": "runtime.pose.yaw_sign",
    "NAV_SMOOTH_POS": "runtime.pose.smooth_pos",
    "NAV_SMOOTH_YAW": "runtime.pose.smooth_yaw",
    "_escape_dir": "runtime.escape.direction",
    "_escape_via_path0": "runtime.escape.via_path0",
    "_escape_path0_drive_mode": "runtime.escape.path0_drive_mode",
    "_escape_path0_anchor": "runtime.escape.path0_anchor",
    "_escape_path0_path": "runtime.escape.path0_path",
    "_escape_path0_line": "runtime.escape.path0_line",
    "_escape_path0_aligned": "runtime.escape.path0_aligned",
    "_escape_path1_yaw": "runtime.escape.path1_yaw",
    "_escape_phase": "runtime.escape.phase",
    "_escape_resume_state": "runtime.escape.resume_state",
    "_escape_target": "runtime.escape.navigation_goal",
    "_escape_semantic_target": "runtime.escape.semantic_target",
    "_escape_replan_from_semantic": "runtime.escape.replan_from_semantic",
    "_escape_plan_fail_cnt": "runtime.escape.fail_count",
    "_preturn_active": "runtime.preturn.active",
    "_preturn_start_pose": "runtime.preturn.start_pose",
    "_preturn_slam_anchor": "runtime.preturn.slam_anchor",
    "_preturn_odom_anchor": "runtime.preturn.odom_anchor",
    "_preturn_active_time": "runtime.preturn.active_time",
    "_preturn_last_tick": "runtime.preturn.last_tick",
    "_preturn_resume_plan_state": "runtime.preturn.resume_plan_state",
    "_auto_vlm_asked": "runtime.vlm.auto_asked",
    "_auto_vlm_ask_t": "runtime.vlm.ask_t",
    "_auto_det_t": "runtime.vlm.det_t",
    "_presence_confirmed": "runtime.vlm.presence_confirmed",
    "_vlm_epoch": "runtime.vlm.epoch",
    "_vlm_request_id": "runtime.vlm.request_id",
    "_vlm_reask_pending": "runtime.vlm.reask_pending",
    "_frontier_query_t": "runtime.vlm.frontier_query_t",
    "_presence_samples": "runtime.vlm.presence_samples",
    "_PRESENCE_WINDOW": "runtime.vlm.PRESENCE_WINDOW",
    "_PRESENCE_SUSTAIN": "runtime.vlm.PRESENCE_SUSTAIN",
    "_PRESENCE_RATIO": "runtime.vlm.PRESENCE_RATIO",
    "path_idx": "runtime.path_idx",
    "path": "runtime.path",
    "state": "runtime.state",
    "step": "runtime.step",
}

MOVED_FUNS = (
    "_reset_intrusion_escape",
    "_set_patrol_goal",
    "_mark_invalid_nav_replan",
    "_start_preturn_odom_hold",
)

CALL_RENAMES = {
    "_reset_intrusion_escape": "runtime.reset_intrusion_escape",
    "_set_patrol_goal": "runtime.set_patrol_goal",
    "_mark_invalid_nav_replan": "runtime.mark_invalid_nav_replan",
    "_start_preturn_odom_hold": "runtime.start_preturn_odom_hold",
}

INIT_DROP = set(REPLACEMENTS) | {
    "runtime",
}


def extract_indented_block(src: str, def_name: str) -> tuple[int, int]:
    """Return [start, end) char offsets of `    def name` through next same-indent def or print."""
    m = re.search(rf"\n    def {re.escape(def_name)}\(", src)
    if not m:
        raise SystemExit(f"missing def {def_name}")
    start = m.start() + 1
    rest = src[m.end():]
    n = re.search(r"\n    def |\n    print\(", rest)
    if not n:
        raise SystemExit(f"cannot find end of {def_name}")
    end = m.end() + n.start()
    return start, end


def drop_function(src: str, def_name: str) -> str:
    start, end = extract_indented_block(src, def_name)
    return src[:start] + src[end:]


def replace_begin_escape(src: str) -> str:
    start, end = extract_indented_block(src, "_begin_intrusion_escape")
    new = '''    def _begin_intrusion_escape(resume_state, target, clearance):
        """冻结 path_old 的实际目标，废弃旧路径并进入同目标 ESCAPE 重规划。"""
        if not runtime.begin_intrusion_escape(resume_state, target):
            return False
        motion_thread.stop()
        goal = runtime.escape.navigation_goal
        print(f"[INTRUSION] phase=start target=({goal[0]:.3f},"
              f"{goal[1]:.3f}) clearance={float(clearance):.3f}m "
              f"resume={resume_state.name} action=replan_same_target")
        return True

'''
    return src[:start] + new + src[end:]


def slim_invalidate(src: str) -> str:
    start, end = extract_indented_block(src, "_invalidate_vlm")
    new = '''    def _invalidate_vlm(reason):
        """使状态变化前已提交、但无法取消的 HTTP 结果失效。"""
        runtime.invalidate_vlm()
        _dbg(f"[VLM-ASYNC] invalidate epoch={runtime.vlm.epoch} reason={reason}")

'''
    return src[:start] + new + src[end:]


def strip_nonlocals(src: str) -> str:
    return re.sub(r"\n        nonlocal[^\n]*", "", src)


def replace_init_slice(src: str, boot_scan_done: bool) -> str:
    m = re.search(
        r"\n    state = NavState\.WAITING\n    runtime = NavigationRuntime\(state\)\n",
        src)
    if not m:
        raise SystemExit("init marker not found")
    n = re.search(r"\n    def _invalidate_vlm\(", src)
    if not n:
        raise SystemExit("invalidate marker not found")
    boot = "True" if boot_scan_done else "False"
    extra_open = ""
    if "nav_open: FINAL_ADJUST" in src[:800] or "_present_reset" in src:
        extra_open = '''
    def _present_register(is_present, now):
        runtime.vlm.register_presence(is_present, now)

    def _present_ratio(now):
        return runtime.vlm.presence_ratio(now)

    def _present_locked(now):
        return runtime.vlm.presence_locked(now)

    def _present_reset(reason=""):
        runtime.vlm.reset_presence()
        _rsn = f" ({reason})" if reason else ""
        print(f"\\033[96m[nav_open][PRESENCE] 窗口清空{_rsn} -> 样本丢弃\\033[0m")

'''
    kept = src[m.end():n.start()]
    rate_m = re.search(r"    rate = rospy\.Rate\(10\)\n", kept)
    motion_m = re.search(r"    motion_thread = MotionThread\(bot, hz=MOTION_HZ\)\n", kept)
    atexit_m = re.search(r"    # ===== 兜底清理 \(atexit\) =====\n", kept)
    yaw_m = re.search(r"    yaw_smoother = YawSmoother\(alpha=FOLLOW_YAW_EMA\)\n", kept)
    last_m = re.search(r"    _last_action_key = None\n", kept)
    if not all((rate_m, motion_m, atexit_m, yaw_m, last_m)):
        raise SystemExit("init keep-markers missing")
    kept_block = (
        rate_m.group(0)
        + "\n"
        + motion_m.group(0)
        + "\n"
        + kept[atexit_m.start():last_m.end()]
        + "\n"
    )
    new_head = (
        f"\n    runtime = NavigationRuntime("
        f"NavState.WAITING, boot_scan_done={boot})\n"
        "    runtime.obstacle.snapshot = build_obstacle_snapshot("
        "None, fixed_y=0.0, revision=0)\n"
    )
    return src[:m.start()] + new_head + kept_block + extra_open + src[n.start():]


def tokenize_replace(src: str) -> str:
    out = []
    tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    prev_name_or_op = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string in REPLACEMENTS:
            skip = prev_name_or_op == "."
            if i > 0 and tokens[i - 1].string == "def":
                skip = True
            if not skip:
                repl = REPLACEMENTS[tok.string]
                out.append((tokenize.NAME, repl, tok.start, tok.end, tok.line))
                prev_name_or_op = repl.split(".")[-1]
                i += 1
                continue
        if tok.type == tokenize.NAME and tok.string in CALL_RENAMES:
            skip = i > 0 and tokens[i - 1].string == "def"
            if not skip:
                repl = CALL_RENAMES[tok.string]
                out.append((tokenize.NAME, repl, tok.start, tok.end, tok.line))
                prev_name_or_op = repl.split(".")[-1]
                i += 1
                continue
        out.append(tok)
        if tok.type in (tokenize.NAME, tokenize.OP):
            prev_name_or_op = tok.string
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                              tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER):
            prev_name_or_op = None
        i += 1
    return tokenize.untokenize(out)


def strip_open_module_presence(src: str) -> str:
    src = re.sub(
        r"\n_PRESENCE_WINDOW = 10[^\n]*\n"
        r"_PRESENCE_SUSTAIN = 5\.0[^\n]*\n"
        r"_PRESENCE_RATIO = 0\.8[^\n]*\n"
        r"_presence_samples = \[][^\n]*\n"
        r"\n"
        r"def _present_register\(is_present, now\):\n"
        r"    \"\"\"[^\"]*\"\"\"\n"
        r"    _presence_samples\.append\(\(now, bool\(is_present\)\)\)\n"
        r"    _over = len\(_presence_samples\) - \(_PRESENCE_WINDOW \+ 1\)\n"
        r"    if _over > 0:\n"
        r"        del _presence_samples\[:_over\]\n"
        r"\n"
        r"def _present_ratio\(now\):\n"
        r"    \"\"\"[^\"]*\"\"\"\n"
        r"    n = len\(_presence_samples\)\n"
        r"    if n < _PRESENCE_WINDOW:\n"
        r"        return None\n"
        r"    _win = _presence_samples\[-_PRESENCE_WINDOW:\]\n"
        r"    _yes = sum\(1 for _, p in _win if p\)\n"
        r"    return _yes / _PRESENCE_WINDOW\n"
        r"\n"
        r"def _present_locked\(now\):\n"
        r"    \"\"\"[^\"]*\"\"\"\n"
        r"    _ratio = _present_ratio\(now\)\n"
        r"    if _ratio is None or _ratio < _PRESENCE_RATIO:\n"
        r"        return False\n"
        r"    _win = _presence_samples\[-_PRESENCE_WINDOW:\]\n"
        r"    return \(now - _win\[0\]\[0\]\) >= _PRESENCE_SUSTAIN\n"
        r"\n"
        r"def _present_reset\(reason=\"\"\):\n"
        r"    \"\"\"[^\"]*\"\"\"\n"
        r"    _presence_samples\.clear\(\)\n"
        r"    _rsn = f\" \(\{reason\}\)\" if reason else \"\"\n"
        r"    print\(f\"\\033\[96m\[nav_open\]\[PRESENCE\] 窗口清空\{_rsn\} -> 样本丢弃\\033\[0m\"\)\n"
        r"\n",
        "\n",
        src,
        count=1,
    )
    return src


def rewire(path: Path, boot_scan_done: bool, is_open: bool) -> None:
    src = path.read_text(encoding="utf-8")
    if is_open:
        src = strip_open_module_presence(src)
    src = replace_init_slice(src, boot_scan_done)
    src = slim_invalidate(src)
    src = replace_begin_escape(src)
    for name in MOVED_FUNS:
        src = drop_function(src, name)
    src = strip_nonlocals(src)
    src = tokenize_replace(src)
    # untokenize can expand spacing; keep file readable enough.
    path.write_text(src, encoding="utf-8")
    print(f"rewired {path.name} lines={src.count(chr(10))+1}")


def main():
    rewire(ROOT / "navvlm_object.py", boot_scan_done=True, is_open=False)
    rewire(ROOT / "navvlm_open.py", boot_scan_done=False, is_open=True)
    rewire(ROOT / "nav_frontier.py", boot_scan_done=False, is_open=False)


if __name__ == "__main__":
    main()
