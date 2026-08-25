#!/usr/bin/env python
"""导航调试与 Episode 结构化记录模块。

主线程只执行非阻塞的配置判断和队列入队；JSONL 写盘、JPEG 拼接与磁盘写入由
后台线程完成。调试线程只写文件和缓存，不修改导航状态、路径或速度指令。

时序约定
--------
- step : 主循环主键，由 nav_auto 每轮开头 set_step(step) 绑定；同轮内所有
         record_* / emit / console / frame 共享同一 step 和 step_ts。
- seq  : 全局单调序号，每次 emit 自增，用于同 step 内按数据流排序。
- 统一外壳: {"ts", "step", "seq", "kind", "data"}

VLM data.stage 约定（兼容旧 status）
------------------------------------
presence / detect / detect_error / no_detection / bbox_3d_fail /
direction / h_ask / locked_skip
旧 status 映射: inquiry→presence, detected→detect, error→detect_error
"""

import copy
import datetime
import json
import os
import queue
import re
import sys
import threading
import time

import cv2
import numpy as np

from nav_constants import (
    DEBUG_DROP_OLD_FRAMES,
    DEBUG_ENABLED,
    DEBUG_EPISODE_FORMAT,
    DEBUG_FEATURES,
    DEBUG_FRAME_PERIOD,
    DEBUG_FRAME_QUEUE_SIZE,
    DEBUG_FRAME_SAVE,
    DEBUG_LOG_TO_FILE,
    DEBUG_PERIODS,
    DEBUG_QUEUE_SIZE,
    ROBOT_RADIUS,
)
from nav_helpers import query_frontier_for_viz

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_FRAME_LAYOUT = "streams_v1"
_FRAME_CONTAINER_WIDTH = 1200
_FRAME_STREAM_GAP = 12
_FRAME_IMAGE_BORDER = 2
_FRAME_TITLE_MARGIN_TOP = 19
_FRAME_TITLE_LINE_HEIGHT = 22
_FRAME_TITLE_MARGIN_BOTTOM = 19
_FRAME_BODY_BG_BGR = (17, 17, 17)
_FRAME_BORDER_BGR = (51, 51, 51)
_FRAME_TITLE_BGR = (0, 255, 0)


class _Tee:
    """将标准输出同时写入控制台和 Episode 日志；按行加 ts/step/seq 戳。

    参数:
        console: 原始 stdout 流
        logf:    文本日志文件句柄 (nav_auto.log)，可为 None
        debugger: NavDebugger 实例，用于 emit_console；可为 None
    """

    def __init__(self, console, logf, debugger=None):
        self.console = console
        self.logf = logf
        self.debugger = debugger
        self._buf = ""  # 行缓冲：跨 write 拼完整行

    def write(self, value):
        """写控制台原文；完整行加戳写入 log 并 emit console JSONL。"""
        if not isinstance(value, str):
            value = str(value)
        try:
            self.console.write(value)
        except Exception:
            pass
        if not value:
            return
        # 去掉 ANSI 再缓冲，避免颜色码污染 log / JSONL
        clean = _ANSI_RE.sub("", value)
        self._buf += clean
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line)

    def _emit_line(self, line):
        """单行落盘: 文本戳 + 结构化 console 记录。

        参数:
            line: 不含尾部换行的一行文本
        返回:
            None
        """
        dbg = self.debugger
        step = getattr(dbg, "_step", None) if dbg is not None else None
        # 先取 seq 再写，保证 log 前缀与 JSONL 同一序号
        seq = dbg._next_seq() if dbg is not None else None
        ts = (dbg.timestamp_for_step(step)
              if dbg is not None else time.time())
        ts_iso = datetime.datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")
        step_s = "null" if step is None else str(int(step))
        seq_s = "null" if seq is None else str(int(seq))
        stamped = f"[{ts_iso} step={step_s} seq={seq_s}] {line}\n"
        if self.logf is not None:
            try:
                self.logf.write(stamped)
            except Exception:
                pass
        if dbg is not None:
            dbg.emit_console(line, ts=ts, step=step, seq=seq, stream="stdout")

    def flush(self):
        """刷新缓冲：残留半行也写出，再 flush 底层流。"""
        if self._buf:
            # 半行也落盘，避免进程退出丢最后一行
            residual = self._buf
            self._buf = ""
            self._emit_line(residual)
        try:
            self.console.flush()
        except Exception:
            pass
        if self.logf is not None:
            try:
                self.logf.flush()
            except Exception:
                pass


class AsyncFrameSaver:
    """异步拼接和保存 RGB/地图快照。

    参数:
        maxsize: 帧队列上限
    属性:
        dropped: 因队列满丢弃的帧数
    """

    def __init__(self, maxsize=DEBUG_FRAME_QUEUE_SIZE):
        self._queue = queue.Queue(maxsize=maxsize)
        self._running = False
        self._thread = None
        self.dropped = 0

    def start(self):
        """启动后台写帧线程。返回 None。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="nav-debug-frame-writer", daemon=True)
        self._thread.start()

    def stop(self):
        """停止后台线程并 join。返回 None。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def push(self, rgb_frame, map_frame, step, save_dir):
        """非阻塞入队一帧 (rgb, map, step, dir)。

        参数:
            rgb_frame: BGR ndarray 或 None（仅一侧时另一侧可 None，由调用方保证至少一侧）
            map_frame: BGR ndarray 或 None
            step: 主循环 step
            save_dir: frames 目录
        返回:
            None
        """
        if not self._running or save_dir is None:
            return
        # overlay_frame/map_frame 在主循环中每轮新建且入队后不再修改，直接传引用，
        # 避免复制大图像再次占用导航线程 CPU 与内存带宽。
        item = (rgb_frame, map_frame, int(step), save_dir)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if not DEBUG_DROP_OLD_FRAMES:
                return
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except queue.Empty:
                pass
            except queue.Full:
                pass

    @staticmethod
    def _prepare_pair(rgb_frame, map_frame):
        """补齐缺失侧，返回可参与 streams 排版的 RGB/地图图像。"""
        if rgb_frame is None and map_frame is None:
            rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            map_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        elif rgb_frame is None:
            h, w = map_frame.shape[:2]
            rgb_frame = np.zeros((h, max(w, 1), 3), dtype=np.uint8)
        elif map_frame is None:
            map_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        return rgb_frame, map_frame

    @staticmethod
    def _resize_to_width(image, width):
        source_h, source_w = image.shape[:2]
        height = max(1, round(source_h * width / max(source_w, 1)))
        interpolation = cv2.INTER_AREA if width < source_w else cv2.INTER_LINEAR
        return cv2.resize(image, (width, height), interpolation=interpolation)

    @classmethod
    def layout_metadata(cls, rgb_frame, map_frame):
        """返回 streams_v1 画布及两侧实际图像区域。"""
        rgb_frame, map_frame = cls._prepare_pair(rgb_frame, map_frame)
        box_width = (_FRAME_CONTAINER_WIDTH - _FRAME_STREAM_GAP) // 2
        image_width = box_width - 2 * _FRAME_IMAGE_BORDER
        image_top = (
            _FRAME_TITLE_MARGIN_TOP
            + _FRAME_TITLE_LINE_HEIGHT
            + _FRAME_TITLE_MARGIN_BOTTOM
        )
        rgb_h, rgb_w = rgb_frame.shape[:2]
        map_h, map_w = map_frame.shape[:2]
        rgb_display_h = max(1, round(rgb_h * image_width / max(rgb_w, 1)))
        map_display_h = max(1, round(map_h * image_width / max(map_w, 1)))
        canvas_h = image_top + max(rgb_display_h, map_display_h) + 2 * _FRAME_IMAGE_BORDER
        right_x = box_width + _FRAME_STREAM_GAP
        content_y = image_top + _FRAME_IMAGE_BORDER
        return {
            "layout": _FRAME_LAYOUT,
            "canvas_shape": [canvas_h, _FRAME_CONTAINER_WIDTH, 3],
            "rgb_region": [
                _FRAME_IMAGE_BORDER,
                content_y,
                image_width,
                rgb_display_h,
            ],
            "map_region": [
                right_x + _FRAME_IMAGE_BORDER,
                content_y,
                image_width,
                map_display_h,
            ],
        }

    @classmethod
    def _combine(cls, rgb_frame, map_frame):
        """按 nav_page.py 的 streams 样式排版 RGB 与顶视地图。

        参数:
            rgb_frame / map_frame: BGR ndarray 或 None
        返回:
            streams_v1 BGR 画布
        """
        rgb_frame, map_frame = cls._prepare_pair(rgb_frame, map_frame)
        layout = cls.layout_metadata(rgb_frame, map_frame)
        canvas_h, canvas_w = layout["canvas_shape"][:2]
        canvas = np.full(
            (canvas_h, canvas_w, 3), _FRAME_BODY_BG_BGR, dtype=np.uint8)

        box_width = (_FRAME_CONTAINER_WIDTH - _FRAME_STREAM_GAP) // 2
        image_width = box_width - 2 * _FRAME_IMAGE_BORDER
        image_top = (
            _FRAME_TITLE_MARGIN_TOP
            + _FRAME_TITLE_LINE_HEIGHT
            + _FRAME_TITLE_MARGIN_BOTTOM
        )
        right_x = box_width + _FRAME_STREAM_GAP
        title_baseline = _FRAME_TITLE_MARGIN_TOP + 16

        cv2.putText(
            canvas, "RGB View", (0, title_baseline), cv2.FONT_HERSHEY_DUPLEX,
            0.62, _FRAME_TITLE_BGR, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, "Top-Down Map", (right_x, title_baseline),
            cv2.FONT_HERSHEY_DUPLEX, 0.62, _FRAME_TITLE_BGR, 1, cv2.LINE_AA)

        rgb_scaled = cls._resize_to_width(rgb_frame, image_width)
        map_scaled = cls._resize_to_width(map_frame, image_width)
        for x, image in ((0, rgb_scaled), (right_x, map_scaled)):
            outer_h = image.shape[0] + 2 * _FRAME_IMAGE_BORDER
            canvas[
                image_top:image_top + outer_h,
                x:x + box_width,
            ] = _FRAME_BORDER_BGR
            y0 = image_top + _FRAME_IMAGE_BORDER
            x0 = x + _FRAME_IMAGE_BORDER
            canvas[
                y0:y0 + image.shape[0],
                x0:x0 + image.shape[1],
            ] = image
        return canvas

    def _worker(self):
        while self._running or not self._queue.empty():
            try:
                rgb_frame, map_frame, step, save_dir = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                combined = self._combine(rgb_frame, map_frame)
                cv2.imwrite(os.path.join(save_dir, f"step_{step:06d}.jpg"), combined)
            except Exception:
                pass
            finally:
                self._queue.task_done()


class NavDebugger:
    """配置驱动的导航调试器和 Episode 结构化记录器。

    时序:
        set_step(n) 在主循环每轮开头调用；此后 emit/record_* 自动挂 step/seq。
    目录:
        episodes/ep_*/{telemetry,events,vlm,memory,frames,console,config,...}
    """

    def __init__(self, project_root=None):
        # 模块已在仓库根目录: 单层 dirname（与 nav_mock 一致）
        self.project_root = project_root or os.path.dirname(
            os.path.abspath(__file__))
        self.episode_dir = None
        self.frame_dir = None
        self.log_file = None
        self._stdout_before_tee = None
        self._event_queue = queue.Queue(maxsize=DEBUG_QUEUE_SIZE)
        self._writer_thread = None
        self._writer_running = False
        self._files = {}
        self._counts = {}
        self._dropped = 0
        self._last_emit = {}
        self._last_state = None
        self._last_frame_save_t = 0.0
        self.frame_saver = AsyncFrameSaver()
        self._cached_frontier_viz = {
            "frontier_xz": None,
            "frontier_dist_m": None,
            "frontier_ms": 0.0,
        }
        self._last_frontier_calc_t = 0.0
        # 时序上下文: step 由主循环绑定；同一 step 的所有数据流共享基准 ts。
        self._step = None
        self._step_ts = None
        self._step_timestamps = {}
        self._seq = 0
        self._seq_lock = threading.Lock()

    # ------------------------------------------------------------------
    #  时序上下文
    # ------------------------------------------------------------------
    def set_step(self, step, timestamp=None):
        """绑定当前主循环 step 及其统一基准时间（每轮开头调用一次）。

        参数:
            step: int，主循环步号
            timestamp: 可选 Unix 秒；默认在绑定时取一次 time.time()
        返回:
            None
        """
        self._step = int(step)
        self._step_ts = (time.time() if timestamp is None
                         else float(timestamp))
        self._step_timestamps[self._step] = self._step_ts

    def timestamp_for_step(self, step=None):
        """返回指定 step 的统一基准时间；未知 step 才使用当前墙钟时间。"""
        if step is None:
            step = self._step
        if step is not None:
            ts = self._step_timestamps.get(int(step))
            if ts is not None:
                return float(ts)
        return time.time()

    def _next_seq(self):
        """分配并返回下一个全局 seq（线程安全）。

        返回:
            int
        """
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def current_step(self):
        """当前绑定 step，未绑定时为 None。

        返回:
            int | None
        """
        return self._step

    # ------------------------------------------------------------------
    #  功能开关 / 节流
    # ------------------------------------------------------------------
    def enabled(self, feature):
        """返回 Debug 总开关和指定功能开关状态。

        参数:
            feature: DEBUG_FEATURES 键名；"snapshot" 映射到 "frame_snapshot"
        返回:
            bool
        """
        if not DEBUG_ENABLED:
            return False
        feature_key = {"snapshot": "frame_snapshot"}.get(feature, feature)
        if feature_key == "episode_log":
            return bool(DEBUG_LOG_TO_FILE and DEBUG_FEATURES.get(feature_key, True))
        if feature_key == "console_jsonl":
            # 默认跟随 episode_log；显式配置优先
            return bool(DEBUG_FEATURES.get(
                "console_jsonl",
                DEBUG_FEATURES.get("episode_log", True)))
        return bool(DEBUG_FEATURES.get(feature_key, False))

    def due(self, feature, now=None, commit=True):
        """按 DEBUG_PERIODS 判断功能是否到达下一次输出时间。

        参数:
            feature: 功能名
            now: 可选时间戳；None 用 time.time()
            commit: True 时若到期则立刻更新 last 时间；False 仅 peek，
                    调用方在真正 emit/save 成功后需 mark_due(feature)
        返回:
            bool — True 表示本周期应输出
        """
        if not self.enabled(feature):
            return False
        now = time.time() if now is None else float(now)
        period = float(DEBUG_PERIODS.get(feature, 0.0))
        last = self._last_emit.get(feature, float("-inf"))
        if period > 0 and now - last < period:
            return False
        if commit:
            self._last_emit[feature] = now
        return True

    def mark_due(self, feature, now=None):
        """在 due(..., commit=False) 之后、真正写出成功时标记周期已用。

        参数:
            feature: 功能名
            now: 可选时间戳
        返回:
            None
        """
        self._last_emit[feature] = time.time() if now is None else float(now)

    # ------------------------------------------------------------------
    #  JSON 安全 / 路径
    # ------------------------------------------------------------------
    @staticmethod
    def _json_safe(value):
        """把任意对象递归转成 JSON 可序列化结构。

        参数:
            value: 任意 Python 对象
        返回:
            dict/list/标量/str
        """
        if isinstance(value, dict):
            return {str(k): NavDebugger._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [NavDebugger._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "name"):
            return value.name
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def setup_episode(self, mode="nav_auto"):
        """创建结构化 Episode 目录并启动后台 writer / Tee。

        参数:
            mode: 运行模式标签，写入 metadata
        返回:
            (episode_dir: str, frame_dir: str)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = os.path.join(self.project_root, "episodes", f"ep_{timestamp}")
        self.frame_dir = os.path.join(self.episode_dir, "frames")
        for relative in (
            "events", "telemetry", "planner", "vlm", "memory", "config",
            "console", "coord",
        ):
            os.makedirs(os.path.join(self.episode_dir, relative), exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)

        metadata = {
            "schema_version": 2,
            "mode": mode,
            "started_at": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "debug_config": {
                "enabled": DEBUG_ENABLED,
                "features": copy.deepcopy(DEBUG_FEATURES),
                "periods": copy.deepcopy(DEBUG_PERIODS),
                "format": DEBUG_EPISODE_FORMAT,
            },
        }
        self._write_json(os.path.join(self.episode_dir, "metadata.json"), metadata)
        self._write_json(
            os.path.join(self.episode_dir, "config", "debug_config.json"),
            metadata["debug_config"])

        if DEBUG_LOG_TO_FILE and self.enabled("episode_log"):
            self.log_file = open(
                os.path.join(self.episode_dir, "nav_auto.log"),
                "a", buffering=1, encoding="utf-8")
            self._stdout_before_tee = sys.stdout
            sys.stdout = _Tee(sys.stdout, self.log_file, debugger=self)

        self._writer_running = True
        self._writer_thread = threading.Thread(
            target=self._writer, name="nav-debug-jsonl-writer", daemon=True)
        self._writer_thread.start()
        self.frame_saver.start()
        self.emit("episode_started", {"mode": mode}, feature="state_events")
        print(f"[NavDebugger] Episode 目录: {self.episode_dir}")
        return self.episode_dir, self.frame_dir

    @staticmethod
    def _write_json(path, payload):
        """同步写 JSON 文件（仅 setup/close 用）。

        参数:
            path: 文件路径
            payload: 可 JSON 序列化对象
        返回:
            None
        """
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def _path_for_kind(self, kind):
        """kind → 相对 episode 的 JSONL 路径。

        参数:
            kind: 记录类型字符串
        返回:
            str 绝对路径
        """
        if kind == "pose":
            return os.path.join(self.episode_dir, "telemetry", "pose.jsonl")
        if kind == "telemetry":
            return os.path.join(self.episode_dir, "telemetry", "status.jsonl")
        if kind in {"control", "command"}:
            return os.path.join(self.episode_dir, "telemetry", "control.jsonl")
        if kind == "performance":
            return os.path.join(self.episode_dir, "telemetry", "performance.jsonl")
        if kind == "frame":
            return os.path.join(self.episode_dir, "frames", "index.jsonl")
        if kind == "planner":
            return os.path.join(self.episode_dir, "planner", "planner_events.jsonl")
        if kind == "vlm":
            return os.path.join(self.episode_dir, "vlm", "vlm_events.jsonl")
        if kind == "coord":
            return os.path.join(self.episode_dir, "coord", "coord_events.jsonl")
        if kind == "memory":
            return os.path.join(self.episode_dir, "memory", "memory_events.jsonl")
        if kind == "console":
            return os.path.join(self.episode_dir, "console", "console.jsonl")
        return os.path.join(self.episode_dir, "events", "events.jsonl")

    # ------------------------------------------------------------------
    #  核心 emit
    # ------------------------------------------------------------------
    def emit(self, kind, payload, feature="telemetry", timestamp=None,
             step=None, seq=None):
        """非阻塞提交一条结构化 Debug 记录（统一外壳 ts/step/seq/kind/data）。

        参数:
            kind: 记录类型 (pose/telemetry/vlm/frame/console/state/...)
            payload: data 字段内容
            feature: DEBUG_FEATURES 开关名
            timestamp: 可选；默认使用该 step 在 set_step() 时绑定的基准时间
            step: 可选；默认 self._step
            seq: 可选；默认自动分配
        返回:
            bool — 是否成功入队
        """
        if self.episode_dir is None or not self.enabled(feature):
            return False
        if step is None:
            step = self._step
        if timestamp is None:
            timestamp = self.timestamp_for_step(step)
        if seq is None:
            seq = self._next_seq()
        # 固定 goal 事件 schema：新日志明确记录是否由路径/站位失效触发重算，
        # 同时让未提供该字段的旧调用保持兼容。
        if kind == "goal" and isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("from_invalid_replan", False)
            payload.setdefault("invalid_replan_reason", None)
        item = {
            "ts": float(timestamp),
            "seq": int(seq),
            "kind": kind,
            "data": self._json_safe(payload),
        }
        if step is not None:
            item["step"] = int(step)
        try:
            self._event_queue.put_nowait((self._path_for_kind(kind), item))
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def emit_console(self, text, ts=None, step=None, seq=None, stream="stdout"):
        """写入 console/console.jsonl（由 _Tee 调用；也可手动）。

        参数:
            text: 一行原文本（无换行）
            ts / step / seq: 可选覆盖
            stream: "stdout" | "stderr"
        返回:
            bool
        """
        if not self.enabled("console_jsonl") and not self.enabled("episode_log"):
            return False
        # console_jsonl 关闭时仍允许 episode_log 文本戳；此处仅 JSONL
        if not self.enabled("console_jsonl"):
            # 若未显式开 console_jsonl，但 episode_log 开着，仍写 JSONL 便于 Web 对齐
            # （enabled() 默认已跟随 episode_log；此分支防御显式 False）
            return False
        return self.emit(
            "console",
            {"text": text, "stream": stream},
            feature="console_jsonl",
            timestamp=ts,
            step=step,
            seq=seq,
        )

    # ------------------------------------------------------------------
    #  各类 record_*
    # ------------------------------------------------------------------
    def record_pose(self, pose, nav_y=None, step=None, source="slam"):
        """写 telemetry/pose.jsonl。

        参数:
            pose: (x, z, yaw) 或 None（None 时写 valid=false 占位，便于早期 step 对齐）
            nav_y: 高度
            step: 可选覆盖
            source: "slam" | "odom" 等
        返回:
            None
        """
        if not self.due("pose_history", commit=False):
            return
        if pose is None:
            ok = self.emit("pose", {
                "valid": False, "source": source,
                "nav_y": None if nav_y is None else float(nav_y),
            }, feature="pose_history", step=step)
        else:
            ok = self.emit("pose", {
                "valid": True,
                "x": float(pose[0]), "z": float(pose[1]), "yaw": float(pose[2]),
                "nav_y": None if nav_y is None else float(nav_y), "source": source,
            }, feature="pose_history", step=step)
        if ok:
            self.mark_due("pose_history")

    def record_state(self, state, step=None):
        """状态变化时记录一次；避免把 10Hz 状态轮询重复写入 Episode。

        参数:
            state: NavState 或 str
            step: 可选；默认当前绑定 step
        返回:
            None
        """
        state_name = self._json_safe(state)
        if state_name == self._last_state:
            return
        self._last_state = state_name
        if self.enabled("state_events"):
            self.emit("state", {"state": state_name},
                      feature="state_events", step=step)

    def record_action(self, action, method=None, detail=None, step=None):
        """写 action 事件。

        参数:
            action: 动作名
            method: 移动方式
            detail: 附加信息
            step: 可选
        返回:
            None
        """
        if self.enabled("action_events"):
            self.emit("action", {
                "action": action, "method": method, "detail": detail,
            }, feature="action_events", step=step)

    def record_command(self, linear, angular, pose=None, step=None):
        """写 control 遥测（节流）。"""
        if self.due("command_stats", commit=False):
            ok = self.emit("command", {
                "linear": float(linear), "angular": float(angular), "pose": pose,
            }, feature="command_stats", step=step)
            if ok:
                self.mark_due("command_stats")

    def record_performance(self, payload, step=None):
        """写 performance.jsonl（节流）。"""
        if self.due("performance_probe", commit=False):
            ok = self.emit("performance", payload,
                           feature="performance_probe", step=step)
            if ok:
                self.mark_due("performance_probe")

    def record_planner(self, payload, step=None):
        """写 planner 事件。"""
        self.emit("planner", payload, feature="planner_events", step=step)

    def record_vlm(self, payload, step=None):
        """写 vlm/vlm_events.jsonl。

        参数:
            payload: dict，建议含 stage（见模块 docstring）；兼容旧 status
            step: 可选
        返回:
            None

        约定 data 字段:
            stage: presence|detect|detect_error|no_detection|bbox_3d_fail|
                   direction|h_ask|locked_skip
            兼容: 若仅有 status 无 stage，按 inquiry/detected/error 映射 stage
        """
        data = dict(payload) if isinstance(payload, dict) else {"raw": payload}
        # 旧 status → stage 兼容映射
        if "stage" not in data and "status" in data:
            _map = {
                "inquiry": "presence",
                "detected": "detect",
                "error": "detect_error",
            }
            data["stage"] = _map.get(data["status"], data["status"])
        # 双向兼容: 有 stage 无 status 时回填 status
        if "status" not in data and "stage" in data:
            _rev = {
                "presence": "inquiry",
                "detect": "detected",
                "detect_error": "error",
            }
            data["status"] = _rev.get(data["stage"], data["stage"])
        self.emit("vlm", data, feature="vlm_events", step=step)

    def record_coord(self, payload, step=None):
        """写 coord/coord_events.jsonl：目标投影 / 点云调试 (供 Web 回放)。

        参数:
            payload: dict；建议含
                stage: "project"(投影上下文) | "pc_region"(bbox 点云统计)
                bbox / bbox_px / img_size / cam_pos_3d / cur_pose / target_3d
                等 (见 nav_auto / nav_helpers 调用处)
        返回:
            None
        """
        self.emit("coord", payload, feature="coord_debug", step=step)

    def record_memory(self, payload, step=None):
        """写 memory/memory_events.jsonl（Frontier 快照与边界告警）。"""
        self.emit("memory", payload, feature="memory_events", step=step)

    @staticmethod
    def _format_memory_console(snapshot):
        """将 frontier_snapshot 格式化成固定单行 console 摘要。"""
        def _range(lo, hi):
            if lo is None or hi is None:
                return "n/a"
            return f"{float(lo):.3f}..{float(hi):.3f}"

        max_id = snapshot.get("max_radius_mem_id")
        max_kf = snapshot.get("max_radius_kf_id")
        max_radius = snapshot.get("radius_W_max")
        max_text = "n/a"
        if max_id is not None and max_kf is not None and max_radius is not None:
            max_text = f"id{int(max_id)}/kf{int(max_kf)}/{float(max_radius):.3f}"
        return (
            f"[MEM] total={int(snapshot.get('n_total', 0))} "
            f"walked={int(snapshot.get('n_walked', 0))} "
            f"cand={int(snapshot.get('n_candidates', 0))} "
            f"anchors={int(snapshot.get('n_anchors', 0))} "
            f"scale={_range(snapshot.get('anchor_scale_min'), snapshot.get('anchor_scale_max'))} "
            f"ratio={_range(snapshot.get('scale_ratio_min'), snapshot.get('scale_ratio_max'))} "
            f"radiusW={_range(snapshot.get('radius_W_min'), snapshot.get('radius_W_max'))} "
            f"max={max_text} "
            f"removed={int(snapshot.get('removed_last', 0))}/"
            f"{int(snapshot.get('removed_total', 0))} "
            f"oob=scale:{int(snapshot.get('scale_oob_count', 0))},"
            f"radius:{int(snapshot.get('radius_oob_count', 0))},"
            f"invalid:{int(snapshot.get('invalid_scale_count', 0))}"
        )

    def save_debug_pair(self, rgb_frame, map_frame, step=None):
        """按配置周期将快照非阻塞提交给后台线程，并写入对应的帧索引。

        允许单侧为 None（占位黑图），避免 rgb_overlay/map_view 独立 due 导致丢帧。

        参数:
            rgb_frame: BGR 或 None
            map_frame: BGR 或 None
            step: 可选；默认当前绑定 step
        返回:
            None
        """
        if self.frame_dir is None or not self.enabled("frame_snapshot"):
            return
        if not DEBUG_FRAME_SAVE:
            return
        if not self.due("snapshot", commit=False):
            return
        if rgb_frame is None and map_frame is None:
            return
        if step is None:
            step = self._step if self._step is not None else 0
        self.frame_saver.push(rgb_frame, map_frame, step, self.frame_dir)
        meta = {
            "path": f"frames/step_{int(step):06d}.jpg",
            "rgb_missing": rgb_frame is None,
            "map_missing": map_frame is None,
        }
        meta.update(self.frame_saver.layout_metadata(rgb_frame, map_frame))
        if rgb_frame is not None:
            meta["rgb_shape"] = list(np.asarray(rgb_frame).shape)
        if map_frame is not None:
            meta["map_shape"] = list(np.asarray(map_frame).shape)
        ok = self.emit("frame", meta, feature="frame_snapshot", step=step)
        if ok:
            self.mark_due("snapshot")

    def query_frontier_viz_cached(self, mem, cur_pose, nav_y, obs_current,
                                  force_update=False):
        """按配置周期计算 Frontier 可视化数据，否则返回缓存。

        参数:
            mem / cur_pose / nav_y / obs_current: 同 query_frontier_for_viz
            force_update: 强制刷新
        返回:
            dict（frontier_xz / frontier_dist_m / frontier_ms / ...）
        """
        if not self.enabled("frontier_viz"):
            return self._cached_frontier_viz
        now = time.time()
        period = float(DEBUG_PERIODS.get("frontier_viz", 1.0))
        if force_update or now - self._last_frontier_calc_t >= period:
            if mem is not None and cur_pose is not None:
                self._cached_frontier_viz = query_frontier_for_viz(
                    mem, pose_xz=(cur_pose[0], cur_pose[1]),
                    nav_y=nav_y if nav_y is not None else 0.0,
                    obstacle_points=obs_current, robot_radius=ROBOT_RADIUS,
                    yaw=float(cur_pose[2]))
                self._last_frontier_calc_t = now
                if hasattr(mem, "collect_diagnostics"):
                    snapshot = mem.collect_diagnostics(
                        n_candidates=int(
                            self._cached_frontier_viz.get("n_walked", 0) or 0
                        ),
                        frontier_found=(
                            self._cached_frontier_viz.get("frontier_xz") is not None
                        ),
                        frontier_message=str(
                            self._cached_frontier_viz.get("message", "") or ""
                        ),
                    )
                    if hasattr(mem, "drain_bounds_warnings"):
                        for warning in mem.drain_bounds_warnings():
                            self.record_memory(warning)
                    self.record_memory(snapshot)
                    print(self._format_memory_console(snapshot))
        return self._cached_frontier_viz

    def _writer(self):
        """后台 JSONL writer 循环。"""
        while self._writer_running or not self._event_queue.empty():
            try:
                path, item = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                stream = self._files.get(path)
                if stream is None:
                    stream = open(path, "a", encoding="utf-8")
                    self._files[path] = stream
                stream.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                    + "\n")
                stream.flush()
                self._counts[path] = self._counts.get(path, 0) + 1
            except Exception:
                self._dropped += 1
            finally:
                self._event_queue.task_done()

    def close(self):
        """停止后台线程、恢复 stdout，并写入 Episode 汇总。

        返回:
            None
        """
        if self.episode_dir is None:
            return
        # 先 flush Tee 残留半行
        if isinstance(sys.stdout, _Tee):
            try:
                sys.stdout.flush()
            except Exception:
                pass
        if self._stdout_before_tee is not None:
            try:
                sys.stdout = self._stdout_before_tee
            except Exception:
                pass
            self._stdout_before_tee = None

        self._writer_running = False
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
        self.frame_saver.stop()
        for stream in self._files.values():
            try:
                stream.flush()
                stream.close()
            except Exception:
                pass
        self._files.clear()
        self._write_json(os.path.join(self.episode_dir, "summary.json"), {
            "schema_version": 2,
            "ended_at": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "records": self._counts,
            "dropped_records": self._dropped,
            "dropped_frames": self.frame_saver.dropped,
            "last_step": self._step,
            "last_seq": self._seq,
        })
        if self.log_file is not None:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None
        self.episode_dir = None
