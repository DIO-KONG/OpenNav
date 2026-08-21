#!/usr/bin/env python3
"""episodes/app/server.py — Episode 结构化 Debug 数据 Web 可视化服务器。

独立运行于 0.0.0.0:8011 端口，解析 episodes/ 目录下的结构化 JSONL、快照与日志，
并提供 REST API 与交互式 Web 界面。
"""

import os
import re
import json
import shutil
import argparse
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, request

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
EPISODES_DIR = PROJECT_ROOT / "episodes"

# 合法 episode 目录名: ep_YYYYMMDD_HHMMSS 或 eps_YYYYMMDD_HHMMSS
_EP_NAME_RE = re.compile(r"^(ep|eps)_(\d{8})_(\d{6})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RECENT_LIMIT = 5

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

def _read_jsonl(file_path):
    records = []
    if not file_path.exists():
        return records
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def _normalize_goal_events(events):
    """为新旧 episode 提供一致的 goal 事件字段。"""
    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "goal":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
            event["data"] = data
        data.setdefault("from_invalid_replan", False)
        data.setdefault("invalid_replan_reason", None)
    return events

def _read_json(file_path):
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None

def _is_episode_dir_name(name):
    """严格校验 episode 目录名，防止路径穿越与脏目录。"""
    return bool(isinstance(name, str) and _EP_NAME_RE.match(name))

def _episode_kind(name):
    """返回 'ep' 或 'eps'。"""
    m = _EP_NAME_RE.match(name or "")
    if not m:
        return None
    return m.group(1)

def _parse_episode_stamp(name):
    """从目录名解析 (date_str 'YYYY-MM-DD', hour int)。非法返回 (None, None)。"""
    m = _EP_NAME_RE.match(name or "")
    if not m:
        return None, None
    ymd = m.group(2)  # YYYYMMDD
    hms = m.group(3)  # HHMMSS
    date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
    hour = int(hms[0:2])
    return date_str, hour

def _stamp_suffix(name):
    """返回 YYYYMMDD_HHMMSS 后缀；非法返回 None。"""
    m = _EP_NAME_RE.match(name or "")
    if not m:
        return None
    return f"{m.group(2)}_{m.group(3)}"

def _resolve_episode_dir(ep_id):
    """校验 id 合法且目录落在 EPISODES_DIR 内。

    返回:
        (Path|None, error_dict|None, http_status|None)
    """
    if not _is_episode_dir_name(ep_id):
        return None, {"error": "invalid_episode_id"}, 400
    ep_dir = (EPISODES_DIR / ep_id).resolve()
    try:
        ep_dir.relative_to(EPISODES_DIR.resolve())
    except ValueError:
        return None, {"error": "invalid_episode_id"}, 400
    if not ep_dir.exists() or not ep_dir.is_dir():
        return None, {"error": "Episode not found"}, 404
    return ep_dir, None, None

def _build_episode_item(item):
    """组装列表项 dict。"""
    name = item.name
    kind = _episode_kind(name) or "ep"
    date_str, hour = _parse_episode_stamp(name)
    meta = _read_json(item / "metadata.json") or {}
    summary_path = item / "summary.json"
    summary = _read_json(summary_path) or {}
    ended = summary_path.exists()

    frame_count = 0
    frames_idx = item / "frames" / "index.jsonl"
    if frames_idx.exists():
        frame_count = len(_read_jsonl(frames_idx))
    elif (item / "frames").exists():
        frame_count = len(list((item / "frames").glob("*.jpg")))

    stamp = _stamp_suffix(name) or name
    started_fallback = meta.get("started_at")
    if not started_fallback:
        # 兼容 ep_ / eps_ 前缀
        started_fallback = stamp

    return {
        "id": name,
        "started_at": started_fallback,
        "mode": meta.get("mode", "nav_auto"),
        "frame_count": frame_count,
        "summary": summary,
        "kind": kind,
        "success": kind == "eps",
        "ended": ended,
        "date": date_str,
        "hour": hour,
    }

def _iter_episode_dirs():
    """倒序产出合法 episode 目录 Path。"""
    if not EPISODES_DIR.exists():
        return
    for item in sorted(EPISODES_DIR.iterdir(), reverse=True):
        if item.is_dir() and _is_episode_dir_name(item.name):
            yield item

@app.after_request
def _cache_policy(resp):
    """缓存策略: API/HTML 禁缓存; JPEG 快照可缓存, 避免播放时反复拉同帧导致卡顿。"""
    ct = (resp.content_type or "").split(";")[0].strip().lower()
    # 静态帧与静态资源允许短缓存 (文件名含 step, 内容不变)
    if ct in ("image/jpeg", "image/jpg", "image/png", "text/css",
              "application/javascript", "text/javascript"):
        resp.headers["Cache-Control"] = "public, max-age=3600"
        resp.headers.pop("Pragma", None)
        return resp
    # 列表/数据 API 与 HTML: 仍禁止缓存, 保证刷新看到新 episode
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/episodes", methods=["GET"])
def list_episodes():
    """列出 episode。

    Query:
        kind: all|ep|eps  (default all)
        all:  0=最新 5 条; 1=全部
        date: YYYY-MM-DD  (仅 all=1)
        hour: 0-23        (仅 all=1 且给了 date)
    """
    kind = (request.args.get("kind") or "all").strip().lower()
    if kind not in ("all", "ep", "eps"):
        return jsonify({"error": "invalid_kind"}), 400

    all_mode = (request.args.get("all") or "0").strip() in ("1", "true", "True", "yes")
    date_q = (request.args.get("date") or "").strip()
    hour_q = (request.args.get("hour") or "").strip()

    if date_q and not _DATE_RE.match(date_q):
        return jsonify({"error": "invalid_date"}), 400
    hour_val = None
    if hour_q != "":
        try:
            hour_val = int(hour_q)
        except ValueError:
            return jsonify({"error": "invalid_hour"}), 400
        if hour_val < 0 or hour_val > 23:
            return jsonify({"error": "invalid_hour"}), 400

    episodes = []
    for item in _iter_episode_dirs():
        item_kind = _episode_kind(item.name)
        if kind != "all" and item_kind != kind:
            continue

        if all_mode:
            date_str, hour = _parse_episode_stamp(item.name)
            if date_q and date_str != date_q:
                continue
            # hour 仅在给了 date 时生效
            if date_q and hour_val is not None and hour != hour_val:
                continue
        # recent 模式忽略 date/hour

        episodes.append(_build_episode_item(item))
        if not all_mode and len(episodes) >= _RECENT_LIMIT:
            break

    return jsonify(episodes)

@app.route("/api/episodes/<ep_id>/mark_success", methods=["POST"])
def mark_episode_success(ep_id):
    """将 episode 标记/取消 success：ep_ ↔ eps_ 目录改名。

    Body JSON: { "success": true|false }
    仅已结束（存在 summary.json）允许改名。
    """
    ep_dir, err, status = _resolve_episode_dir(ep_id)
    if err:
        return jsonify(err), status

    body = request.get_json(silent=True) or {}
    if "success" not in body:
        return jsonify({"error": "missing_success"}), 400
    want_success = bool(body.get("success"))

    kind = _episode_kind(ep_id)
    stamp = _stamp_suffix(ep_id)
    if not kind or not stamp:
        return jsonify({"error": "invalid_episode_id"}), 400

    if want_success and kind != "ep":
        return jsonify({"error": "already_success", "id": ep_id}), 400
    if (not want_success) and kind != "eps":
        return jsonify({"error": "not_success", "id": ep_id}), 400

    # 仅已结束可标记，避免导航写盘中 rename
    if not (ep_dir / "summary.json").exists():
        return jsonify({"error": "episode_not_ended"}), 409

    new_id = f"eps_{stamp}" if want_success else f"ep_{stamp}"
    new_dir = (EPISODES_DIR / new_id).resolve()
    try:
        new_dir.relative_to(EPISODES_DIR.resolve())
    except ValueError:
        return jsonify({"error": "invalid_target"}), 400

    if new_dir.exists():
        return jsonify({"error": "target_exists", "target": new_id}), 409

    try:
        os.rename(str(ep_dir), str(new_dir))
    except OSError as exc:
        return jsonify({"error": "rename_failed", "detail": str(exc)}), 500

    return jsonify({
        "id": new_id,
        "old_id": ep_id,
        "success": want_success,
        "kind": "eps" if want_success else "ep",
    })

@app.route("/api/episodes/<ep_id>", methods=["DELETE"])
def delete_episode(ep_id):
    """删除已结束的 episode 目录（ep_ / eps_ 均可）。

    仅当 summary.json 存在时允许删除，避免误删正在写盘的 episode。
    """
    ep_dir, err, status = _resolve_episode_dir(ep_id)
    if err:
        return jsonify(err), status

    if not (ep_dir / "summary.json").exists():
        return jsonify({"error": "episode_not_ended"}), 409

    # 再次确认仍落在 EPISODES_DIR 内
    try:
        ep_dir.resolve().relative_to(EPISODES_DIR.resolve())
    except ValueError:
        return jsonify({"error": "invalid_episode_id"}), 400

    try:
        shutil.rmtree(str(ep_dir))
    except OSError as exc:
        return jsonify({"error": "delete_failed", "detail": str(exc)}), 500

    return jsonify({"ok": True, "id": ep_id})

@app.route("/api/episodes/<ep_id>/data", methods=["GET"])
def get_episode_data(ep_id):
    ep_dir, err, status = _resolve_episode_dir(ep_id)
    if err:
        return jsonify(err), status

    metadata = _read_json(ep_dir / "metadata.json")
    summary = _read_json(ep_dir / "summary.json")
    debug_config = _read_json(ep_dir / "config" / "debug_config.json")

    poses = _read_jsonl(ep_dir / "telemetry" / "pose.jsonl")
    statuses = _read_jsonl(ep_dir / "telemetry" / "status.jsonl")
    events = _normalize_goal_events(
        _read_jsonl(ep_dir / "events" / "events.jsonl")
    )
    vlm_events = _read_jsonl(ep_dir / "vlm" / "vlm_events.jsonl")
    coord_events = _read_jsonl(ep_dir / "coord" / "coord_events.jsonl")
    memory_events = _read_jsonl(ep_dir / "memory" / "memory_events.jsonl")
    frames = _read_jsonl(ep_dir / "frames" / "index.jsonl")
    performance = _read_jsonl(ep_dir / "telemetry" / "performance.jsonl")
    # 新 schema: console/console.jsonl; 兼容旧路径 events/console.jsonl
    console_events = _read_jsonl(ep_dir / "console" / "console.jsonl")
    if not console_events:
        console_events = _read_jsonl(ep_dir / "events" / "console.jsonl")

    # 如果没有 index.jsonl，自动对 step_XXXXXX.jpg 进行推断补全
    if not frames and (ep_dir / "frames").exists():
        jpg_files = sorted((ep_dir / "frames").glob("step_*.jpg"))
        for img_p in jpg_files:
            try:
                step_num = int(img_p.stem.replace("step_", ""))
                frames.append({
                    "step": step_num,
                    "data": {
                        "path": f"frames/{img_p.name}"
                    }
                })
            except Exception:
                pass

    # 新快照由 nav_debug 写 layout=streams_v1；旧 episode 保持 raw_pair_v1，
    # 由前端根据 rgb_shape/map_shape 重排成一致的双流样式。
    for frame in frames:
        data = frame.setdefault("data", {})
        data.setdefault("layout", "raw_pair_v1")

    return jsonify({
        "id": ep_id,
        "metadata": metadata,
        "summary": summary,
        "debug_config": debug_config,
        "poses": poses,
        "statuses": statuses,
        "events": events,
        "vlm_events": vlm_events,
        "coord_events": coord_events,
        "memory_events": memory_events,
        "frames": frames,
        "performance": performance,
        "console_events": console_events,
    })

@app.route("/episodes/<ep_id>/frames/<filename>", methods=["GET"])
def get_frame_file(ep_id, filename):
    """返回 JPEG 快照; 允许浏览器缓存, 播放时不再反复拉同帧。"""
    ep_dir, err, status = _resolve_episode_dir(ep_id)
    if err:
        return jsonify(err), status
    # 文件名只允许简单 jpg 名，防止 frames/../ 穿越
    if not re.match(r"^[\w.\-]+$", filename or ""):
        return jsonify({"error": "invalid_filename"}), 400
    file_path = (ep_dir / "frames" / filename).resolve()
    try:
        file_path.relative_to((ep_dir / "frames").resolve())
    except ValueError:
        return jsonify({"error": "invalid_filename"}), 400
    if not file_path.exists():
        return jsonify({"error": "Frame not found"}), 404
    # conditional + max_age: 304 / 磁盘缓存, 缓解播放刷新跟不上的问题
    try:
        return send_file(
            str(file_path),
            mimetype="image/jpeg",
            conditional=True,
            max_age=3600,
            etag=True,
        )
    except TypeError:
        # 兼容旧 Flask (无 max_age/etag 关键字)
        resp = send_file(str(file_path), mimetype="image/jpeg", conditional=True)
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

@app.route("/episodes/<ep_id>/log", methods=["GET"])
def get_episode_log(ep_id):
    ep_dir, err, status = _resolve_episode_dir(ep_id)
    if err:
        return jsonify(err), status
    log_path = ep_dir / "nav_auto.log"
    if not log_path.exists():
        return jsonify({"error": "Log file not found"}), 404
    return send_file(str(log_path), mimetype="text/plain")

def main():
    parser = argparse.ArgumentParser(description="Episode Debug Web Visualizer Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8011, help="Port to listen on")
    args = parser.parse_args()

    print(f"🚀 Episode Debug Viewer running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
