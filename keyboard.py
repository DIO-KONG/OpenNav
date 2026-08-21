#!/usr/bin/env python3
"""
键盘遥控 + 实时 RGB 显示
========================
按一下走一下 (timed_move), 同时 OpenCV 窗口实时显示摄像头画面。

按键:
    W / ↑   前进一步    S / ↓   后退一步
    A / ←   左转一步    D / →   右转一步
    Q       调快        E       调慢
    Space   紧急停      P       开始/停止录制
    Esc     退出 (自动停止录制)

依赖:
    pip install requests
    ROS: rospy, cv_bridge, sensor_msgs, cv2

前置:
    1. 底盘节点 + HTTP API 服务已启动
    2. 摄像头节点已启动
    3. source camera_ws/devel/setup.bash && source tracer_ros/devel/setup.bash

用法:
    python3 keyboard_teleop_viewer.py
    python3 keyboard_teleop_viewer.py --host 192.168.1.100:8080 --dur 0.5
"""

import sys
import os
import json
import argparse
import threading
from datetime import datetime

import cv2
import numpy as np

from RGBRosConnector import RGBRosConnector

# ── 录制相关常量 ──────────────────────────────────────────────
CACHE_ROOT = "wenhao/slam_cam/data"                    # 所有录制会话的根目录
VIDEO_FPS  = 31.0                         # 输出视频帧率  设置31以保证30帧输出
FOURCC     = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码器

# 全局录制运行时（main 中用 global 操作这些变量，此处仅做声明）
recording     = False
session_dir   = None
frame_idx     = 0
sessions_meta = []
last_save_time = None


def import_rw_api():
    import importlib.util
    path = "tracer_http_interface/scripts/rw_api.py"
    spec = importlib.util.spec_from_file_location("rw_api", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TracerRobot


# 按一下 → 走一步
ACTIONS = {
    ord('w'): "forward",  ord('W'): "forward",  82: "forward",
    ord('s'): "backward", ord('S'): "backward", 84: "backward",
    ord('a'): "left",     ord('A'): "left",     81: "left",
    ord('d'): "right",    ord('D'): "right",    83: "right",
}


# ── 录制辅助函数 ──────────────────────────────────────────────

def _start_recording():
    """创建新录制会话目录，初始化状态。"""
    global recording, session_dir, frame_idx, sessions_meta

    session_id = datetime.now().strftime("rec_%Y%m%d_%H%M%S")
    session_dir = os.path.join(CACHE_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)

    recording = True
    frame_idx = 0

    sessions_meta.append({
        "id": session_id,
        "cache_dir": session_dir,
        "start_time": datetime.now().isoformat(),
        "stop_time": None,
        "frame_count": 0,
        "resolution": None,
        "fps": VIDEO_FPS,
    })
    last_save_time = None
    print(f"[REC] 开始录制 → {session_dir}")


def _stop_recording():
    """停止当前录制，写入元数据。"""
    global recording, session_dir, frame_idx, sessions_meta

    recording = False
    if sessions_meta:
        meta = sessions_meta[-1]
        meta["stop_time"] = datetime.now().isoformat()
        meta["frame_count"] = frame_idx
        # 计算帧率
        elapsed = (datetime.fromisoformat(meta["stop_time"]) 
           - datetime.fromisoformat(meta["start_time"])).total_seconds()
        if elapsed > 0:
            meta["fps"] = round(meta["frame_count"] / elapsed)
        # 从最后写入的图片读取分辨率
        if meta["resolution"] is None and session_dir:
            for fname in sorted(os.listdir(session_dir)):
                if fname.endswith(".jpg"):
                    img = cv2.imread(os.path.join(session_dir, fname))
                    if img is not None:
                        h, w = img.shape[:2]
                        meta["resolution"] = [w, h]
                    break

        meta_path = os.path.join(session_dir, "rec_meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"[REC] 停止录制 — {frame_idx} 帧 → {meta_path}")

    session_dir = None
    frame_idx = 0


def _assemble_video(meta):
    """根据元数据将帧图片拼装为 mp4 视频文件。"""
    cache_dir = meta["cache_dir"]
    if not os.path.isdir(cache_dir):
        print(f"[WARN] 目录不存在，跳过: {cache_dir}")
        return

    frames = sorted(f for f in os.listdir(cache_dir) if f.endswith(".jpg"))
    if not frames:
        print(f"[WARN] {meta['id']}: 无帧，跳过视频组装")
        return

    # 从第一帧推断分辨率（若 meta 中缺失）
    resolution = meta.get("resolution")
    if resolution is None:
        first = cv2.imread(os.path.join(cache_dir, frames[0]))
        if first is not None:
            h, w = first.shape[:2]
            resolution = [w, h]
            meta["resolution"] = resolution

    if resolution is None:
        print(f"[ERROR] {meta['id']}: 无法获取分辨率，跳过")
        return

    video_path = os.path.join(CACHE_ROOT, f"{meta['id']}.mp4")
    writer = cv2.VideoWriter(video_path, FOURCC, meta.get("fps", VIDEO_FPS),
                             (resolution[0], resolution[1]))
    if not writer.isOpened():
        print(f"[ERROR] {meta['id']}: 无法创建视频写入器")
        return

    written = 0
    for fname in frames:
        img = cv2.imread(os.path.join(cache_dir, fname))
        if img is None:
            continue
        writer.write(img)
        written += 1

    writer.release()
    print(f"[VIDEO] {video_path}  — {written} 帧, {resolution[0]}×{resolution[1]}, {meta.get('fps', VIDEO_FPS)}fps raw, encoded to {VIDEO_FPS}fps")


def main():
    global recording, session_dir, frame_idx, session_meta, writer_thread, writer_exit_flag


    p = argparse.ArgumentParser(description="键盘遥控 + 实时 RGB 显示 (按一下走一下)")
    p.add_argument("--host", default="localhost:8080", help="底盘 HTTP API 地址")
    p.add_argument("--linear", type=float, default=0.3, help="线速度 m/s")
    p.add_argument("--angular", type=float, default=0.5, help="角速度 rad/s")
    p.add_argument("--dur", type=float, default=1.0, help="每步持续时间 秒")
    args = p.parse_args()

    TracerRobot = import_rw_api()
    bot = TracerRobot(base_url=f"http://{args.host}")
    bot.stop()

    cam = RGBRosConnector(topic="/camera_f/color/image_raw")
    print("等待摄像头首帧...")
    if not cam.wait_for_frame(timeout=5):
        print("[ERROR] 5秒内没收到 RGB 帧，退出")
        bot.stop()
        sys.exit(1)
    print("[OK] 摄像头就绪")

    lin_v = args.linear
    ang_v = args.angular
    dur = args.dur
    move_thread = None   # 运动线程 (None=空闲, 避免阻塞主循环导致窗口卡住)

    print("\n按键说明:")
    print("  W/↑ 前进一步  S/↓ 后退一步  A/← 左转一步  D/→ 右转一步")
    print("  Q 调快         E 调慢         Space 紧急停   Esc 退出")
    print("  P 开始/停止录制")
    print(f"  速度: 线={lin_v:.2f}m/s  角={ang_v:.2f}rad/s  步长={dur:.1f}s\n")

    while True:
        f = cam.get_frame()

        # 录制帧写入（原始帧，不含 HUD）
        now = datetime.now()
        if recording and f is not None:
            if last_save_time is None or (now - last_save_time).total_seconds() >= 1.0 / VIDEO_FPS:
                fname = f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(os.path.join(session_dir, fname), f)
                frame_idx += 1

        display = f.copy() if f is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        hud = f"v={lin_v:.2f}  w={ang_v:.2f}  dur={dur:.1f}s"
        cv2.putText(display, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if recording:
            rec_text = f"REC {frame_idx:06d}"
            cv2.putText(display, rec_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("AgX Teleop", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc
            if recording:
                _stop_recording()
            break

        if key == ord(' '):  # Space
            bot.stop()
            print("  紧急停")
            continue

        if key == ord('p') or key == ord('P'):
            if not recording:
                _start_recording()
            else:
                _stop_recording()
            continue

        if key == ord('q') or key == ord('Q'):
            lin_v = min(lin_v + 0.1, 1.0)
            ang_v = min(ang_v + 0.1, 1.5)
            print(f"  调快: 线={lin_v:.2f}  角={ang_v:.2f}")
            continue

        if key == ord('e') or key == ord('E'):
            lin_v = max(lin_v - 0.1, 0.1)
            ang_v = max(ang_v - 0.1, 0.1)
            print(f"  调慢: 线={lin_v:.2f}  角={ang_v:.2f}")
            continue

        action = ACTIONS.get(key)
        if action:
            if move_thread is not None and move_thread.is_alive():
                print("  [SKIP] 上一条运动还在执行")
            else:
                try:
                    if action == "forward":
                        fn = lambda: bot.move_forward(
                            speed=lin_v, duration=dur)
                    elif action == "backward":
                        fn = lambda: bot.move_backward(
                            speed=lin_v, duration=dur)
                    elif action == "left":
                        fn = lambda: bot.turn_left(
                            angular_speed=ang_v, duration=dur)
                    elif action == "right":
                        fn = lambda: bot.turn_right(
                            angular_speed=ang_v, duration=dur)
                    move_thread = threading.Thread(
                        target=fn, daemon=True)
                    move_thread.start()
                    print(f"  {action}  v={lin_v:.2f}  w={ang_v:.2f}"
                          f"  dur={dur:.1f}s")
                except Exception as e:
                    print(f"[WARN] 失败: {e}")

    bot.stop()

    # ── 退出前拼装视频 ──────────────────────────────────────
    if sessions_meta:
        print(f"\n正在拼装 {len(sessions_meta)} 个录制会话...")
        for meta in sessions_meta:
            _assemble_video(meta)
    else:
        print("\n无录制会话，跳过视频拼装")

    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
