#!/usr/bin/env python3
"""
keyboard_qwen.py — 键盘遥控 + Qwen VLM 在线 debug
==================================================
在 keyboard.py 基础上集成 Qwen VLM 在线问询:
  - 移动: W/A/S/D/Q/E/Space/Esc 
  - I:    终端输入自定义 prompt, 以当前帧送 Qwen (vlm_chat, 纯文本回复)
  - O:    终端输入检测目标, 以当前帧送 Qwen 做 object detection
          (guided JSON + DETECTION_PROMPT 模板), 再用 Mobile SAM 出 mask
  - R:    清空上次 prompt 和检测目标
  - 自动保存 debug 结果到 debug_qwen/ (帧 + 响应)

可视化: OpenCV 窗口 (实时 RGB + 状态 + 检测 bbox + SAM mask + 聊天回复)

线程架构:
  - 主线程: cv2.imshow + waitKey (移动 + I/O/R/Esc)
  - 终端输入线程: 等 input_mode 信号后提示输入 (I→chat, O→detect)
  - VLM 调用: 起一个 daemon thread 跑 API, 不阻塞主循环

依赖:
  pip install openai opencv-python numpy
  ROS: rospy, cv_bridge, sensor_msgs

前置:
  1. 底盘节点 + HTTP API 服务已启动
  2. 摄像头节点已启动
  3. Qwen VLM 服务 (vLLM) 在 --vlm-url 运行

用法:
  python3 keyboard_qwen.py
  python3 keyboard_qwen.py --host 192.168.1.100:8080 --vlm-url http://localhost:8222/v1
"""

import sys
import os
import json
import time
import base64
import queue
import argparse
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from RGBRosConnector import RGBRosConnector


# ============================================================
#  配置
# ============================================================

DEFAULT_VLM_URL = "http://localhost:8222/v1"
DEFAULT_VLM_MODEL = ""

DETECTION_PROMPT = """Detect {} and identify their reference designators (reference numbers), and output the results in the following JSON format:
```json
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "type_of_component", "sub_label": "Reference_designator"}},
  ...
]
```"""

DEBUG_DIR = Path("debug_qwen")

MOBILE_SAM_CHECKPOINT_PATH = '/home/agilex/yinzecheng/opennav/mobile_sam/mobile_sam.pt'

# --- Mobile SAM predictor (懒加载) ---
# 不在模块顶层直接加载, 避免导入时就占用 GPU 显存。
# 首次调用 sam_segment() 时才加载模型 + 建 predictor。
_sam_predictor = None


def get_sam_predictor():
    """懒加载 SAM predictor。首次调用时加载 vit_t 模型到 CUDA。"""
    global _sam_predictor
    if _sam_predictor is None:
        from mobile_sam import SamPredictor, sam_model_registry
        model_type = "vit_t"
        sam = sam_model_registry[model_type](
            checkpoint=MOBILE_SAM_CHECKPOINT_PATH)
        sam = sam.to('cuda')
        _sam_predictor = SamPredictor(sam)
        print("[SAM] 模型已加载 (vit_t, cuda)")
    return _sam_predictor


def sam_segment(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    batch of xyxy inference, so return batch masks.

    Get masks for all detected bounding boxes using SAM.
    Arguments:
            image: image of shape (H, W, 3) — BGR numpy
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
    Returns:
            masks: masks of shape (N, H, W), 或 None (无 bbox 时)
    """
    if len(xyxy) == 0:
        return None
    predictor = get_sam_predictor()
    predictor.set_image(image)
    result_masks = []
    for box in xyxy:
            masks, scores, logits = predictor.predict(
                    box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
    return np.array(result_masks)


# ============================================================
#  VLM 辅助函数 (适配自 AIO.py, 改为接收 numpy)
# ============================================================

def numpy_to_data_url(img_bgr, quality=85):
    """BGR numpy → base64 JPEG data URL (对标 AIO.image_to_data_url)。"""
    _, buf = cv2.imencode(".jpg", img_bgr,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text):
    """从模型返回文本中提取 JSON (来自 AIO.py)。"""
    import re
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"```json\s*(.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1) + "]")
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"```\s*(.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1) + "]")
        return json.loads(text.strip())
    except (json.JSONDecodeError, Exception):
        print("\033[93mWARNING: JSON parse error. Raw:\033[0m")
        print(repr(text))
        return None


def qwen_bbox_to_pixel(bbox_2d, w, h):
    """Qwen 归一化坐标 (0-1000) → 像素坐标 (来自 AIO.py)。"""
    qx1, qy1, qx2, qy2 = [float(v) for v in bbox_2d]
    x1 = int(qx1 / 1000 * w)
    y1 = int(qy1 / 1000 * h)
    x2 = int(qx2 / 1000 * w)
    y2 = int(qy2 / 1000 * h)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def vlm_detect(client, model, img_bgr, prompt):
    """
    以 guided JSON 调 Qwen 做 object detection。
    返回 [{bbox_2d, label, sub_label}, ...] 或 None。
    (适配自 AIO.qwen35, 改为接收 numpy 而非文件路径)
    """
    image_url = numpy_to_data_url(img_bgr)
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "bbox_2d": {"type": "array",
                            "items": {"type": "number"}},
                "label": {"type": "string"},
                "sub_label": {"type": "string"}
            },
            "required": ["bbox_2d", "label", "sub_label"]
        }
    }
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
        }],
        temperature=0,
        max_tokens=1024,
        extra_body={
            "guided_json": schema,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    return extract_json(response.choices[0].message.content)


def vlm_chat(client, model, img_bgr, prompt):
    """
    以当前帧 + prompt 问 Qwen (无对话上下文, 每次独立)。
    返回 reply_text。
    """
    image_url = numpy_to_data_url(img_bgr)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    }]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def vlm_detect_with_sam(client, model, img_bgr, prompt):
    """VLM detect + SAM segment, 一次搞定。返回 (dets, masks)。
    只取第一个检测结果 (避免多目标时 SAM 对异常 bbox 崩溃)。"""
    dets = vlm_detect(client, model, img_bgr, prompt)
    masks = None
    if dets:
        dets = dets[:1]  # 只取第一个检测
        h, w = img_bgr.shape[:2]
        xyxy = []
        for det in dets:
            bbox = det.get("bbox_2d", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = qwen_bbox_to_pixel(bbox, w, h)
                xyxy.append([x1, y1, x2, y2])
        if xyxy:
            try:
                masks = sam_segment(img_bgr, np.array(xyxy))
            except Exception as se:
                print(f"\033[93m[SAM] 分割失败: {se}\033[0m")
    return dets, masks


# ============================================================
#  VLM 异步调用 (简单 daemon thread, 无锁)
# ============================================================

def start_vlm_task(func, *args):
    """
    起 daemon thread 跑 VLM 调用, 不阻塞主循环。
    返回 thread 对象, 主循环检查 thread.is_alive() + thread.result。
    """
    def _run():
        t.result = None
        t.error = None
        try:
            t.result = func(*args)
        except Exception as e:
            t.error = str(e)
    t = threading.Thread(target=_run, daemon=True)
    t.result = None
    t.error = None
    t.start()
    return t


# ============================================================
#  Debug 保存
# ============================================================

def save_debug(tag, frame=None, data=None):
    """保存 debug 帧和数据到 DEBUG_DIR。"""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = []
    if frame is not None:
        p = DEBUG_DIR / f"{ts}_{tag}_frame.png"
        cv2.imwrite(str(p), frame)
        paths.append(p)
    if data is not None:
        if isinstance(data, (list, dict)):
            p = DEBUG_DIR / f"{ts}_{tag}_result.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif isinstance(data, str):
            p = DEBUG_DIR / f"{ts}_{tag}_response.txt"
            with open(p, "w", encoding="utf-8") as f:
                f.write(data)
        paths.append(p)
    return paths


# ============================================================
#  可视化绘制
# ============================================================

def draw_detections(frame, detections, masks=None):
    """在帧上绘制 bbox (绿色框 + 标签) + SAM mask (半透明叠加)。"""
    if not detections:
        return
    h, w = frame.shape[:2]
    # --- SAM mask 半透明叠加 ---
    if masks is not None:
        # 每个检测框用不同颜色, 便于区分
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255),
                  (255, 0, 255), (255, 255, 0),
                  (0, 128, 255), (128, 255, 0), (255, 128, 0)]
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            color = colors[i % len(colors)]
            overlay = frame.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    # --- bbox 框 + 标签 ---
    for i, det in enumerate(detections):
        bbox = det.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = qwen_bbox_to_pixel(bbox, w, h)
        label = det.get("label", "")
        sub = det.get("sub_label", "")
        text = f"{label} {sub}".strip()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6),
                       (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_overlay(frame, status):
    """在帧顶部画半透明状态条。"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 93), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 16
    vlm_color = (0, 200, 255) if status["vlm_busy"] else (0, 255, 0)
    cv2.putText(frame,
                f"[VLM] {'BUSY...' if status['vlm_busy'] else 'IDLE'}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, vlm_color, 1)
    y += 15
    cv2.putText(frame,
                f"[BOT] v={status['lin']:.2f} w={status['ang']:.2f}"
                f" dur={status['dur']:.1f}s",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 1)
    y += 15
    cv2.putText(frame,
                f"[PROMPT] {status['last_prompt'][:50]}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1)
    y += 15
    cv2.putText(frame,
                f"[TGT] {status['det_target'][:50]}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 165, 0), 1)
    y += 15
    det_str = (f"{status['det_count']} objs"
               if status["det_count"] > 0 else "none")
    mask_str = (f"SAM:{status['mask_count']}"
                if status["mask_count"] > 0 else "")
    cv2.putText(frame,
                f"[DET] {det_str}  {mask_str}  {status['last_det'][:40]}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1)
    # 底部提示
    hint = ("[I]Chat  [O]Detect  [R]Clear  "
            "[WASD]Move  [Q/E]Speed  [Space]Stop  [Esc]Exit")
    cv2.putText(frame, hint, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (140, 140, 140), 1)


def draw_chat_response(frame, response):
    """在帧底部画聊天回复 (自动换行, 最多 5 行)。"""
    if not response:
        return
    h, w = frame.shape[:2]
    max_chars = max(1, w // 7)
    lines = []
    for raw in response.split("\n"):
        while len(raw) > max_chars:
            lines.append(raw[:max_chars])
            raw = raw[max_chars:]
        lines.append(raw)
    lines = lines[:5]
    y_start = h - 24 - len(lines) * 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start - 4), (w, h - 18),
                   (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line[:max_chars],
                    (6, y_start + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 0), 1)


# ============================================================
#  底盘导入 (同 keyboard.py)
# ============================================================

def import_rw_api():
    import importlib.util
    path = "tracer_http_interface/scripts/rw_api.py"
    spec = importlib.util.spec_from_file_location("rw_api", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TracerRobot


ACTIONS = {
    ord("w"): "forward", ord("W"): "forward",
    ord("s"): "backward", ord("S"): "backward",
    ord("a"): "left", ord("A"): "left",
    ord("d"): "right", ord("D"): "right",
}


# ============================================================
#  终端输入线程
# ============================================================

def terminal_input_loop(input_queue, input_mode, running):
    """daemon 线程: 等 input_mode 被主线程设置后, 提示用户输入。

    input_mode[0]: None=空闲等待, "chat"=问自定义prompt, "detect"=问检测目标
    input_queue: 放 (mode, text) 元组供主循环消费
    """
    while running[0]:
        mode = input_mode[0]
        if mode is None:
            time.sleep(0.02)
            continue
        input_mode[0] = None  # 消费掉
        try:
            if mode == "chat":
                line = input("\033[36mchat> \033[0m")
            elif mode == "detect":
                line = input("\033[35mdetect what? > \033[0m")
            else:
                continue
            input_queue.put((mode, line.strip()))
        except (EOFError, OSError):
            break


# ============================================================
#  主函数
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="键盘遥控 + Qwen VLM 在线 debug")
    p.add_argument("--host", default="localhost:8080",
                   help="底盘 HTTP API 地址")
    p.add_argument("--vlm-url", default=DEFAULT_VLM_URL,
                   help="Qwen VLM 服务地址 (OpenAI 兼容)")
    p.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL,
                   help="VLM 模型名 (空=用服务器默认)")
    p.add_argument("--linear", type=float, default=0.3,
                   help="线速度 m/s")
    p.add_argument("--angular", type=float, default=0.5,
                   help="角速度 rad/s")
    p.add_argument("--dur", type=float, default=1.0,
                   help="每步持续时间 秒")
    p.add_argument("--det-prompt", default=DETECTION_PROMPT,
                   help="object detection 的 prompt")
    args = p.parse_args()

    # --- 底盘 ---
    TracerRobot = import_rw_api()
    bot = TracerRobot(base_url=f"http://{args.host}")
    bot.stop()

    # --- 摄像头 ---
    cam = RGBRosConnector(topic="/camera_f/color/image_raw")
    print("等待摄像头首帧...")
    if not cam.wait_for_frame(timeout=5):
        print("[ERROR] 5 秒内没收到 RGB 帧，退出")
        bot.stop()
        sys.exit(1)
    print("[OK] 摄像头就绪")

    # --- VLM ---
    client = OpenAI(
        base_url=args.vlm_url,
        api_key="EMPTY",
        timeout=30.0,
    )
    model = args.vlm_model
    print(f"[OK] VLM 连接: {args.vlm_url}  model={model or '(default)'}")

    # --- 终端输入线程 (按键触发, 不常驻) ---
    input_queue = queue.Queue()
    input_mode = [None]   # None="空闲", "chat"=等输入prompt, "detect"=等输入目标
    running = [True]
    input_thread = threading.Thread(
        target=terminal_input_loop,
        args=(input_queue, input_mode, running), daemon=True)
    input_thread.start()

    # --- 状态 ---
    lin_v = args.linear
    ang_v = args.angular
    dur = args.dur
    last_prompt = ""           # 记住上次输入的 prompt (空Enter复用)
    last_det_target = ""       # 记住上次 detection 目标 (空Enter复用)
    last_detections = None
    last_masks = None          # SAM 分割结果 (N, H, W) 或 None
    last_det_summary = ""
    last_chat_response = ""

    # VLM 异步线程 (None=空闲)
    vlm_thread = None
    vlm_task_type = None      # "detect" 或 "chat"
    # 运动线程 (None=空闲, 避免阻塞主循环导致窗口卡住)
    move_thread = None

    print("\n\033[92m按键说明:\033[0m")
    print("  W/↑ 前进  S/↓ 后退  A/← 左转  D/→ 右转")
    print("  Q 调快    E 调慢    Space 停   Esc 退出")
    print("  I 问Qwen(终端输入自定义prompt)  O 检测(终端输入目标)")
    print("  R 清空上次prompt和检测目标")
    print(f"  速度: 线={lin_v:.2f}m/s  角={ang_v:.2f}rad/s"
          f"  步长={dur:.1f}s")
    print(f"  Debug 保存到: {DEBUG_DIR.resolve()}/\n")

    # --- 主循环 ---
    while True:
        frame = cam.get_frame()
        if frame is not None:
            display = frame.copy()
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1) 检查 VLM 线程是否完成
        if vlm_thread is not None and not vlm_thread.is_alive():
            if vlm_thread.error:
                print(f"\033[91m[VLM ERROR] {vlm_thread.error}\033[0m")
            elif vlm_thread.result is not None:
                if vlm_task_type == "detect":
                    dets, masks = vlm_thread.result
                    last_detections = dets
                    last_masks = masks
                    n = len(dets) if dets else 0
                    n_masks = len(masks) if masks is not None else 0
                    last_det_summary = (
                        ", ".join(d.get("label", "?") for d in dets[:5])
                        if dets else "(none)")
                    # 保存 (画上 bbox + mask 再存)
                    save_frame = vlm_thread.img.copy()
                    draw_detections(save_frame, dets, masks)
                    save_debug("det", frame=save_frame, data=dets)
                    if masks is not None:
                        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                        ts_str = datetime.now().strftime(
                            "%Y%m%d_%H%M%S_%f")[:-3]
                        np.savez_compressed(
                            str(DEBUG_DIR / f"{ts_str}_det_masks.npz"),
                            masks=masks)
                    print(f"\033[32m[DET] {n} objects: "
                          f"{last_det_summary}  | SAM masks: {n_masks}"
                          f"\033[0m")
                elif vlm_task_type == "chat":
                    reply = vlm_thread.result
                    last_chat_response = reply
                    save_debug("chat", frame=vlm_thread.img,
                               data=f"PROMPT:\n{vlm_thread.prompt}"
                                    f"\n\nRESPONSE:\n{reply}")
                    print(f"\033[33m[CHAT] Qwen: {reply[:120]}"
                          f"{'...' if len(reply) > 120 else ''}\033[0m")
            vlm_thread = None
            vlm_task_type = None

        vlm_busy = vlm_thread is not None

        # 2) 检查终端输入 (I=chat, O=detect 触发)
        if not vlm_busy:
            try:
                mode, text = input_queue.get_nowait()
                if mode == "chat":
                    prompt = text
                    if prompt == "" and last_prompt:
                        prompt = last_prompt  # 空Enter, 复用上次
                    if prompt and frame is not None:
                        last_prompt = prompt
                        last_detections = None  # 清除旧检测
                        last_masks = None
                        last_det_summary = ""
                        last_det_target = ""  # chat 模式不关心检测目标
                        vlm_thread = start_vlm_task(
                            vlm_chat, client, model, frame.copy(), prompt)
                        vlm_thread.img = frame.copy()
                        vlm_thread.prompt = prompt
                        vlm_task_type = "chat"
                        print(f"\033[36m[CHAT] 已发送: "
                              f"{prompt[:60]}\033[0m")
                elif mode == "detect":
                    target = text
                    if target == "" and last_det_target:
                        target = last_det_target  # 空Enter, 复用上次
                    if target and frame is not None:
                        last_det_target = target
                        det_prompt = args.det_prompt.format(target)
                        vlm_thread = start_vlm_task(
                            vlm_detect_with_sam,
                            client, model, frame.copy(), det_prompt)
                        vlm_thread.img = frame.copy()
                        vlm_thread.prompt = det_prompt
                        vlm_task_type = "detect"
                        print(f"\033[32m[DET] 目标: {target}"
                              f"  已发送检测请求\033[0m")
                    elif not target:
                        print("\033[93m[DET] 无上次目标, "
                              "请先输入检测目标\033[0m")
            except queue.Empty:
                pass

        # 3) 绘制
        if last_detections:
            draw_detections(display, last_detections, last_masks)
        draw_chat_response(display, last_chat_response)
        draw_overlay(display, {
            "vlm_busy": vlm_busy,
            "lin": lin_v, "ang": ang_v, "dur": dur,
            "last_prompt": last_prompt,
            "det_count": len(last_detections) if last_detections else 0,
            "mask_count": len(last_masks) if last_masks is not None else 0,
            "last_det": last_det_summary,
            "det_target": last_det_target,
        })

        cv2.imshow("Qwen Debug", display)
        key = cv2.waitKey(1) & 0xFF

        # 4) 按键处理 (移动逻辑同 keyboard.py)
        if key == 27:  # Esc
            break

        if key == ord(" "):  # Space
            bot.stop()
            print("  紧急停")
            continue

        if key == ord("i") or key == ord("I"):
            if vlm_busy:
                print("\033[93m[CHAT] VLM 忙, 请稍后\033[0m")
            else:
                input_mode[0] = "chat"
                prev = (f" (空Enter复用: {last_prompt})"
                        if last_prompt else "")
                print(f"\033[36m[CHAT] 请在终端输入 prompt"
                      f"{prev}:\033[0m")
            continue

        if key == ord("o") or key == ord("O"):
            if vlm_busy:
                print("\033[93m[DET] VLM 忙, 请稍后\033[0m")
            else:
                input_mode[0] = "detect"
                prev = (f" (空Enter复用: {last_det_target})"
                        if last_det_target else "")
                print(f"\033[35m[DET] 请在终端输入要检测的目标"
                      f"{prev}:\033[0m")
            continue

        if key == ord("r") or key == ord("R"):
            last_prompt = ""
            last_det_target = ""
            last_chat_response = ""
            print("\033[93m[CHAT] 已清空上次 prompt 和检测目标\033[0m")
            continue

        if key == ord("q") or key == ord("Q"):
            lin_v = min(lin_v + 0.1, 1.0)
            ang_v = min(ang_v + 0.1, 1.5)
            print(f"  调快: 线={lin_v:.2f}  角={ang_v:.2f}")
            continue

        if key == ord("e") or key == ord("E"):
            lin_v = max(lin_v - 0.1, 0.1)
            ang_v = max(ang_v - 0.1, 0.1)
            print(f"  调慢: 线={lin_v:.2f}  角={ang_v:.2f}")
            continue

        action = ACTIONS.get(key)
        if action:
            if move_thread is not None and move_thread.is_alive():
                print(f"  [SKIP] 上一条运动还在执行")
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
                    print(f"  [WARN] 失败: {e}")

    # --- 清理 ---
    running[0] = False
    bot.stop()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
