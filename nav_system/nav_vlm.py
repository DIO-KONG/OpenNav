#!/usr/bin/env python
"""nav_vlm — Qwen VLM 目标检测 + Mobile SAM 分割管理器。

把 keyboard_qwen.py 的 VLM detect 逻辑抽成可复用模块:
  - VlmDetector: 懒加载 OpenAI client + (可选) Mobile SAM;
    detect(img, target) 同步调用, 直接返回结果 (单 client, 无多线程)。
  - 纯 numpy/BGR, 无 ROS 依赖, nav_mock / nav_page 直接 import 即可。

依赖 (openai / mobile_sam / torch) 均为懒导入: 仅当真正用到才 import,
故无 VLM 服务的开发机 import 本模块不会拉起重依赖 (import 期只触 numpy/cv2)。
"""

import base64
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import cv2

from nav_constants import (VLM_DETECTION_PROMPT, MOBILE_SAM_CHECKPOINT_PATH,
                           VLM_PRESENCE_PROMPT)

# 与 keyboard_qwen 一致; 改 nav_constants.VLM_DETECTION_PROMPT 即全局生效。
DETECTION_PROMPT = VLM_DETECTION_PROMPT

from nav_helpers import _presence_is_yes


# ============================================================
#  Mobile SAM predictor (懒加载)
# ============================================================
_sam_predictor = None


def get_sam_predictor():
    """懒加载 SAM predictor。首次调用时加载 vit_t 模型到 CUDA。"""
    global _sam_predictor
    if _sam_predictor is None:
        from mobile_sam import SamPredictor, sam_model_registry
        model_type = "vit_t"
        sam = sam_model_registry[model_type](
            checkpoint=MOBILE_SAM_CHECKPOINT_PATH)
        sam = sam.to("cuda")
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def sam_segment(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """batch bbox -> batch masks (同 keyboard_qwen)。"""
    if len(xyxy) == 0:
        return None
    predictor = get_sam_predictor()
    predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = predictor.predict(
            box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# ============================================================
#  VLM 辅助函数 (适配自 keyboard_qwen, 改为接收 numpy)
# ============================================================
def numpy_to_data_url(img_bgr, quality=85):
    """BGR numpy -> base64 JPEG data URL。"""
    _, buf = cv2.imencode(".jpg", img_bgr,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text):
    """从模型返回文本中提取 JSON (来自 keyboard_qwen)。"""
    import re
    import json
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
        print("\033[93m[VLM] JSON parse error. Raw:\033[0m")
        print(repr(text))
        return None


def qwen_bbox_to_pixel(bbox_2d, w, h):
    """Qwen 归一化坐标 (0-1000) -> 像素坐标 (来自 keyboard_qwen)。"""
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


def vlm_detect_with_presence(detector, img, target, *,
                             require_presence=True,
                             presence_cache=None):
    """纯 VLM 采集: 先问 present (可选 + 缓存), 再 detect, 返回像素 bbox。

    不涉及 3D 投影 / 调试落盘 (那些留在调用方, 因为跨层)。
    返回 (present, vdets, vmasks, bbox_px); 任一项失败为 None:
      - present=False : 画面无目标 (presence=no/空), 调用方应跳过检测
      - vdets=='error': VLM 内部错误 (vmasks 为错误信息串)
      - bbox_px=None  : detect 有结果但无有效 bbox (无效/空 dets)
    presence_cache: list[bool]; 传入则命中 present 时置 True, 后续跳过问询
      (FINAL_ADJUST 等需每帧重问的场景传 None)。
    """
    present = True
    if require_presence:
        if not (presence_cache is not None and presence_cache[0]):
            _pa = detector.ask(img, VLM_PRESENCE_PROMPT)
            present = _presence_is_yes(_pa)
            if present and presence_cache is not None:
                presence_cache[0] = True
    if not present:
        return False, None, None, None

    _vdets, _vmasks, _vtgt = detector.detect(img, target)
    if _vdets == "error":
        return False, _vdets, _vmasks, None
    if not _vdets:
        return True, _vdets, _vmasks, None
    _vb = _vdets[0].get("bbox_2d")
    if not (_vb and len(_vb) == 4):
        return True, _vdets, _vmasks, None
    h, w = img.shape[:2]
    _x1, _y1, _x2, _y2 = qwen_bbox_to_pixel(_vb, w, h)
    return True, _vdets, _vmasks, [_x1, _y1, _x2, _y2]


# [2026-07-29] save_vlm_pair 已移除: 不再保存 VLM pair 文件 (ask/detect 均不再调用)


def vlm_detect(client, model, img_bgr, prompt):
    """以 guided JSON 调 Qwen 做 object detection。"""
    print(f"[VLM] >>> PROMPT:\n{prompt}")
    print(f"[VLM] >>> IMG shape={img_bgr.shape} dtype={img_bgr.dtype}")
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
                {"type": "image_url", "image_url": {"url": image_url}},
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
    content = response.choices[0].message.content
    print(f"[VLM] <<< RAW RESPONSE:\n{content}")
    return extract_json(content)


def vlm_detect_with_sam(client, model, img_bgr, prompt):
    """VLM detect + SAM segment。只取第一个检测结果。"""
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
                print(f"\033[93m[VLM] SAM 分割失败: {se}\033[0m")
    return dets, masks


def vlm_ask(client, model, img_bgr, prompt):
    """纯多模态问答 (不强制 JSON schema), 返回模型原始文本回答。

    用于 h 键自由问答: 用户自设 prompt, 模型自由回答, 文本直接显示到网页。
    """
    image_url = numpy_to_data_url(img_bgr)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
        }],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ============================================================
#  VlmDetector — 同步检测管理器
# ============================================================
class VlmDetector:
    """封装 Qwen VLM 检测 (± SAM), 同步调用 (单 client, 无需多线程)。

    用法:
        det = VlmDetector(vlm_url="http://...:8222/v1")
        res = det.detect(img_bgr, "trash bin")   # 阻塞, 直接返回结果
    res 形态:
        (dets, masks, target)  正常: dets=[{bbox_2d,label,sub_label}], masks 可 None
        ("error", msg, target)   异常
    """

    def __init__(self, vlm_url, vlm_model="",
                 det_prompt=DETECTION_PROMPT, use_sam=True,
                 sam_ckpt=MOBILE_SAM_CHECKPOINT_PATH):
        from openai import OpenAI
        self.client = OpenAI(base_url=vlm_url, api_key="EMPTY", timeout=30.0)
        self.model = vlm_model
        self.det_prompt = det_prompt
        self.use_sam = use_sam
        self.sam_ckpt = sam_ckpt

    def detect(self, img_bgr, target):
        """同步检测, 阻塞直到返回 (含 SAM 首次加载)。

        直接对传入帧做 VLM 检测 (± SAM), 返回
        (dets, masks, target) 正常 或 ("error", msg, target) 异常。
        """
        # 仅替换 "Detect {}" 中的占位符; 模板内 JSON 示例的 {} 必须原样保留,
        # 故用 replace 而非 str.format (后者会把 JSON 大括号当占位符解析 -> KeyError)。
        prompt = self.det_prompt.replace("{}", target)
        try:
            if self.use_sam:
                dets, masks = vlm_detect_with_sam(
                    self.client, self.model, img_bgr, prompt)
            else:
                dets = vlm_detect(self.client, self.model, img_bgr, prompt)
                masks = None
            ans = repr(dets) if dets else "no detection"
            return (dets, masks, target)
        except Exception as e:
            return ("error", str(e), target)

    def ask(self, img_bgr, prompt):
        """同步自由问答 (h 键), 返回模型文本 (或 'error: ...')。

        与 detect 不同: 不强制 guided-JSON schema, 直接多模态 chat 拿原始文本回答,
        用于把 VLM 的回答显示到网页 (不触发任何移动/导航)。
        """
        try:
            ans = vlm_ask(self.client, self.model, img_bgr, prompt)
            return ans
        except Exception as e:
            msg = f"error: {e}"
            print(f"\033[91m[VLM-H] {msg}\033[0m")
            return msg


@dataclass(frozen=True)
class VlmJob:
    """提交给单线程 VLM worker 的不可变任务描述。"""

    request_id: int
    kind: str
    image: np.ndarray
    target: Optional[str] = None
    prompt: Optional[str] = None
    submit_step: int = 0
    submit_state: str = ""
    epoch: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VlmResult:
    """VLM worker 的结果；导航状态只能由主线程消费并更新。"""

    request_id: int
    kind: str
    answer: Optional[str]
    dets: Any
    masks: Any
    error: Optional[str]
    submit_step: int
    submit_state: str
    epoch: int
    started_at: float
    finished_at: float
    context: Dict[str, Any]


class AsyncVlmWorker:
    """串行执行所有自动 VLM 请求，主循环仅 submit/poll。

    request/result 队列都限制为 1；从提交到结果被 poll 期间保持 busy，
    防止结果尚未消费时又启动下一次请求。VlmDetector 与 Mobile SAM 仅由
    本线程访问。
    """

    _KINDS = {"auto_detect", "reloc_presence", "reloc_detect",
              "direction", "open_yesno"}

    def __init__(self, detector: VlmDetector):
        self.detector = detector
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=1)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._active_request_id = None
        self._thread = None

    def start(self):
        """启动 worker（幂等）。"""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._run, name="vlm-worker", daemon=True)
            self._thread.start()

    def submit(self, job: VlmJob) -> bool:
        """空闲时提交一个任务；已有 in-flight/result 时返回 False。"""
        if not isinstance(job, VlmJob) or job.kind not in self._KINDS:
            return False
        if self._stop_evt.is_set():
            return False
        with self._lock:
            if self._active_request_id is not None:
                return False
            self._active_request_id = int(job.request_id)
        try:
            self._requests.put_nowait(job)
            return True
        except queue.Full:
            with self._lock:
                self._active_request_id = None
            return False

    def poll(self) -> Optional[VlmResult]:
        """非阻塞取出一个完成结果。"""
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            if self._active_request_id == result.request_id:
                self._active_request_id = None
        return result

    def busy(self) -> bool:
        with self._lock:
            return self._active_request_id is not None

    def stop(self, timeout=1.0):
        """请求退出并有限等待；不会等待最长 HTTP timeout。"""
        self._stop_evt.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))

    @staticmethod
    def _presence_yes(answer) -> bool:
        if not isinstance(answer, str):
            return False
        text = answer.strip().lower()
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False
        return "yes" in text

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                job = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break

            started_at = time.time()
            answer = None
            dets = None
            masks = None
            error = None
            try:
                if job.kind == "direction":
                    answer = self.detector.ask(job.image, job.prompt or "")
                    if answer is None or str(answer).startswith("error"):
                        error = str(answer or "empty answer")
                elif job.kind == "reloc_presence":
                    answer = self.detector.ask(
                        job.image, job.prompt or "")
                    if answer is None or str(answer).startswith("error"):
                        error = str(answer or "empty answer")
                elif job.kind == "reloc_detect":
                    dets, masks, _ = self.detector.detect(
                        job.image, job.target or "")
                    if dets == "error":
                        error = str(masks)
                        dets = None
                        masks = None
                elif job.kind == "open_yesno":
                    # nav_open: 自由问答 yes/no, 不强制 JSON, 直接返回文本
                    answer = self.detector.ask(
                        job.image, job.prompt or "")
                    if answer is None or str(answer).startswith("error"):
                        error = str(answer or "empty answer")
                else:
                    need_presence = bool(
                        job.context.get("need_presence", True))
                    if need_presence:
                        answer = self.detector.ask(
                            job.image, job.prompt or "")
                        if answer is None or str(answer).startswith("error"):
                            error = str(answer or "empty answer")
                    else:
                        answer = "cached(yes)"
                    if error is None and (
                            self._presence_yes(answer) or not need_presence):
                        dets, masks, _ = self.detector.detect(
                            job.image, job.target or "")
                        if dets == "error":
                            error = str(masks)
                            dets = None
                            masks = None
            except Exception as exc:
                error = repr(exc)

            result = VlmResult(
                request_id=job.request_id,
                kind=job.kind,
                answer=answer,
                dets=dets,
                masks=masks,
                error=error,
                submit_step=job.submit_step,
                submit_state=job.submit_state,
                epoch=job.epoch,
                started_at=started_at,
                finished_at=time.time(),
                context=job.context,
            )
            try:
                self._results.put_nowait(result)
            except queue.Full:
                try:
                    self._results.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._results.put_nowait(result)
                except queue.Full:
                    with self._lock:
                        if self._active_request_id == job.request_id:
                            self._active_request_id = None


# ============================================================
#  绘制 (供 nav_page.draw_debug_overlay 复用)
# ============================================================
def draw_detections(frame, detections, masks=None):
    """在帧上绘制 bbox (绿色框 + 标签) + SAM mask (半透明叠加)。"""
    if not detections:
        return frame
    h, w = frame.shape[:2]
    # --- SAM mask 半透明叠加 ---
    if masks is not None:
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
    return frame
