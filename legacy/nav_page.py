import time
import cv2
import numpy as np
from nav_path import ROBOT_RADIUS  # 顶视地图 footprint 半径 (viz 显示用)
from nav_constants import MAP_VIEW_SCALE
import threading
from flask import Flask, request, jsonify, Response
from nav_vlm import draw_detections

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NAV Debug</title>
<style>
  body { background:#111; color:#0f0; font-family:monospace; margin:0; padding:20px; }
  h2 { margin:0 0 8px 0; }
  .container { max-width:1200px; margin:0 auto; }
  .streams { display:flex; gap:12px; flex-wrap:wrap; }
  .stream-box { flex:1; min-width:420px; }
  .stream-box img { width:100%; border:2px solid #333; display:block; }
  .keys { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }
  .key { padding:6px 16px; background:#222; border:1px solid #444;
         border-radius:4px; font-size:14px; transition:background .1s; }
  .key.active { background:#0a3; color:#fff; }
  .status { margin-top:8px; font-size:13px; color:#888; }
</style>
</head>
<body>
<div class="container">
  <h2>NAV Debug</h2>
  <div class="streams">
    <div class="stream-box">
      <h3>RGB View</h3>
      <img src="/stream" id="stream" />
    </div>
    <div class="stream-box">
      <h3>Top-Down Map</h3>
      <img src="/map_stream" id="map_stream" />
    </div>
  </div>
  <div class="keys">
    <span class="key" id="key-j">J &mdash; VLM-L</span>
    <span class="key" id="key-k">K &mdash; VLM-F</span>
    <span class="key" id="key-l">L &mdash; VLM-R</span>
    <span class="key" id="key-o">O &mdash; DETECT</span>
    <span class="key" id="key-h">H &mdash; ASK</span>
    <span class="key" id="key-f">F &mdash; FRONTIER</span>
    <span class="key" id="key-space">SPACE &mdash; ESTOP</span>
  </div>
  <div class="detect-row" style="margin-top:10px; display:flex; gap:8px; align-items:center;">
    <label style="color:#0f0; font-size:14px;">VLM Detect:</label>
    <input id="detect-target" type="text" placeholder="检测目标 (如 trash bin)"
           style="padding:6px 10px; background:#222; border:1px solid #444; color:#0f0; border-radius:4px; font-family:monospace; min-width:260px;" />
    <button id="detect-btn" style="padding:6px 16px; background:#0a3; color:#fff; border:none; border-radius:4px; font-size:14px; cursor:pointer;">Detect (O)</button>
    <span id="detect-status" style="color:#8f8; font-size:13px;"></span>
  </div>
  <div class="status" id="status">Click page: J/K/L/O=VLM/DET, H=ASK, F=frontier, SPACE=estop</div>
  <div class="hqa-row" style="margin-top:12px; display:flex; gap:8px; align-items:center;">
    <button id="h-btn" style="padding:6px 16px; background:#06c; color:#fff; border:none; border-radius:4px; font-size:14px; cursor:pointer;">Ask VLM (H)</button>
    <span id="h-status" style="color:#8f8; font-size:13px;"></span>
  </div>
  <div class="hqa-answer" id="h-answer" style="margin-top:8px; padding:10px 12px; background:#1a1a1a; border:1px solid #333; border-radius:4px; color:#0f0; font-size:14px; white-space:pre-wrap; min-height:20px;">
    <span style="color:#666;">VLM 回答将显示在这里 (按 H 或点 Ask VLM)...</span>
  </div>
</div>
<script>
// 使用相对路径, 自动跟随页面所在的 host:port,
// 无论从本机 (localhost) 还是其他设备打开都能正确发送

function sendKey(key) {
  fetch('/key', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:key})
  }).then(function(r){return r.json();}).then(function(d){
    var el = document.getElementById('key-'+key);
    if (el) { el.classList.add('active');
      setTimeout(function(){el.classList.remove('active');},300); }
    var st = document.getElementById('status');
    if (d.action === 'stop') {
      st.textContent = 'ESTOP: ' + (d.halt ? 'HALTED' : 'RESUMED') +
        ' at ' + new Date().toLocaleTimeString();
    } else if (d.key) {
      st.textContent = 'Sent: ' + d.key.toUpperCase() +
        ' at ' + new Date().toLocaleTimeString();
    }
  });
}

var MOCK_KEYS = ['j','k','l','o','h','f'];

document.addEventListener('keydown', function(e) {
  if (e.key === ' ') {
    sendKey('stop');
    e.preventDefault();
    return;
  }
  var key = e.key.toLowerCase();
  if (MOCK_KEYS.indexOf(key) >= 0) {
    sendKey(key);
    e.preventDefault();
  }
});

// VLM 检测: 取输入框目标词, 随 o 键一并发给后端 (target 字段)
function sendDetect() {
  var inp = document.getElementById('detect-target');
  var tgt = inp ? inp.value.trim() : '';
  fetch('/key', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:'o', target: tgt})
  }).then(function(r){return r.json();}).then(function(d){
    var ds = document.getElementById('detect-status');
    if (ds) {
      ds.textContent = 'Detect' + (tgt ? ' "' + tgt + '"' : ' (默认)') +
        ' at ' + new Date().toLocaleTimeString();
    }
    var el = document.getElementById('key-o');
    if (el) { el.classList.add('active');
      setTimeout(function(){el.classList.remove('active');},300); }
  });
}
document.getElementById('detect-btn').addEventListener('click', sendDetect);
document.getElementById('detect-target').addEventListener('keydown', function(e){
  if (e.key === 'Enter') { sendDetect(); e.preventDefault(); }
});

// h 键自由问答: 发 h 给后端, 答案由 /vlm_h 轮询显示
function sendH() {
  fetch('/key', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:'h'})
  }).then(function(r){return r.json();}).then(function(d){
    var hs = document.getElementById('h-status');
    if (hs) hs.textContent = '提问 at ' + new Date().toLocaleTimeString();
    var el = document.getElementById('key-h');
    if (el) { el.classList.add('active');
      setTimeout(function(){el.classList.remove('active');},300); }
  });
}
document.getElementById('h-btn').addEventListener('click', sendH);

// 轮询 h 键 VLM 问答答案并显示
function pollHAnswer() {
  fetch('/vlm_h').then(function(r){return r.json();}).then(function(d){
    var box = document.getElementById('h-answer');
    if (box) {
      if (d.answer) {
        box.textContent = d.answer;
      } else {
        box.innerHTML = '<span style="color:#666;">VLM 回答将显示在这里 (按 H 或点 Ask VLM)...</span>';
      }
    }
  }).catch(function(){});
}
setInterval(pollHAnswer, 1000);
pollHAnswer();
</script>
</body>
</html>"""

# 模块级共享对象: 由 nav_mock.main() 在创建 MockState 实例后注入
# (nav_page.mock = mock), 避免 nav_page 反向 import nav_mock 造成循环依赖。
# 路由函数在请求时才访问它, 故 import 期不会触发 NameError。
mock = None

app = Flask(__name__)


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/stream")
def stream():
    def generate():
        while True:
            frame = mock.get_frame()
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
            time.sleep(0.08)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/map_stream")
def map_stream():
    def generate():
        while True:
            frame = mock.get_map_frame()
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
            time.sleep(0.1)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/key", methods=["POST"])
def handle_key():
    data = request.get_json(force=True)
    key = data.get("key", "")
    if key in ("j", "k", "l"):
        mock.set_vlm_key(key)
        return jsonify({"status": "ok", "action": "vlm", "key": key})
    elif key == "o":
        target = data.get("target")
        if target:
            mock.set_detect_target(target)
        mock.set_det_key(key)
        return jsonify({"status": "ok", "action": "detect",
                        "key": key, "target": target})
    elif key == "h":
        mock.set_h_key(key)
        return jsonify({"status": "ok", "action": "h_ask", "key": key})
    elif key == "f":
        mock.set_frontier_request()
        return jsonify({"status": "ok", "action": "frontier", "key": key})
    elif key == "stop":
        new_halt = not mock.get_halt()
        mock.set_halt(new_halt)
        print(f"[nav_2d] ESTOP toggled -> {'HALTED' if new_halt else 'RESUMED'}")
        return jsonify({"status": "ok", "action": "stop", "halt": new_halt})
    return jsonify({"status": "ignored", "key": key}), 400


@app.route("/vlm_h")
def vlm_h_answer():
    """返回 h 键 VLM 自由问答的最新文本答案 (网页轮询)。"""
    return jsonify({"answer": mock.get_vlm_h_answer() if mock else None})


def start_flask(port=5001):
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               threaded=True, debug=False,
                               use_reloader=False),
        daemon=True)
    t.start()
    print(f"[nav_2d] Debug Web: http://localhost:{port} "
          f"(从其他设备访问请改用本机 IP:{port})")


# ============================================================
#  位姿平滑器 (防画龙): yaw EMA + 角速度输出 (死区/slew/EMA)
#  (从 nav_mock 迁移: viz 本来就是画在网页上的)
# ============================================================
class YawSmoother:
    """yaw 指数平滑 (EMA), 环绕感知。

    只平滑 yaw (画龙主因), 不平滑位置 (避免地图/patrol 目标对不上)。
    用归一化角度差做 EMA, 避免跨 +/-pi 时跳变。
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._yaw = None
        self._lock = threading.Lock()

    def update(self, yaw):
        with self._lock:
            y = float(yaw)
            if self._yaw is None:
                self._yaw = y
                return self._yaw
            d = y - self._yaw
            d = (d + np.pi) % (2 * np.pi) - np.pi   # 归一化到 [-pi, pi]
            self._yaw += self.alpha * d
            self._yaw = (self._yaw + np.pi) % (2 * np.pi) - np.pi
            return self._yaw

    def reset(self):
        with self._lock:
            self._yaw = None


class OutputSmoother:
    """角速度输出平滑: 死区 -> slew rate -> EMA。

    消除 yaw 抖动/跳变造成的角速度脉冲, 防画龙:
      1. 死区: 微小角速度直接归零 (不转向, 直走)
      2. slew: 限制每帧角速度变化, 阻断 yaw 阶跃跳变的大脉冲
      3. EMA: 低通, 平滑剩余高频抖动
    """
    def __init__(self, alpha=0.4, deadband=0.087, slew=0.08):
        self.alpha = alpha
        self.deadband = deadband
        self.slew = slew
        self._omega = 0.0
        self._lock = threading.Lock()

    def update(self, omega, deadband=None):
        with self._lock:
            w = float(omega)
            # 1. 死区 (D: 转弯时可传 deadband=0.0 跳过, 避免小 omega 被吃掉)
            db = self.deadband if deadband is None else deadband
            if abs(w) < db:
                w = 0.0
            # 2. slew rate (相对上一帧输出)
            delta = w - self._omega
            if abs(delta) > self.slew:
                w = self._omega + np.sign(delta) * self.slew
            # 3. EMA
            self._omega += self.alpha * (w - self._omega)
            return self._omega

    def reset(self):
        with self._lock:
            self._omega = 0.0

# ============================================================
#  Debug Overlay / 顶视地图可视化
# ============================================================
def draw_debug_overlay(img, info):
    """在 BGR 帧上绘制半透明 debug 信息 + mock 区域框。"""
    if img is None:
        img = np.zeros((240, 480, 3), dtype=np.uint8)

    frame = img.copy()
    h, w = frame.shape[:2]

    # --- 半透明黑色背景条 ---
    bar_h = min(h, 18 + len(info.get("lines", [])) * 16)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # --- 文字 ---
    y = 18
    for line in info.get("lines", []):
        if isinstance(line, tuple):
            text, color = line
        else:
            text, color = line, (0, 255, 0)
        cv2.putText(frame, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    color, 1, cv2.LINE_AA)
        y += 16

    # --- mock 区域框 ---
    for rect in info.get("rects", []):
        x1, y1, x2, y2 = rect["bbox"]
        color = rect["color"]
        label = rect["label"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)

    # --- VLM 检测框 + SAM mask (o 键触发) ---
    _vlm_dets = info.get("vlm_dets")
    _vlm_masks = info.get("vlm_masks")
    if _vlm_dets:
        frame = draw_detections(frame, _vlm_dets, _vlm_masks)

    # --- 底部 key 提示 ---
    hint = "[j]VLM-L [k]VLM-F [l]VLM-R [o]DET [h]ASK [f]FRONTIER | [SPACE]ESTOP"
    cv2.putText(frame, hint, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1, cv2.LINE_AA)

    # --- 朝向罗盘 (右上角) ---
    rp = info.get("robot_pose")
    if rp is not None and frame.shape[1] > 100:
        using_odom = info.get("using_odom", False)
        _compass_col = (0, 165, 255) if using_odom else (0, 255, 255)
        cx_c, cy_c = frame.shape[1] - 50, 50
        R = 36
        cv2.circle(frame, (cx_c, cy_c), R, (90, 90, 90), 2)
        # 与地图 w2p 同约定: 翻 X; Z→下(前)=从天到地视角
        # 前向世界向量 (sin yaw, cos yaw) -> 像素 (-sin, +cos)
        ang = float(rp[2])
        dx = -np.sin(ang)
        dy = np.cos(ang)
        ex, ey = int(cx_c + dx * (R - 4)), int(cy_c + dy * (R - 4))
        cv2.line(frame, (cx_c, cy_c), (ex, ey), _compass_col, 3)
        cv2.circle(frame, (cx_c, cy_c), 3, _compass_col, -1)
        cv2.putText(frame, f"{np.degrees(rp[2]):.0f}°",
                    (cx_c - 14, cy_c + R + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _compass_col, 1, cv2.LINE_AA)

    return frame

def draw_map_view(info):
    """绘制 2D 顶视图 (X-Z 平面, Y 轴向下为高度)。

    info 字段:
        obstacle_points  (N,3) 世界坐标点云, 或 None
        robot_pose       (x, z, yaw) 或 None
        patrol_target    (x, z) 或 None
        final_target     (x, z) 或 None
        lookahead        (x, z) 纯追踪伪目标点, 或 None
        path             [(x,z), ...] 或 None
        path_idx         当前 waypoint index
        d_tgt            到目标点的距离 (m), 或 None
        d_robot          到最近障碍的距离 (m), 或 None
        state            NavState
        nav_y            当前高度 Y
        n_map_points     总地图点数
    """
    MAP_W, MAP_H = 480, 480
    # 基准 80 px/m、±3 m; 乘/除 MAP_VIEW_SCALE 联动 (见 nav_constants)
    #   DPI 变小 → 同画布看更远; RANGE 变大 → 裁剪方盒跟上视野
    #   DPI 取 int: 传给 cv2 的像素坐标必须是整型 (w2p/网格/贴图均显式 int)
    _view_scale = float(MAP_VIEW_SCALE) if MAP_VIEW_SCALE else 1.0
    if _view_scale <= 0:
        _view_scale = 1.0
    DPI = max(1, int(80.0 / _view_scale))    # px/m (map DPI)
    RANGE = 3.0 * _view_scale                  # 半径 (m), 超出截断

    canvas = np.full((MAP_H, MAP_W, 3), 18, dtype=np.uint8)

    # 中心 = 机器人位置 (或原点)
    cx, cz = 0.0, 0.0
    robot_pose = info.get("robot_pose")
    if robot_pose is not None:
        cx, cz = robot_pose[0], robot_pose[1]
    # 像素中心
    pcx, pcy = MAP_W // 2, MAP_H // 2

    def w2p(wx, wz):
        """世界 (x,z) → 像素 (px, py)。  翻 X 使地图左右与实际点云一致; Z→下(前)=从天到地视角。"""
        px = int(pcx - (wx - cx) * DPI)
        py = int(pcy + (wz - cz) * DPI)
        return px, py

    # --- 网格 (随 RANGE 自适应条数; 不再写死 ±3) ---
    _n_grid = max(1, int(np.ceil(RANGE)))
    for i in range(-_n_grid, _n_grid + 1):
        gx = int(pcx + i * DPI)
        gy = int(pcy + i * DPI)
        cv2.line(canvas, (gx, 0), (gx, MAP_H), (35, 35, 35), 1)
        cv2.line(canvas, (0, gy), (MAP_W, gy), (35, 35, 35), 1)
    # 中心十字
    cv2.line(canvas, (pcx - 8, pcy), (pcx + 8, pcy), (60, 60, 60), 1)
    cv2.line(canvas, (pcx, pcy - 8), (pcx, pcy + 8), (60, 60, 60), 1)

    # --- 障碍物点云 --- (必须与 w2p 同约定, 否则与巡逻星/路径/机器人错位)
    obs = info.get("obstacle_points")
    if obs is not None and len(obs) > 0:
        xs = obs[:, 0]
        zs = obs[:, 2]
        # 只画范围内的
        mask = (np.abs(xs - cx) < RANGE) & (np.abs(zs - cz) < RANGE)
        xs = xs[mask]
        zs = zs[mask]
        # 向量化点绘制: 原逐点 Python for 循环 (canvas[py,px]=color) 在障碍点达数十万~
        # 上百万时耗时恒定 ~1.4s/圈, 是主循环 draw 段瓶颈。改用 numpy 批量算像素坐标 +
        # fancy-index 一次性着色, 输出像素与原实现完全一致 (int() 与 astype(int) 均朝零截断)。
        px = (pcx - (xs - cx) * DPI).astype(np.int32)
        py = (pcy + (zs - cz) * DPI).astype(np.int32)
        inb = (px >= 0) & (px < MAP_W) & (py >= 0) & (py < MAP_H)
        canvas[py[inb], px[inb]] = (140, 140, 80)  # 淡黄色点
        cv2.putText(canvas, f"obs:{len(xs)}", (MAP_W - 90, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (140, 140, 80), 1, cv2.LINE_AA)

    # --- 记忆 walked 高斯区域 (半透明填充 disk/square, 与 frontier_grid 形状一致) ---
    mem_gaussians = info.get("memory_gaussians")
    n_mem_drawn = 0
    if mem_gaussians is not None and len(mem_gaussians) > 0:
        mem_shape = info.get("memory_shape", "disk")
        mem_kappa = info.get("memory_kappa", 1.0)
        mem_alpha = float(info.get("memory_alpha", 0.30))
        overlay = np.zeros_like(canvas)
        for cx_w, cz_w, sigma_w in mem_gaussians:
            if abs(cx_w - cx) > RANGE or abs(cz_w - cz) > RANGE:
                continue
            px, py = w2p(cx_w, cz_w)
            radius_px = int(mem_kappa * sigma_w * DPI)
            if radius_px <= 0:
                continue
            if mem_shape == "square":
                cv2.rectangle(overlay, (px - radius_px, py - radius_px),
                              (px + radius_px, py + radius_px), (0, 255, 255), -1)
            else:
                cv2.circle(overlay, (px, py), radius_px, (0, 255, 255), -1)
            n_mem_drawn += 1
        canvas = cv2.addWeighted(canvas, 1.0, overlay, mem_alpha, 0)
    # (mem 数量文本已从 HUD 移除, 减少信息过载; walked 区域仍可视化)

    # --- 栅格 nearest-frontier (品红菱形, 仅 viz; 不参与导航) ---
    frontier_xz = info.get("frontier_xz")
    if frontier_xz is not None:
        try:
            fx, fz = float(frontier_xz[0]), float(frontier_xz[1])
            if abs(fx - cx) < RANGE and abs(fz - cz) < RANGE:
                p = w2p(fx, fz)
                cv2.drawMarker(canvas, p, (255, 255, 255),
                               cv2.MARKER_DIAMOND, 22, 3)
                cv2.drawMarker(canvas, p, (255, 0, 255),
                               cv2.MARKER_DIAMOND, 18, 2)
                cv2.putText(canvas, "F", (p[0] + 10, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 0, 255), 1, cv2.LINE_AA)
                if robot_pose is not None:
                    rp = w2p(robot_pose[0], robot_pose[1])
                    cv2.line(canvas, rp, p, (180, 0, 180), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "F:out", (MAP_W - 90, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (255, 0, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    # --- 栅格 debug 覆盖层 (OCC=红, WALKED=青, alpha=0.30) ---
    # 坐标系对应关系:
    #   Grid 数组:   axis0=ix(X), axis1=iz(Z).  世界: X=origin_x+ix*res, Z=origin_z+iz*res.
    #   Canvas:      np[row, col] ← row=py(Z), col=px(X).  w2p: px=pcx-(X-cx)*S, py=pcy+(Z-cz)*S.
    # 因此:  Grid[ix,iz] → Canvas[py(Z), px(X)].
    # 建图时直接按 Canvas 轴序: gc_row = iz (Z), gc_col = ix (X).
    # 然后 X 轴翻转 (w2p 中 X↑→px↓), 粘贴时 flip LR.
    grid_occ_inf = info.get("grid_occ_inf")
    grid_walked = info.get("grid_walked")
    grid_n = info.get("grid_n", 0)
    if grid_occ_inf is not None and grid_walked is not None and grid_n > 0:
        try:
            grid_origin_x = float(info.get("grid_origin_x", 0.0))
            grid_origin_z = float(info.get("grid_origin_z", 0.0))
            grid_res = float(info.get("grid_res", 0.10))
            # Build colour grid with axes (iz=Row, ix=Col) to match canvas (row=Z, col=X).
            # occ_inf/walked have shape (n,n) indexed [ix, iz]; use .T to swap to [iz, ix].
            gc = np.zeros((grid_n, grid_n, 3), dtype=np.uint8)   # (iz=Row, ix=Col, RGB)
            occ_b = np.asarray(grid_occ_inf, dtype=bool)
            walk_b = np.asarray(grid_walked, dtype=bool)
            gc[occ_b.T] = (0, 0, 255)                 # OCC   = red   [iz,ix] ← .T
            # Do NOT paint WALKED over OCC — cyan-over-red hid true obstacles and
            # made robot-in-OCC look like free corridor (false free for debug).
            walked_vis = walk_b & ~occ_b
            gc[walked_vis.T] = (255, 255, 0)           # WALKED∩FREE = cyan
            # Flip columns so ix=0 (min X) goes to image RIGHT (matches w2p X flip).
            gc_img = cv2.flip(gc, 1)
            # 用世界 AABB 连续投影定贴图矩形, 再 NEAREST resize。
            # 旧实现 cell_px=int(res*DPI) 再 n*cell_px 累加: 当 res*DPI 非整数
            # (MAP_VIEW_SCALE 使 DPI 非 80/整数) 时每格截断误差, 整窗漂移数十字节。
            x_min_w = grid_origin_x
            z_min_w = grid_origin_z
            x_max_w = grid_origin_x + grid_n * grid_res
            z_max_w = grid_origin_z + grid_n * grid_res
            # w2p: +X→左, +Z→下 ⇒ 画布左= maxX, 右=minX, 上=minZ, 下=maxZ
            x_left_f = pcx - (x_max_w - cx) * DPI
            x_right_f = pcx - (x_min_w - cx) * DPI
            y_top_f = pcy + (z_min_w - cz) * DPI
            y_bot_f = pcy + (z_max_w - cz) * DPI
            x0 = int(np.floor(x_left_f))
            y0 = int(np.floor(y_top_f))
            x1 = int(np.ceil(x_right_f))
            y1 = int(np.ceil(y_bot_f))
            gw = max(1, x1 - x0)
            gh = max(1, y1 - y0)
            gc_expanded = cv2.resize(
                gc_img, (gw, gh), interpolation=cv2.INTER_NEAREST)
            # Clamp 到画布后 paste
            x0c = max(0, x0); x1c = min(MAP_W, x1)
            y0c = max(0, y0); y1c = min(MAP_H, y1)
            gx0 = x0c - x0; gx1 = gw - (x1 - x1c)
            gy0 = y0c - y0; gy1 = gh - (y1 - y1c)
            if x0c < x1c and y0c < y1c:
                patch = gc_expanded[gy0:gy1, gx0:gx1]
                roi = canvas[y0c:y1c, x0c:x1c]
                canvas[y0c:y1c, x0c:x1c] = (
                    patch.astype(np.float32) * 0.30 +
                    roi.astype(np.float32) * 0.70
                ).astype(np.uint8)
        except Exception:
            pass

    # --- 路径 ---
    path = info.get("path")
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            p1 = w2p(path[i][0], path[i][1])
            p2 = w2p(path[i + 1][0], path[i + 1][1])
            color = (200, 200, 0)  # 青色路径线
            cv2.line(canvas, p1, p2, color, 2)
        # 已走过的路点高亮
        path_idx = info.get("path_idx", 0)
        for i in range(min(path_idx, len(path))):
            p = w2p(path[i][0], path[i][1])
            cv2.circle(canvas, p, 3, (0, 180, 180), -1)

    # --- patrol 目标 (原检测点, 淡化黄星) + 实际导航终点 (青绿实心圆) ---
    patrol = info.get("patrol_target")
    patrol_nav = info.get("patrol_nav")
    if patrol is not None:
        p = w2p(patrol[0], patrol[1])
        cv2.drawMarker(canvas, p, (0, 200, 200),
                       cv2.MARKER_STAR, 12, 1)
    if patrol_nav is not None:
        p = w2p(patrol_nav[0], patrol_nav[1])
        cv2.circle(canvas, p, 7, (0, 255, 170), -1)
        cv2.circle(canvas, p, 9, (255, 255, 255), 1)
        cv2.putText(canvas, "nav", (p[0] + 10, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 170), 1, cv2.LINE_AA)

    # --- final 目标 (红星) + 实际导航终点 ---
    final = info.get("final_target")
    final_nav = info.get("final_nav")
    if final is not None:
        p = w2p(final[0], final[1])
        cv2.drawMarker(canvas, p, (0, 0, 255),
                       cv2.MARKER_STAR, 14, 2)
    if final_nav is not None:
        p = w2p(final_nav[0], final_nav[1])
        cv2.circle(canvas, p, 7, (0, 255, 170), -1)
        cv2.circle(canvas, p, 9, (255, 255, 255), 1)
        cv2.putText(canvas, "nav", (p[0] + 10, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 170), 1, cv2.LINE_AA)

    # --- 纯追踪 lookahead 伪目标点 (品红叉) ---
    lookahead = info.get("lookahead")
    if robot_pose is not None and lookahead is not None:
        p = w2p(lookahead[0], lookahead[1])
        rp = w2p(robot_pose[0], robot_pose[1])
        cv2.drawMarker(canvas, p, (255, 0, 255),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.line(canvas, rp, p, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "look", (p[0] + 8, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 0, 255), 1, cv2.LINE_AA)

    # --- 机器人位置 + 朝向 ---
    # 使用 odom 回退位姿时染成橙色, 否则绿色 (SLAM 位姿)。
    using_odom = info.get("using_odom", False)
    _robot_col = (0, 165, 255) if using_odom else (0, 255, 0)
    _robot_edge = (0, 90, 180) if using_odom else (0, 80, 0)
    _robot_foot = (0, 165, 255) if using_odom else (0, 200, 0)
    if robot_pose is not None:
        p = w2p(robot_pose[0], robot_pose[1])
        # 与 w2p 同约定: 翻 X 使地图左右与实际点云一致; Z→下(前)=从天到地视角
        # yaw=0 指向屏幕下(前); 前向世界向量 (sin yaw, cos yaw) -> 像素 (-sin, +cos)
        yaw = robot_pose[2]
        size = 18
        dx = -np.sin(yaw) * size
        dy = np.cos(yaw) * size
        pts = np.array([
            [p[0] + int(dx), p[1] + int(dy)],
            [p[0] + int(-dy * 0.5), p[1] + int(dx * 0.5)],
            [p[0] + int(dy * 0.5), p[1] + int(-dx * 0.5)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], _robot_col)
        cv2.polylines(canvas, [pts], True, _robot_edge, 2)
        cv2.circle(canvas, p, 4, (255, 255, 255), -1)
        cv2.circle(canvas, p, 4, _robot_edge, 1)
        # 机器人"体积" footprint: 半径 ROBOT_RADIUS 的圆 (避障膨胀同此值)
        foot_px = int(ROBOT_RADIUS * DPI)
        cv2.circle(canvas, (int(p[0]), int(p[1])), foot_px, _robot_foot, 1)
        cv2.putText(canvas, f"r={ROBOT_RADIUS:.2f}",
                    (int(p[0]) + foot_px + 3, int(p[1]) - foot_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    _robot_foot, 1, cv2.LINE_AA)

    # --- HUD ---
    state = info.get("state")
    if state is None:
        state_str = "N/A"
    elif hasattr(state, "name"):
        state_str = state.name
    else:
        state_str = str(state)
    nav_y = info.get("nav_y")
    nav_y_str = f"{nav_y:.2f}" if nav_y is not None else "N/A"
    n_pts = info.get("n_map_points", 0)
    look = info.get("lookahead")
    look_str = (f"({look[0]:.2f},{look[1]:.2f})"
                if look is not None else "none")
    d_tgt = info.get("d_tgt")
    d_robot = info.get("d_robot")
    path_idx = info.get("path_idx", 0)
    path = info.get("path")
    path_str = (f"{path_idx}/{len(path)}"
                if path is not None else "none")
    d_tgt_str = f"{d_tgt:.2f}" if d_tgt is not None else "N/A"
    d_robot_str = f"{d_robot:.2f}" if d_robot is not None else "N/A"

    # 当前活跃目标来源 + 坐标 (frontier / vlm_dir / vlm_det)
    _goal_src = info.get("goal_source")
    _goal_pt = info.get("final_target") or info.get("patrol_target")
    if _goal_pt is not None:
        goal_str = (f"{_goal_src or '?'} "
                    f"({_goal_pt[0]:.2f},{_goal_pt[1]:.2f})")
    else:
        goal_str = "none"

    fr = info.get("frontier_xz")
    fr_dist = info.get("frontier_dist_m")
    fr_ms = info.get("frontier_ms")
    if fr is not None:
        fr_str = f"({float(fr[0]):.2f},{float(fr[1]):.2f})"
        if fr_dist is not None:
            fr_str += f" d={float(fr_dist):.2f}m"
        if fr_ms is not None:
            fr_str += f" {float(fr_ms):.1f}ms"
    else:
        fr_str = "none"

    hud_lines = [
        f"state={state_str}",
        f"robot=({cx:.2f},{cz:.2f}) Y={nav_y_str}"
        + (" [ODOM]" if using_odom else " [SLAM]"),
        f"map_pts={n_pts}",
        f"goal={goal_str}",
        f"front={fr_str}",
        f"look={look_str}",
        f"d_tgt={d_tgt_str} d_obs={d_robot_str}",
        f"path={path_str}",
        f"range=+/-{RANGE:.0f}m  dpi={DPI}px/m",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(canvas, line, (8, 16 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # 图例 (右下)
    leg_y = MAP_H - 152
    cv2.drawMarker(canvas, (MAP_W - 92, leg_y),
                   (0, 255, 255), cv2.MARKER_STAR, 16, 2)
    cv2.putText(canvas, "patrol", (MAP_W - 74, leg_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (MAP_W - 92, leg_y + 24),
                   (0, 0, 255), cv2.MARKER_STAR, 16, 2)
    cv2.putText(canvas, "final", (MAP_W - 74, leg_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (MAP_W - 84, leg_y + 47), 8, (0, 200, 0), 1)
    cv2.putText(canvas, f"foot r={ROBOT_RADIUS:.2f}",
                (MAP_W - 70, leg_y + 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 200, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (MAP_W - 92, leg_y + 64),
                   (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.putText(canvas, "look", (MAP_W - 74, leg_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 0, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (MAP_W - 92, leg_y + 86),
                   (255, 0, 255), cv2.MARKER_DIAMOND, 12, 2)
    cv2.putText(canvas, "frontier", (MAP_W - 74, leg_y + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (MAP_W - 84, leg_y + 104), 3, (255, 255, 0), -1)
    cv2.putText(canvas, "mem", (MAP_W - 74, leg_y + 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (MAP_W - 92, leg_y + 122),
                   (0, 165, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)
    cv2.putText(canvas, "odom", (MAP_W - 74, leg_y + 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 165, 255), 1, cv2.LINE_AA)

    return canvas
