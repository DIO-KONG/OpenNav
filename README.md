# OpenNav

OpenNav is a ROS-based indoor mobile-robot navigation stack for an AgileX Tracer chassis. It combines visual SLAM, point-cloud obstacle extraction, RRT-style local planning, semantic/frontier memory, VLM target detection, and a finite-state navigation controller.

The main runtime is `nav_system/main.py`. It runs a 10 Hz navigation loop and starts the supporting services it needs: MASt3R-SLAM, odometry, the Tracer HTTP client, an asynchronous VLM worker, point-cloud caching, emergency-stop input, and a Flask debug view.

## Architecture

```text
RGB camera ROS topic (/camera/f_color/image_raw)
                |
                v
        MASt3R-SLAM wrapper ----> pose + dense points
                |                         |
                v                         v
        navigation FSM <---------- obstacle snapshot
          |       |  \
          |       |   +--> nav_memory (walked/frontier memory)
          |       +------> VLM (Qwen via OpenAI-compatible API)
          v
  TracerRobot HTTP client ----> tracer_http_interface ----> /cmd_vel

  Flask debug UI: http://localhost:5001/
```

Important directories:

| Path | Purpose |
| --- | --- |
| `nav_system/` | Navigation FSM, policies, planning, control, VLM integration, and runtime entry point |
| `nav_memory/` | Persistent walked-memory and frontier-grid utilities |
| `MASt3R-SLAM/` | Bundled visual SLAM implementation and native backend |
| `mobile_sam/` | Mobile SAM model and predictor used to refine VLM detections |
| `tracer_http_interface/` | ROS-to-HTTP bridge for AgileX Tracer control and telemetry |
| `vlm/` | vLLM startup script and local Qwen model directory |
| `episodes/` | Runtime debug output and structured episode logs |

## Prerequisites

- Ubuntu with ROS1 (the code uses `rospy`, `roslaunch`, `rosrun`, and catkin).
- An AgileX Tracer base and its ROS driver, or a compatible simulator publishing the same topics.
- A camera publishing RGB frames on `/camera/f_color/image_raw`.
- Python 3.10/3.11 environment with a PyTorch build matching the installed CUDA driver. A GPU is required for the bundled MASt3R and Mobile SAM paths.
- `git`, `catkin_make`, and a working ROS environment (`roscore` available).
- MASt3R and retrieval checkpoints under `MASt3R-SLAM/checkpoints/` (see the bundled [MASt3R-SLAM instructions](MASt3R-SLAM/README.md)).
- The Mobile SAM checkpoint at `mobile_sam/mobile_sam.pt`. The default constant currently points to `/home/agilex/yinzecheng/opennav/mobile_sam/mobile_sam.pt`; update `MOBILE_SAM_CHECKPOINT_PATH` in `nav_system/nav_constants.py` if your checkout lives elsewhere.

## Installation

Clone the repository, then verify that the `MASt3R-SLAM/` gitlink is populated. The current checkout contains `mobile_sam/` directly, while the historical `.gitmodules` file refers to a different MASt3R path, so clean clones may require checking out that dependency manually.

```bash
git clone <repository-url> opennav
cd opennav
# Verify this directory contains the MASt3R-SLAM sources:
ls MASt3R-SLAM
```

Create or activate a Python environment, then install the local SLAM packages and the HTTP bridge dependencies:

```bash
conda create -n opennav python=3.11
conda activate opennav

pip install -e MASt3R-SLAM/thirdparty/mast3r
pip install -e MASt3R-SLAM/thirdparty/in3d
pip install --no-build-isolation -e MASt3R-SLAM
pip install -r tracer_http_interface/requirements.txt
```

Install the remaining Python dependencies used by the navigation modules (including `rospy`, `opencv-python`, `numpy`, `openai`, `flask`, `torch`, `torchvision`, `lietorch`, and the ROS message packages) using the versions appropriate for your ROS distribution and CUDA build. The repository does not currently provide a single root `requirements.txt`.

Build and source the catkin workspace that contains the Tracer driver and this package:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Running the stack

Start the ROS master and the Tracer base driver first. For a physical Tracer, the CAN setup commonly looks like:

```bash
roscore
sudo modprobe gs_usb
rosrun tracer_bringup bringup_can2usb.bash
roslaunch tracer_bringup tracer_robot_base.launch
```

In a second shell, start the HTTP bridge on port 8080:

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch tracer_http_interface tracer_http_interface.launch
```

The bridge exposes Swagger documentation at `http://localhost:8080/docs`. OpenNav connects to this service through `http://localhost:8080`.

In a third shell, start the VLM service. The default script serves `vlm/Qwen3-VL-8B-Instruct-FP8` on port 8222:

```bash
cd /path/to/opennav
bash vlm/start.bash
```

Pass a different model directory as the first argument when needed:

```bash
bash vlm/start.bash /models/Qwen3-VL-8B-Instruct-FP8
```

Finally, start navigation from the repository root:

```bash
cd /path/to/opennav
python nav_system/main.py --mode open
```

The VLM endpoint can be overridden without editing source:

```bash
python nav_system/main.py \
  --mode object \
  --vlm-url http://localhost:8222/v1
```

The Flask debug UI is available at `http://localhost:5001/`. It shows the RGB stream, top-down map, current state, and controls for VLM direction queries, detection, frontier requests, free-form VLM questions, and the emergency stop toggle.

## Navigation modes

`--mode` accepts:

- `open` (default): performs an opening scan and requires a sustained VLM presence window before final target adjustment.
- `object`: performs an opening scan, uses VLM direction queries during patrol, and accepts a single valid 3D target detection for final adjustment.
- `frontier`: selects geometric frontier goals from the memory grid and does not use VLM direction queries for patrol planning.

All modes share the same recovery states for SLAM loss (`RELOC`), obstacle intrusion (`ESCAPE`), route planning/following, and final target approach.

## Configuration

Most runtime tuning lives in [`nav_system/nav_constants.py`](nav_system/nav_constants.py), including:

- robot radius, obstacle inflation, and path-planning tolerances;
- motion limits and pure-pursuit parameters;
- VLM URL, target name, and detection cadence;
- memory/frontier-grid settings;
- debug logging and Web UI switches.

The semantic vocabulary is in [`nav_memory/config_mem.yaml`](nav_memory/config_mem.yaml). Camera intrinsics are loaded from [`MASt3R-SLAM/config/intrinsics.yaml`](MASt3R-SLAM/config/intrinsics.yaml) when present. The SLAM wrapper expects its MASt3R and retrieval weights in `MASt3R-SLAM/checkpoints/`.

## Tracer HTTP API

The bridge provides motion, light, telemetry, odometry, and health endpoints. Typical checks are:

```bash
curl http://localhost:8080/api/health
curl http://localhost:8080/api/odom
curl -X POST http://localhost:8080/api/motion/stop
```

Continuous velocity commands are protected by the bridge watchdog; use `/api/motion/timed_move` for one-shot test motions. See [`tracer_http_interface/README.md`](tracer_http_interface/README.md) for the complete endpoint reference and launch arguments.

## Logs and troubleshooting

- Runtime episode artifacts are written below `episodes/` when the debug features in `nav_constants.py` are enabled.
- If navigation remains in `WAITING`, verify that ROS is running, the RGB topic is publishing, and MASt3R checkpoints are present.
- If the robot does not move, check the Tracer HTTP bridge at port 8080, `/api/health`, and whether the Web UI emergency stop is latched.
- If VLM requests fail, verify that `http://localhost:8222/v1` is reachable and that the served model accepts OpenAI-compatible multimodal chat requests.
- If Mobile SAM fails to load, correct `MOBILE_SAM_CHECKPOINT_PATH` and ensure CUDA is available to PyTorch.
- Stop the process with `Ctrl+C`; the runtime registers cleanup handlers for the motion thread, VLM worker, point-cloud worker, robot client, and SLAM.

## Related documentation

- [MASt3R-SLAM README](MASt3R-SLAM/README.md)
- [Tracer HTTP interface README](tracer_http_interface/README.md)
- [Mobile SAM README](mobile_sam/readme.md)

## Status

This repository is an actively developed research/prototype stack. Hardware topics, model checkpoints, calibration, and local ROS package names are expected to be adapted to the deployment machine before autonomous operation.
