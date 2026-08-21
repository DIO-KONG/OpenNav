#!/usr/bin/env python
# Usage from the repository root (PowerShell):
#   conda activate daily
#   python workspace\frame_reshape\scrips\reshape_stream_frames.py --overwrite
#
# The default input is episodes\epx_20260812_200933\frames. All frames are
# processed into <episode>\frame_reshape, including cropped RGB/map images,
# simulated stream frames, the timestamp-timed MP4, and JSONL/JSON metadata.
# Use --limit 10 for a short sample, --final-hold-seconds 5 to keep the final
# DONE state visible longer, or --help to list all path and video options.
"""Split recorded debug frames and render frames matching nav_page streams."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import cv2
import numpy as np


SEPARATOR_WIDTH = 4
CONTAINER_WIDTH = 1200
STREAM_GAP = 12
IMAGE_BORDER = 2
TITLE_MARGIN_TOP = 19
TITLE_LINE_HEIGHT = 22
TITLE_MARGIN_BOTTOM = 19

BODY_BG_BGR = (17, 17, 17)       # #111
IMAGE_BORDER_BGR = (51, 51, 51)  # #333
TITLE_BGR = (0, 255, 0)          # #0f0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_input = repo_root / "episodes" / "epx_20260812_200933" / "frames"

    parser = argparse.ArgumentParser(
        description=(
            "Crop RGB/map images from recorded debug frames and render "
            "timestamp-ordered frames that simulate nav_page.py .streams."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Cropped-image output; default: <episode>/frame_reshape/metadata.",
    )
    parser.add_argument(
        "--streams-dir",
        type=Path,
        default=None,
        help="Simulated-frame output; default: <episode>/frame_reshape/streams_sim.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of timestamp-ordered frames to process; 0 processes all.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="Only process records whose step is at least this value.",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95, choices=range(1, 101)
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing images and manifests.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="MP4 output; default: <episode>/frame_reshape/streams_timeline_30fps.mp4.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="Constant output FPS used to sample the recorded timestamp timeline.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="Four-character OpenCV video codec (default: mp4v).",
    )
    parser.add_argument(
        "--final-hold-seconds",
        type=float,
        default=2.0,
        help="Minimum time to keep the final source frame visible in the video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Process images and manifests without creating an MP4.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress after this many source frames; 0 disables it.",
    )
    return parser.parse_args()


def load_records(index_path: Path, start_step: int | None, limit: int) -> list[dict]:
    records = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            data = record.get("data", {})
            if record.get("kind") != "frame" or not data.get("path"):
                continue
            if data.get("rgb_missing") or data.get("map_missing"):
                continue
            if "rgb_shape" not in data or "map_shape" not in data:
                raise ValueError(f"Missing image shapes at {index_path}:{line_number}")
            step = int(record.get("step", 0))
            if start_step is not None and step < start_step:
                continue
            records.append(record)

    records.sort(key=lambda item: (float(item["ts"]), int(item.get("step", 0))))
    if limit > 0:
        records = records[:limit]
    return records


def crop_pair(combined: np.ndarray, record: dict) -> tuple[np.ndarray, np.ndarray]:
    data = record["data"]
    rgb_h, rgb_w = map(int, data["rgb_shape"][:2])
    map_h, map_w = map(int, data["map_shape"][:2])
    expected_h = max(rgb_h, map_h)
    expected_w = rgb_w + SEPARATOR_WIDTH + map_w

    actual_h, actual_w = combined.shape[:2]
    if actual_h < expected_h or actual_w < expected_w:
        raise ValueError(
            f"Combined frame is {actual_w}x{actual_h}, expected at least "
            f"{expected_w}x{expected_h} for step {record.get('step')}"
        )

    rgb = combined[:rgb_h, :rgb_w].copy()
    map_x = rgb_w + SEPARATOR_WIDTH
    map_image = combined[:map_h, map_x : map_x + map_w].copy()
    return rgb, map_image


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    height = max(1, round(source_h * width / source_w))
    interpolation = cv2.INTER_AREA if width < source_w else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def render_streams(rgb: np.ndarray, map_image: np.ndarray) -> np.ndarray:
    box_width = (CONTAINER_WIDTH - STREAM_GAP) // 2
    image_width = box_width - 2 * IMAGE_BORDER
    rgb_scaled = resize_to_width(rgb, image_width)
    map_scaled = resize_to_width(map_image, image_width)

    image_top = TITLE_MARGIN_TOP + TITLE_LINE_HEIGHT + TITLE_MARGIN_BOTTOM
    tallest_image = max(rgb_scaled.shape[0], map_scaled.shape[0])
    canvas_height = image_top + tallest_image + 2 * IMAGE_BORDER
    canvas = np.full(
        (canvas_height, CONTAINER_WIDTH, 3), BODY_BG_BGR, dtype=np.uint8
    )

    left_x = 0
    right_x = box_width + STREAM_GAP
    title_baseline = TITLE_MARGIN_TOP + 16
    cv2.putText(
        canvas,
        "RGB View",
        (left_x, title_baseline),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        TITLE_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Top-Down Map",
        (right_x, title_baseline),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        TITLE_BGR,
        1,
        cv2.LINE_AA,
    )

    for x, image in ((left_x, rgb_scaled), (right_x, map_scaled)):
        outer_h = image.shape[0] + 2 * IMAGE_BORDER
        canvas[image_top : image_top + outer_h, x : x + box_width] = IMAGE_BORDER_BGR
        y0 = image_top + IMAGE_BORDER
        x0 = x + IMAGE_BORDER
        canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image

    return canvas


def write_jpeg(path: Path, image: np.ndarray, quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def write_manifest(path: Path, rows: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def write_timestamp_video(
    stream_rows: list[dict],
    streams_dir: Path,
    video_path: Path,
    fps: float,
    codec: str,
    final_hold_seconds: float,
    overwrite: bool,
) -> dict:
    if fps <= 0:
        raise ValueError("--video-fps must be positive")
    if len(codec) != 4:
        raise ValueError("--video-codec must contain exactly four characters")
    if final_hold_seconds <= 0:
        raise ValueError("--final-hold-seconds must be positive")
    if video_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing video: {video_path}")

    timestamps = [float(row["ts"]) for row in stream_rows]
    positive_deltas = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    median_source_interval = (
        statistics.median(positive_deltas) if positive_deltas else 1.0 / fps
    )
    start_ts = timestamps[0]
    final_source_start_index = max(
        0, math.ceil((timestamps[-1] - start_ts) * fps - 1e-9)
    )
    final_hold_frames = max(1, math.ceil(final_hold_seconds * fps))
    output_count = final_source_start_index + final_hold_frames

    first_path = streams_dir / stream_rows[0]["path"]
    current_image = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if current_image is None:
        raise FileNotFoundError(f"Could not read simulated frame: {first_path}")
    height, width = current_image.shape[:2]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(
            f"Could not open video writer for {video_path} with codec {codec}"
        )

    source_index = 0
    try:
        for output_index in range(output_count):
            target_ts = start_ts + output_index / fps
            next_index = source_index
            while (
                next_index + 1 < len(stream_rows)
                and float(stream_rows[next_index + 1]["ts"]) <= target_ts
            ):
                next_index += 1

            if next_index != source_index:
                source_index = next_index
                frame_path = streams_dir / stream_rows[source_index]["path"]
                current_image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if current_image is None:
                    raise FileNotFoundError(
                        f"Could not read simulated frame: {frame_path}"
                    )
                if current_image.shape[:2] != (height, width):
                    raise ValueError(
                        f"Unexpected frame size {current_image.shape[:2]} at {frame_path}"
                    )

            writer.write(current_image)
            if (output_index + 1) % max(1, round(fps * 30)) == 0:
                print(
                    f"Video progress: {output_index + 1}/{output_count} frames "
                    f"({(output_index + 1) / fps:.1f}s)"
                )
    finally:
        writer.release()

    return {
        "path": video_path.name,
        "codec": codec,
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": len(stream_rows),
        "output_frame_count": output_count,
        "start_ts": start_ts,
        "last_source_ts": timestamps[-1],
        "median_source_interval_s": median_source_interval,
        "final_source_output_start_frame": final_source_start_index,
        "final_hold_frames": final_hold_frames,
        "final_hold_duration_s": final_hold_frames / fps,
        "duration_s": output_count / fps,
        "timing_mode": "latest_source_frame_at_or_before_output_timestamp",
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = input_dir.parent / "frame_reshape"
    metadata_dir = (
        args.metadata_dir.resolve()
        if args.metadata_dir is not None
        else output_root / "metadata"
    )
    streams_dir = (
        args.streams_dir.resolve()
        if args.streams_dir is not None
        else output_root / "streams_sim"
    )
    index_path = input_dir / "index.jsonl"

    if not index_path.is_file():
        raise FileNotFoundError(f"Frame index not found: {index_path}")
    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be zero or positive")

    rgb_dir = metadata_dir / "rgb"
    map_dir = metadata_dir / "map"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    streams_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(index_path, args.start_step, args.limit)
    if not records:
        raise RuntimeError("No matching frame records were found")

    metadata_rows = []
    stream_rows = []
    for order, record in enumerate(records, start=1):
        step = int(record.get("step", 0))
        timestamp = float(record["ts"])
        timestamp_us = round(timestamp * 1_000_000)
        stem = f"{order:06d}_step_{step:06d}_ts_{timestamp_us}"
        source_path = input_dir.parent / record["data"]["path"]

        combined = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if combined is None:
            raise FileNotFoundError(f"Could not read source frame: {source_path}")
        rgb, map_image = crop_pair(combined, record)
        simulated = render_streams(rgb, map_image)

        rgb_path = rgb_dir / f"{stem}.jpg"
        map_path = map_dir / f"{stem}.jpg"
        stream_path = streams_dir / f"{stem}.jpg"
        write_jpeg(rgb_path, rgb, args.jpeg_quality, args.overwrite)
        write_jpeg(map_path, map_image, args.jpeg_quality, args.overwrite)
        write_jpeg(stream_path, simulated, args.jpeg_quality, args.overwrite)

        common = {
            "order": order,
            "step": step,
            "ts": timestamp,
            "ts_us": timestamp_us,
            "source": str(source_path.relative_to(input_dir.parents[2])).replace("\\", "/"),
        }
        metadata_rows.append(
            {
                **common,
                "rgb_path": str(rgb_path.relative_to(metadata_dir)).replace("\\", "/"),
                "rgb_shape": list(rgb.shape),
                "map_path": str(map_path.relative_to(metadata_dir)).replace("\\", "/"),
                "map_shape": list(map_image.shape),
            }
        )
        stream_rows.append(
            {
                **common,
                "path": stream_path.name,
                "shape": list(simulated.shape),
            }
        )
        if args.progress_every and order % args.progress_every == 0:
            print(f"Image progress: {order}/{len(records)} source frames")

    write_manifest(metadata_dir / "index.jsonl", metadata_rows, args.overwrite)
    write_manifest(streams_dir / "index.jsonl", stream_rows, args.overwrite)
    print(
        f"Processed {len(records)} frames into {metadata_dir} and {streams_dir}"
    )

    if not args.no_video:
        video_path = (
            args.video_output.resolve()
            if args.video_output is not None
            else output_root / "streams_timeline_30fps.mp4"
        )
        video_info = write_timestamp_video(
            stream_rows,
            streams_dir,
            video_path,
            args.video_fps,
            args.video_codec,
            args.final_hold_seconds,
            args.overwrite,
        )
        video_info_path = video_path.with_suffix(".json")
        if video_info_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing video metadata: {video_info_path}"
            )
        video_info_path.write_text(
            json.dumps(video_info, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote timestamp-timed video with {video_info['output_frame_count']} "
            f"frames to {video_path}"
        )


if __name__ == "__main__":
    main()
