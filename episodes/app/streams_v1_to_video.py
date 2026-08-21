#!/usr/bin/env python
"""Encode nav_debug streams_v1 JPEG frames into a timestamp-timed MP4.

Run from the repository root, for example:

    conda activate daily
    python workspace/frame_reshape/scrips/streams_v1_to_video.py episodes/eps_20260815_185314/frames --overwrite

The latest source frame at or before each output timestamp is held on screen.
By default the final source frame remains visible for exactly 1 second at
30 FPS. The script does not crop or rearrange streams_v1 images.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2


EXPECTED_LAYOUT = "streams_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode nav_debug streams_v1 frames using timestamps from "
            "frames/index.jsonl."
        )
    )
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Episode frames directory containing index.jsonl and step_*.jpg.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output MP4 path. Default: "
            "<episode>/frame_reshape/streams_timeline_30fps.mp4."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Constant output frame rate (default: 30).",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="Four-character OpenCV codec (default: mp4v).",
    )
    parser.add_argument(
        "--final-hold-seconds",
        type=float,
        default=1.0,
        help="Time to keep the final source frame visible (default: 1.0).",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="Print progress at this many encoded seconds; 0 disables it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing MP4 and JSON metadata file.",
    )
    return parser.parse_args()


def load_frame_records(frames_dir: Path) -> list[dict]:
    index_path = frames_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Frame index not found: {index_path}")

    records = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") != "frame":
                continue
            data = record.get("data")
            if not isinstance(data, dict) or not data.get("path"):
                raise ValueError(f"Missing frame path at {index_path}:{line_number}")
            layout = data.get("layout")
            if layout != EXPECTED_LAYOUT:
                raise ValueError(
                    f"Unsupported layout {layout!r} at {index_path}:{line_number}; "
                    f"expected {EXPECTED_LAYOUT!r}"
                )
            if record.get("ts") is None:
                raise ValueError(f"Missing timestamp at {index_path}:{line_number}")

            filename = Path(str(data["path"])).name
            frame_path = frames_dir / filename
            if not frame_path.is_file():
                raise FileNotFoundError(f"Frame image not found: {frame_path}")
            records.append(
                {
                    "ts": float(record["ts"]),
                    "step": int(record.get("step", 0)),
                    "path": frame_path,
                }
            )

    records.sort(key=lambda item: (item["ts"], item["step"]))
    if not records:
        raise RuntimeError(f"No {EXPECTED_LAYOUT} frame records found in {index_path}")
    return records


def read_frame(path: Path, expected_size: tuple[int, int] | None = None):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode JPEG frame: {path}")
    height, width = image.shape[:2]
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError(
            f"Unexpected frame size {width}x{height} at {path}; "
            f"expected {expected_size[0]}x{expected_size[1]}"
        )
    return image


def encode_video(
    records: list[dict],
    output_path: Path,
    fps: float,
    codec: str,
    final_hold_seconds: float,
    progress_seconds: float,
    overwrite: bool,
) -> dict:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if len(codec) != 4:
        raise ValueError("--codec must contain exactly four characters")
    if final_hold_seconds <= 0:
        raise ValueError("--final-hold-seconds must be positive")
    if progress_seconds < 0:
        raise ValueError("--progress-seconds must be zero or positive")

    metadata_path = output_path.with_suffix(".json")
    if not overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    start_ts = records[0]["ts"]
    last_ts = records[-1]["ts"]
    source_span_seconds = max(0.0, last_ts - start_ts)
    final_source_start_frame = max(
        0, math.ceil(source_span_seconds * fps - 1e-9)
    )
    final_hold_frames = max(1, math.ceil(final_hold_seconds * fps))
    output_frame_count = final_source_start_frame + final_hold_frames

    current_image = read_frame(records[0]["path"])
    height, width = current_image.shape[:2]
    expected_size = (width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.part{output_path.suffix}"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        expected_size,
    )
    if not writer.isOpened():
        raise OSError(
            f"Could not open video writer for {output_path} with codec {codec!r}"
        )

    source_index = 0
    progress_interval = (
        max(1, round(progress_seconds * fps)) if progress_seconds > 0 else None
    )
    try:
        for output_index in range(output_frame_count):
            target_ts = start_ts + output_index / fps
            next_source_index = source_index
            while (
                next_source_index + 1 < len(records)
                and records[next_source_index + 1]["ts"] <= target_ts
            ):
                next_source_index += 1

            if next_source_index != source_index:
                source_index = next_source_index
                current_image = read_frame(
                    records[source_index]["path"], expected_size
                )

            writer.write(current_image)
            if (
                progress_interval is not None
                and (output_index + 1) % progress_interval == 0
            ):
                print(
                    f"Video progress: {output_index + 1}/{output_frame_count} "
                    f"frames ({(output_index + 1) / fps:.1f}s)"
                )
    except Exception:
        writer.release()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.release()

    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise OSError(f"Video writer produced no output: {temporary_path}")
    temporary_path.replace(output_path)

    info = {
        "path": str(output_path),
        "layout": EXPECTED_LAYOUT,
        "codec": codec,
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": len(records),
        "output_frame_count": output_frame_count,
        "start_ts": start_ts,
        "last_source_ts": last_ts,
        "source_span_seconds": source_span_seconds,
        "final_source_output_start_frame": final_source_start_frame,
        "final_hold_frames": final_hold_frames,
        "final_hold_duration_seconds": final_hold_frames / fps,
        "duration_seconds": output_frame_count / fps,
        "timing_mode": "latest_source_frame_at_or_before_output_timestamp",
    }
    metadata_path.write_text(
        json.dumps(info, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return info


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else frames_dir.parent
        / "frame_reshape"
        / "streams_timeline_30fps.mp4"
    )
    records = load_frame_records(frames_dir)
    info = encode_video(
        records=records,
        output_path=output_path,
        fps=args.fps,
        codec=args.codec,
        final_hold_seconds=args.final_hold_seconds,
        progress_seconds=args.progress_seconds,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {info['output_frame_count']} frames "
        f"({info['duration_seconds']:.3f}s) to {output_path}"
    )


if __name__ == "__main__":
    main()
