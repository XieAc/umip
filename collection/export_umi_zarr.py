"""Export locally built UMI episodes to the official UMI replay-buffer Zarr format.

The builder stores per-frame observations in episode directories and also keeps
validity masks for diagnostics. This exporter removes invalid frame runs before
writing a single ``data/...zarr.zip`` file, so the official UmiDataset never sees
NaNs or invalid RGB placeholders.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import zarr


DATA_KEYS = (
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "action",
    "robot0_demo_start_pose",
    "robot0_demo_end_pose",
)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous true values."""

    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def load_episode_segments(episode_dir: Path, min_length: int, resample_fps: float | None = None,
                          max_gap_seconds: float = 0.3) -> list[dict[str, Any]]:
    data_dir = episode_dir / "data"
    source_info = json.loads(
        (episode_dir / "metadata" / "source.json").read_text(encoding="utf-8")
    )
    valid = np.load(data_dir / "valid_mask.npy", allow_pickle=False).astype(bool)
    rgb = np.load(data_dir / "camera0_rgb.npy", mmap_mode="r", allow_pickle=False)
    pos = np.load(data_dir / "robot0_eef_pos.npy", mmap_mode="r", allow_pickle=False)
    rot = np.load(
        data_dir / "robot0_eef_rot_axis_angle.npy", mmap_mode="r", allow_pickle=False
    )
    width = np.load(data_dir / "robot0_gripper_width.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(data_dir / "d435i_timestamps.npy", allow_pickle=False)

    if not (
        len(valid) == len(rgb) == len(pos) == len(rot) == len(width) == len(timestamps)
        and rgb.shape[1:] == (224, 224, 3)
        and pos.shape[1:] == (3,)
        and rot.shape[1:] == (3,)
        and width.shape[1:] == (1,)
    ):
        raise ValueError(f"episode shape mismatch: {episode_dir}")

    finite = (
        np.isfinite(pos).all(axis=1)
        & np.isfinite(rot).all(axis=1)
        & np.isfinite(width).all(axis=1)
    )
    usable = valid & finite
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    if not np.isfinite(timestamps).all() or np.any(dt <= 0):
        raise ValueError(f"non-increasing or non-finite timestamps: {episode_dir}")
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    frequency_hz = float(1.0 / np.median(positive_dt)) if len(positive_dt) else None
    segments: list[dict[str, Any]] = []
    for start, end in contiguous_true_runs(usable):
        # Never interpolate across a long acquisition interruption.
        cuts = [start, end]
        if resample_fps is not None:
            cuts = [start, *(start + 1 + np.flatnonzero(dt[start:end-1] > max_gap_seconds)), end]
        for run_start, run_end in zip(cuts[:-1], cuts[1:]):
            length = run_end - run_start
            if resample_fps is not None:
                length = int(np.floor((timestamps[run_end-1] - timestamps[run_start]) * resample_fps)) + 1
            if length < min_length:
                continue
            segments.append(
                {
                    "source_episode": episode_dir.name,
                    "source_session": source_info["source_session_name"],
                    "source_start": int(run_start),
                    "source_end": int(run_end),
                    "length": length,
                    "frequency_hz": resample_fps or frequency_hz,
                }
            )
    return segments


def resample_observations(timestamps, rgb, pos, rot, width, fps):
    """Uniform time grid: nearest real RGB, linear state, spherical rotations."""
    from scipy.spatial.transform import Rotation, Slerp

    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) == 1:
        return np.asarray(rgb), pos.astype(np.float32), rot.astype(np.float32), width.astype(np.float32)
    relative = times - times[0]
    count = int(np.floor(relative[-1] * fps)) + 1
    target = np.arange(count, dtype=np.float64) / fps
    right = np.searchsorted(relative, target).clip(0, len(relative)-1)
    left = (right - 1).clip(0)
    nearest = np.where(target - relative[left] <= relative[right] - target, left, right)
    positions = np.column_stack([np.interp(target, relative, pos[:, i]) for i in range(3)])
    widths = np.interp(target, relative, width[:, 0])[:, None]
    rotations = Slerp(relative, Rotation.from_rotvec(rot))(target).as_rotvec()
    return np.asarray(rgb[nearest]), positions.astype(np.float32), rotations.astype(np.float32), widths.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-segment-length", type=int, default=32)
    parser.add_argument("--resample-fps", type=float, default=None,
                        help="Resample each valid segment to uniform Hz before export.")
    parser.add_argument("--max-gap-seconds", type=float, default=0.3,
                        help="Split before interpolating over larger acquisition gaps.")
    parser.add_argument(
        "--exclude-source-sessions",
        nargs="*",
        default=[],
        help="原始 session 编号，例如：--exclude-source-sessions 003 005",
    )
    return parser.parse_args()


def write_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    total_frames: int,
    excluded_source_sessions: list[str],
    frequencies_hz: list[float],
    resample_fps: float | None = None,
    max_gap_seconds: float = 0.3,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "zarr_episode",
        "source_episode",
        "source_session",
        "source_start_frame",
        "source_end_frame_exclusive",
        "frame_count",
        "global_start_frame",
        "global_end_frame_exclusive",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary_path = path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "format": "official UMI ReplayBuffer Zarr v2",
                "episodes": len(rows),
                "total_frames": total_frames,
                "manifest_csv": path.name,
                "excluded_source_sessions": excluded_source_sessions,
                "exclusion_reasons": {session: "excluded by command line; see dataset audit report"
                                      for session in excluded_source_sessions},
                "resample_fps": resample_fps,
                "resampling": ("nearest real RGB; linear positions/width; spherical rotation interpolation"
                               if resample_fps else None),
                "max_interpolation_gap_seconds": max_gap_seconds if resample_fps else None,
                "image_preprocess": "RGB, 224x224 center letterbox, no predefined gripper mask",
                "frequency_hz_median": (
                    float(np.median(frequencies_hz)) if frequencies_hz else None
                ),
                "frequency_hz_range": (
                    [float(min(frequencies_hz)), float(max(frequencies_hz))]
                    if frequencies_hz else None
                ),
                "action_definition": "per-frame [eef_pos_xyz_m, eef_rot_axis_angle_rad, gripper_width_m]",
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.min_segment_length < 1:
        raise ValueError("--min-segment-length must be positive")
    if args.resample_fps is not None and (not np.isfinite(args.resample_fps) or args.resample_fps <= 0):
        raise ValueError("--resample-fps must be finite and positive")
    if not np.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0:
        raise ValueError("--max-gap-seconds must be finite and positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    episode_dirs = sorted(
        path
        for path in args.episodes_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not episode_dirs:
        raise FileNotFoundError(f"no numeric episodes under {args.episodes_root}")

    excluded_source_sessions = {
        str(value).strip().zfill(3)
        for value in args.exclude_source_sessions
        if str(value).strip()
    }
    segments: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        source_info = json.loads(
            (episode_dir / "metadata" / "source.json").read_text(encoding="utf-8")
        )
        source_session = str(source_info["source_session_name"]).strip().zfill(3)
        if source_session in excluded_source_sessions:
            print(f"skip source session={source_session} local_episode={episode_dir.name}")
            continue
        segments.extend(load_episode_segments(episode_dir, args.min_segment_length,
                                             args.resample_fps, args.max_gap_seconds))
    if not segments:
        raise RuntimeError("no usable episode segments")

    total_frames = sum(int(segment["length"]) for segment in segments)
    episode_ends = np.cumsum([int(segment["length"]) for segment in segments], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    with tempfile.TemporaryDirectory(
        prefix="xac_umi_zarr_", dir=str(args.output.parent)
    ) as temporary_root:
        directory_store = zarr.DirectoryStore(str(Path(temporary_root) / "dataset.zarr"))
        root = zarr.group(store=directory_store, overwrite=True)
        data = root.create_group("data")
        meta = root.create_group("meta")
        meta.array("episode_ends", data=episode_ends, chunks=episode_ends.shape, compressor=None)

        arrays = {
            "camera0_rgb": data.zeros(
                "camera0_rgb",
                shape=(total_frames, 224, 224, 3),
                chunks=(1, 224, 224, 3),
                dtype=np.uint8,
                compressor=compressor,
            ),
            "robot0_eef_pos": data.zeros(
                "robot0_eef_pos",
                shape=(total_frames, 3),
                chunks=(512, 3),
                dtype=np.float32,
                compressor=compressor,
            ),
            "robot0_eef_rot_axis_angle": data.zeros(
                "robot0_eef_rot_axis_angle",
                shape=(total_frames, 3),
                chunks=(512, 3),
                dtype=np.float32,
                compressor=compressor,
            ),
            "robot0_gripper_width": data.zeros(
                "robot0_gripper_width",
                shape=(total_frames, 1),
                chunks=(512, 1),
                dtype=np.float32,
                compressor=compressor,
            ),
            "action": data.zeros(
                "action",
                shape=(total_frames, 7),
                chunks=(512, 7),
                dtype=np.float32,
                compressor=compressor,
            ),
            "robot0_demo_start_pose": data.zeros(
                "robot0_demo_start_pose",
                shape=(total_frames, 6),
                chunks=(512, 6),
                dtype=np.float32,
                compressor=compressor,
            ),
            "robot0_demo_end_pose": data.zeros(
                "robot0_demo_end_pose",
                shape=(total_frames, 6),
                chunks=(512, 6),
                dtype=np.float32,
                compressor=compressor,
            ),
        }

        manifest_rows: list[dict[str, Any]] = []
        global_start = 0
        for zarr_episode, segment in enumerate(segments):
            episode_dir = args.episodes_root / segment["source_episode"]
            data_dir = episode_dir / "data"
            source_start = int(segment["source_start"])
            source_end = int(segment["source_end"])
            source_rgb = np.load(data_dir / "camera0_rgb.npy", mmap_mode="r", allow_pickle=False)
            source_pos = np.load(data_dir / "robot0_eef_pos.npy", mmap_mode="r", allow_pickle=False)
            source_rot = np.load(
                data_dir / "robot0_eef_rot_axis_angle.npy", mmap_mode="r", allow_pickle=False
            )
            source_width = np.load(
                data_dir / "robot0_gripper_width.npy", mmap_mode="r", allow_pickle=False
            )
            pos = np.asarray(source_pos[source_start:source_end], dtype=np.float32)
            rot = np.asarray(source_rot[source_start:source_end], dtype=np.float32)
            width = np.asarray(source_width[source_start:source_end], dtype=np.float32)
            rgb = source_rgb[source_start:source_end]
            if args.resample_fps is not None:
                timestamps = np.load(data_dir / "d435i_timestamps.npy", allow_pickle=False)[source_start:source_end]
                rgb, pos, rot, width = resample_observations(timestamps, rgb, pos, rot, width, args.resample_fps)
            n = len(pos)
            action = np.concatenate((pos, rot, width), axis=1)
            start_pose = np.concatenate((pos[0], rot[0]))
            end_pose = np.concatenate((pos[-1], rot[-1]))
            arrays["camera0_rgb"][global_start:global_start + n] = rgb
            arrays["robot0_eef_pos"][global_start:global_start + n] = pos
            arrays["robot0_eef_rot_axis_angle"][global_start:global_start + n] = rot
            arrays["robot0_gripper_width"][global_start:global_start + n] = width
            arrays["action"][global_start:global_start + n] = action
            arrays["robot0_demo_start_pose"][global_start:global_start + n] = np.broadcast_to(
                start_pose, (n, 6)
            )
            arrays["robot0_demo_end_pose"][global_start:global_start + n] = np.broadcast_to(
                end_pose, (n, 6)
            )
            manifest_rows.append(
                {
                    "zarr_episode": zarr_episode,
                    "source_episode": segment["source_episode"],
                    "source_session": segment["source_session"],
                    "source_start_frame": source_start,
                    "source_end_frame_exclusive": source_end,
                    "frame_count": n,
                    "global_start_frame": global_start,
                    "global_end_frame_exclusive": global_start + n,
                }
            )
            global_start += n
            print(
                f"{zarr_episode + 1:02d}/{len(segments):02d} "
                f"source_session={segment['source_session']} "
                f"local_episode={segment['source_episode']} frames={n}"
            )
        directory_store.close()
        source_store = zarr.DirectoryStore(str(Path(temporary_root) / "dataset.zarr"))
        try:
            with zarr.ZipStore(str(args.output), mode="w") as zip_store:
                zarr.copy_store(source=source_store, dest=zip_store, if_exists="replace")
        finally:
            source_store.close()

    write_manifest(
        args.manifest,
        manifest_rows,
        total_frames,
        sorted(excluded_source_sessions),
        [float(segment["frequency_hz"]) for segment in segments if segment["frequency_hz"]],
        args.resample_fps,
        args.max_gap_seconds,
    )
    print(f"wrote {args.output}")
    print(f"episodes={len(segments)} frames={total_frames}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
