"""Create and validate the fixed gripper-pose-in-iPhone transform for XAC.

The saved matrix follows ``T_W_G = T_W_I @ T_I_G``. It describes frame G
(gripper/TCP) expressed in frame I (iPhone): translation is the TCP origin in
I, and rotation columns are the G axes expressed in I.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from xac_config import TIG_CALIBRATION_PATH

DEFAULT_OUTPUT = TIG_CALIBRATION_PATH


def vector3(text: str) -> np.ndarray:
    try:
        value = np.asarray([float(item.strip()) for item in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc
    if value.shape != (3,) or not np.isfinite(value).all():
        raise argparse.ArgumentTypeError("expected three finite comma-separated numbers")
    return value


def validate_transform(matrix: np.ndarray) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("T_I_G must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("rotation determinant must be +1")


def describe(matrix: np.ndarray) -> None:
    xyz_mm = matrix[:3, 3] * 1000.0
    rpy_deg = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    print("\nT_I_G (gripper/TCP pose expressed in iPhone frame):")
    print(np.array2string(matrix, precision=9, suppress_small=True))
    print(f"translation [mm]: {xyz_mm[0]:.3f}, {xyz_mm[1]:.3f}, {xyz_mm[2]:.3f}")
    print(f"rotation xyz [deg]: {rpy_deg[0]:.3f}, {rpy_deg[1]:.3f}, {rpy_deg[2]:.3f}")


def save(matrix: np.ndarray, output: Path, source: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.stem}_{stamp}.bak.npy")
        shutil.copy2(output, backup)
        print(f"old calibration backed up to: {backup}")
    temporary = output.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, matrix.astype(np.float64), allow_pickle=False)
    temporary.replace(output)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "convention": "T_W_G = T_W_I @ T_I_G",
        "translation_unit": "meter",
        **source,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output}")


def load_candidate(path: Path) -> tuple[np.ndarray, dict]:
    """Load a measured candidate from NumPy or solver JSON output."""

    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        data = json.loads(resolved.read_text(encoding="utf-8"))
        matrix_data = data.get("T_I_G_candidate", data.get("T_I_G"))
        if matrix_data is None:
            raise ValueError(f"JSON 中没有可用的 T_I_G_candidate/T_I_G：{resolved}")
        matrix = np.asarray(matrix_data, dtype=np.float64)
        source = {
            "method": "imported_experimental_candidate",
            "source": str(resolved),
            "candidate_method": data.get("method"),
        }
    else:
        matrix = np.asarray(np.load(resolved, allow_pickle=False), dtype=np.float64)
        source = {"method": "imported_measured_matrix", "source": str(resolved)}
    validate_transform(matrix)
    return matrix, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-mm", type=vector3, help="TCP origin in iPhone frame: x,y,z millimetres")
    parser.add_argument("--rotation-deg", type=vector3, help="G frame orientation in I frame: extrinsic xyz Euler degrees")
    parser.add_argument("--matrix", type=Path, help="load a measured candidate from .npy or solver .json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="only validate and print the current output")
    parser.add_argument("--yes", action="store_true", help="save without the confirmation prompt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if args.check:
        if not output.exists():
            raise FileNotFoundError(f"calibration not found: {output}")
        matrix = np.asarray(np.load(output, allow_pickle=False), dtype=np.float64)
        validate_transform(matrix)
        describe(matrix)
        print("geometry check: PASS")
        return 0

    if args.matrix is not None:
        if args.translation_mm is not None or args.rotation_deg is not None:
            raise ValueError("--matrix cannot be combined with measured translation/rotation")
        matrix, source = load_candidate(args.matrix)
    else:
        if args.translation_mm is None or args.rotation_deg is None:
            raise ValueError("provide both --translation-mm and --rotation-deg, or use --matrix")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_euler("xyz", args.rotation_deg, degrees=True).as_matrix()
        matrix[:3, 3] = args.translation_mm / 1000.0
        source = {
            "method": "measured_mounting_pose",
            "translation_mm": args.translation_mm.tolist(),
            "rotation_xyz_degrees": args.rotation_deg.tolist(),
        }

    validate_transform(matrix)
    describe(matrix)
    print("\nConfirm the axes and distances on the real XAC device before saving.")
    if not args.yes and input("Write this calibration? Type YES: " ).strip() != "YES":
        print("not saved")
        return 2
    save(matrix, output, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
