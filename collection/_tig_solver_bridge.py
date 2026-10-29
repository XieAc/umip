"""Load the shared TIG solvers with the XAC calibration backend.

The numerical implementation is stored in ``tig_solvers``. This adapter
replaces its device module with the local ``stray_shift_position`` so all
reads of T_I_G use calibration/T_I_G.npy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE / "tig_solvers"
def tig_root() -> Path:
    from xac_config import TIG_ROOT

    return TIG_ROOT


def _load(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"缺少共享 TIG 求解器：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 TIG 求解器：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_validate_solver() -> ModuleType:
    import stray_shift_position as xac_stray

    sys.modules["stray_shift_position"] = xac_stray
    module = _load("validate_tig_fixed_point", SHARED_DIR / "validate_tig_fixed_point.py")
    module.DEFAULT_STRAY_ROOT = tig_root()
    module.DROP_HIGHEST_VARIANCE_CONTACT = True
    return module


def load_rotation_solver() -> ModuleType:
    import stray_shift_position as xac_stray

    sys.modules["stray_shift_position"] = xac_stray
    sys.modules["validate_tig_fixed_point"] = load_validate_solver()
    module = _load("_xac_shared_calibrate_tig_rotation", SHARED_DIR / "calibrate_tig_rotation.py")
    module.DEFAULT_INTERACTIVE_ROOT = tig_root()
    module.DROP_HIGHEST_VARIANCE_CONTACT = "--synthetic" not in sys.argv
    return module
