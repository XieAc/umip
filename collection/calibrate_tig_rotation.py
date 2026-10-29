"""XAC entry point for experimental T_I_G rotation calibration.

Run without arguments for the interactive workflow, or use the same command
line options documented by ``python calibrate_tig_rotation.py --help``.
"""

import sys

from _tig_solver_bridge import load_rotation_solver


if __name__ == "__main__":
    solver = load_rotation_solver()
    if "--synthetic" in sys.argv and "--tig-json" not in sys.argv:
        solver.tig.load_tig_matrix = lambda _path: (
            solver.SYNTHETIC_TRUE_T_I_G.copy(),
            "synthetic built-in transform",
        )
    raise SystemExit(solver.main())
