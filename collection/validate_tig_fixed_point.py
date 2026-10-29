"""XAC entry point for fixed-point probe validation and translation fitting."""

from _tig_solver_bridge import load_validate_solver


if __name__ == "__main__":
    raise SystemExit(load_validate_solver().main())
