#!/usr/bin/env python3
"""
Dashboard runner for the Robin Logistics Environment.

Launch with: python run_dashboard.py
"""

from typing import Callable

# Import the environment with a robust fallback for different package names
try:  # Preferred import path
    from robin_logistics import LogisticsEnvironment  # type: ignore
except Exception:  # pragma: no cover
    from robin_logistics_env import LogisticsEnvironment  # type: ignore

# Import the solver function from local solver module
from solver import solver as solver_fn


def main() -> None:
    env = LogisticsEnvironment()
    # The environment expects a callable with signature solver(env) -> Dict
    env.set_solver(solver_fn)  # type: ignore[arg-type]
    env.launch_dashboard()


if __name__ == "__main__":
    main()