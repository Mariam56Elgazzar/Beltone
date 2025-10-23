#!/usr/bin/env python3
"""
Baseline MWVRP solver skeleton for the Beltone 2nd AI Hackathon.

- Exposes a required entrypoint: solver(env) -> Dict
- Does NOT create or import the environment inside solver (only in __main__)
- No caching is used
- Safe to import from other modules (e.g., dashboard)

You should iterate on `solver(env)` to build your actual logic.
"""
from typing import Dict, List

# Optionally keep a second name for compatibility with older imports
# (run_dashboard in this repo previously imported `my_solver`)

def solver(env) -> Dict:
    """Return a trivially valid solution structure.

    This baseline returns an empty set of routes. It is intended purely as a
    structural starting point. The environment's validator may allow empty
    solutions for smoke tests, but for the hackathon you must implement a real
    routing strategy that fulfills orders while respecting constraints.
    """
    # Do not import or construct the environment here.
    # Querying metadata is fine if you want (e.g., env.get_all_order_ids())
    # but the baseline keeps it minimal and fast.
    return {"routes": []}

# Backwards-compatible alias so existing scripts using `my_solver` still work
my_solver = solver


if __name__ == "__main__":
    # NOTE: For local testing only. When submitting your final solver file to the
    # portal, you should comment out the block below, per competition rules.
    try:
        # First try the module name commonly used by the package
        from robin_logistics import LogisticsEnvironment  # type: ignore
    except Exception:  # pragma: no cover - import fallback
        # Fallback if the runtime exposes a different import path
        from robin_logistics_env import LogisticsEnvironment  # type: ignore

    env = LogisticsEnvironment()

    solution = solver(env)

    print("Validating baseline solution ...")
    is_valid, msg, summary = env.validate_solution_complete(solution)
    print(f"Validation: {is_valid} - {msg}")

    if is_valid:
        env.reset_all_state()
        ok, exec_msg = env.execute_solution(solution)
        print(f"Execution: {ok} - {exec_msg}")
        stats = env.get_solution_statistics(solution)
        print("\n--- Baseline Metrics ---")
        print(
            f"Orders served: {stats.get('unique_orders_served', 0)}/"
            f"{stats.get('total_orders', 0)}"
        )
        print(f"Total distance: {stats.get('total_distance', 0):.2f} km")
        print("\n--- Summary ---")
        print(summary)
    else:
        print("Baseline solution is not valid in this scenario. Implement logic in solver(env).")
