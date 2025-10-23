from DataVerse_solver_40 import solver
from robin_logistics import LogisticsEnvironment

# 1. Initialize the Environment (Required)
# NOTE: This line needs to be executed to create the 'env' object.
env = LogisticsEnvironment()

# 2. Get the Solution (Call your solver function)
# WARNING: This will take several minutes due to the GA complexity.
print("Running solver to generate solution...")
solution = solver(env)
print(f"Solver returned {len(solution.get('routes', []))} routes.")

# --- Validation, Execution, and Reporting ---

# 3. Validate the solution using the 'env' instance
is_valid, msg, summary = env.validate_solution_complete(solution)
print("\n--- Validation Results ---")
print(f"Validation: {is_valid} - {msg}, total routes: {len(solution['routes'])}")

if is_valid:
    # 4. Execute the solution only if valid (after reset)
    env.reset_all_state() # Ensure a clean execution state
    ok, exec_msg = env.execute_solution(solution)
    print(f"Execution: {ok} - {exec_msg}")

    # 5. Get and print statistics
    stats = env.get_solution_statistics(solution)
    print("\n--- Final Metrics ---")
    print(f"Orders served: {stats.get('unique_orders_served', 0)}/{stats.get('total_orders', 0)}")
    print(f"Total distance: {stats.get('total_distance', 0):.2f} km")
    
    # Print the full summary for debugging purposes
    print("\n--- Detailed Route Summary ---")
    print(summary)