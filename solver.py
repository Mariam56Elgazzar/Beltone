#!/usr/bin/env python3
"""
Greedy multi-warehouse vehicle routing solver for the Beltone 2nd AI Hackathon.

Key properties:
- Exposes required entrypoint `solver(env) -> Dict`.
- Does NOT import/initialize the environment inside `solver` (only in __main__).
- Avoids any caching across runs.

High-level heuristic:
1) For each order, greedily allocate items to nearby warehouses and vehicles
   while respecting capacity and inventory levels. We allow multi-warehouse
   pickups and multi-vehicle per order when needed.
2) Each vehicle builds one route consisting of move/pickup/deliver steps and
   returns to its home warehouse at the end.

The step schema is intentionally simple and generic:
  - move:    {"type":"move", "path":[nodeA, nodeB, ...]}
  - pickup:  {"type":"pickup", "warehouse_id":..., "sku_id":..., "quantity":...}
  - deliver: {"type":"deliver", "order_id":..., "sku_id":..., "quantity":...}
  - unload:  {"type":"unload", "warehouse_id":..., "sku_id":..., "quantity":...}

Adjust these if your environment requires slightly different keys.
"""
from typing import Dict, List, Tuple, Optional, Any
import heapq
import math


# --------------------------- Graph / Path Utilities ---------------------------

def _build_adjacency(road_data: Dict[str, Any]) -> Dict[int, List[Tuple[int, float]]]:
    """Build a compact adjacency list from provided road network data.

    Accepts multiple possible shapes for `road_data` to maximize compatibility.
    """
    adjacency: Dict[int, List[Tuple[int, float]]] = {}

    if not road_data:
        return adjacency

    # Case 1: Direct adjacency list
    raw_adj = road_data.get("adjacency_list") or road_data.get("adjacency")
    if raw_adj:
        for node_key, neighbors in raw_adj.items():
            try:
                node = int(node_key)
            except Exception:
                node = node_key  # keep original type
            adjacency[node] = []
            for entry in neighbors:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    nxt, d = entry[0], float(entry[1])
                elif isinstance(entry, dict):
                    nxt = entry.get("to") or entry.get("node") or entry.get("neighbor")
                    d = entry.get("distance") or entry.get("length") or entry.get("cost")
                    if nxt is None or d is None:
                        continue
                    d = float(d)
                else:
                    continue
                try:
                    nxt = int(nxt)
                except Exception:
                    pass
                adjacency[node].append((nxt, float(d)))
        return adjacency

    # Case 2: Edges list (u, v, dist)
    edges = road_data.get("edges")
    if edges:
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                u, v, d = e[0], e[1], float(e[2])
            elif isinstance(e, dict):
                u = e.get("u") or e.get("from") or e.get("source")
                v = e.get("v") or e.get("to") or e.get("target")
                d = e.get("distance") or e.get("length") or e.get("cost")
                if u is None or v is None or d is None:
                    continue
                d = float(d)
            else:
                continue
            try:
                u = int(u)
            except Exception:
                pass
            try:
                v = int(v)
            except Exception:
                pass
            adjacency.setdefault(u, []).append((v, float(d)))
        return adjacency

    return adjacency


def _dijkstra_path(
    adjacency: Dict[int, List[Tuple[int, float]]], start: int, goal: int
) -> Optional[List[int]]:
    """Compute a shortest path via Dijkstra; returns list of nodes, or None."""
    if start == goal:
        return [start]

    if not adjacency or start not in adjacency:
        return None

    distance_by_node: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start)]

    while heap:
        dist_u, u = heapq.heappop(heap)
        if u == goal:
            break
        if dist_u != distance_by_node.get(u):
            continue
        for v, w in adjacency.get(u, []):
            nd = dist_u + float(w)
            if nd < distance_by_node.get(v, math.inf):
                distance_by_node[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if goal not in distance_by_node:
        return None

    # Reconstruct path
    path: List[int] = []
    cur = goal
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = prev[cur]
    path.reverse()
    return path


def _path_distance(adjacency: Dict[int, List[Tuple[int, float]]], path: List[int]) -> float:
    """Sum edge distances along a path; returns 0.0 for degenerate paths."""
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        found = False
        for v, w in adjacency.get(a, []):
            if v == b:
                total += float(w)
                found = True
                break
        if not found:
            total += 0.0
    return total


# ------------------------------ Solver Utilities ------------------------------

def _extract_node_from_warehouse(warehouse_obj: Any) -> Optional[int]:
    """Best-effort extraction of a warehouse node id from a warehouse object."""
    if warehouse_obj is None:
        return None
    for attr in ("node", "node_id", "location", "nodeNumber", "node_idx"):
        if hasattr(warehouse_obj, attr):
            try:
                return int(getattr(warehouse_obj, attr))
            except Exception:
                pass
    if isinstance(Warehouse := warehouse_obj, dict):
        for key in ("node", "node_id", "location", "nodeNumber", "node_idx"):
            if key in Warehouse:
                try:
                    return int(Warehouse[key])
                except Exception:
                    pass
    return None


def _get_warehouse_node(env: Any, warehouse_id: str) -> Optional[int]:
    try:
        wh = env.get_warehouse_by_id(warehouse_id)
    except Exception:
        wh = None
    node = _extract_node_from_warehouse(wh)
    return node


def _append_move(steps: List[Dict[str, Any]], path: Optional[List[int]]) -> None:
    if not path or len(path) <= 1:
        return
    steps.append({"type": "move", "path": path})


def _append_pickup(
    steps: List[Dict[str, Any]], warehouse_id: str, sku_id: str, quantity: int
) -> None:
    steps.append(
        {
            "type": "pickup",
            "warehouse_id": warehouse_id,
            "sku_id": sku_id,
            "quantity": int(quantity),
        }
    )


def _append_deliver(
    steps: List[Dict[str, Any]], order_id: str, sku_id: str, quantity: int
) -> None:
    steps.append(
        {
            "type": "deliver",
            "order_id": order_id,
            "sku_id": sku_id,
            "quantity": int(quantity),
        }
    )


def _append_unload(
    steps: List[Dict[str, Any]], warehouse_id: str, sku_id: str, quantity: int
) -> None:
    steps.append(
        {
            "type": "unload",
            "warehouse_id": warehouse_id,
            "sku_id": sku_id,
            "quantity": int(quantity),
        }
    )


def _units_that_fit(
    remaining_weight: float, remaining_volume: float, unit_weight: float, unit_volume: float
) -> int:
    if unit_weight <= 0 or unit_volume <= 0:
        return 0
    by_weight = math.floor(remaining_weight / unit_weight)
    by_volume = math.floor(remaining_volume / unit_volume)
    return max(0, min(by_weight, by_volume))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get_vehicle_home_node(env: Any, vehicle_id: str) -> Optional[int]:
    try:
        node = env.get_vehicle_home_warehouse(vehicle_id)
        return int(node)
    except Exception:
        return None


def _get_order_location_node(env: Any, order_id: str) -> Optional[int]:
    try:
        node = env.get_order_location(order_id)
        return int(node)
    except Exception:
        return None


def _get_vehicle_initial_capacity(env: Any, vehicle_id: str) -> Tuple[float, float]:
    try:
        w, v = env.get_vehicle_remaining_capacity(vehicle_id)  # type: ignore[misc]
        return float(w), float(v)
    except Exception:
        pass
    try:
        w, v = env.get_vehicle_current_capacity(vehicle_id)  # type: ignore[misc]
        return float(w), float(v)
    except Exception:
        return float(0), float(0)


def _get_sku_stats(env: Any, sku_id: str) -> Tuple[float, float]:
    try:
        info = env.get_sku_details(sku_id) or {}
        w = float(info.get("weight", info.get("unit_weight", 0)))
        vol = float(info.get("volume", info.get("unit_volume", 0)))
        return w, vol
    except Exception:
        return 0.0, 0.0


# ----------------------------------- Solver -----------------------------------

def solver(env) -> Dict:
    """Greedy MWVRP solver producing a complete solution dictionary.

    Heuristics:
    - Greedy closest-warehouse pickups per order-SKU
    - Greedy vehicle selection by round-robin (can be improved)
    - Capacity- and inventory-aware partial fulfillment with multi-vehicle support
    """
    # Basic queries (be resilient to API differences)
    try:
        order_ids: List[str] = list(env.get_all_order_ids())  # type: ignore[attr-defined]
    except Exception:
        try:
            order_ids = list(env.get_all_orders())  # type: ignore[attr-defined]
        except Exception:
            order_ids = []

    try:
        vehicle_ids: List[str] = list(env.get_available_vehicles())  # type: ignore[attr-defined]
    except Exception:
        try:
            vehicle_ids = list(env.get_all_vehicles())  # type: ignore[attr-defined]
        except Exception:
            vehicle_ids = []

    road_data: Dict[str, Any] = {}
    try:
        road_data = env.get_road_network_data()  # type: ignore[attr-defined]
    except Exception:
        road_data = {}
    adjacency = _build_adjacency(road_data)

    # Prepare vehicle states
    vehicle_state: Dict[str, Dict[str, Any]] = {}
    for vid in vehicle_ids:
        home_node = _get_vehicle_home_node(env, vid)
        cap_w, cap_v = _get_vehicle_initial_capacity(env, vid)
        vehicle_state[vid] = {
            "home_node": home_node,
            "current_node": home_node,
            "remaining_weight": cap_w,
            "remaining_volume": cap_v,
            "steps": [],
            "load": {},  # sku_id -> quantity currently on board
            "distance": 0.0,
        }

    # Planned inventory snapshot to avoid over-allocation across vehicles
    planned_inventory: Dict[str, Dict[str, int]] = {}

    def get_planned_inventory(warehouse_id: str) -> Dict[str, int]:
        if warehouse_id not in planned_inventory:
            try:
                inv = env.get_warehouse_inventory(warehouse_id) or {}
            except Exception:
                inv = {}
            planned_inventory[warehouse_id] = {str(k): _safe_int(v) for k, v in inv.items()}
        return planned_inventory[warehouse_id]

    # Movement helper
    def move_vehicle(vid: str, dst_node: Optional[int]) -> None:
        st = vehicle_state[vid]
        src_node = st["current_node"]
        if dst_node is None or src_node is None:
            return
        if adjacency:
            path = _dijkstra_path(adjacency, int(src_node), int(dst_node))
        else:
            path = [src_node, dst_node]
        _append_move(st["steps"], path)
        st["current_node"] = dst_node
        if adjacency and path:
            st["distance"] += _path_distance(adjacency, path)

    # Choose next warehouse for a sku
    def choose_next_warehouse(sku_id: str, min_qty: int, origin_node: Optional[int]) -> Optional[Tuple[str, int, Optional[int]]]:
        try:
            wh_list: List[str] = list(env.get_warehouses_with_sku(sku_id, min_quantity=1))  # type: ignore[attr-defined]
        except Exception:
            wh_list = []
        best_choice: Optional[Tuple[str, int, Optional[int], float]] = None
        for wh_id in wh_list:
            inv = get_planned_inventory(wh_id)
            available = _safe_int(inv.get(sku_id, 0))
            if available <= 0:
                continue
            take = min(available, min_qty)
            node = _get_warehouse_node(env, wh_id)
            if origin_node is not None and node is not None and adjacency:
                path = _dijkstra_path(adjacency, int(origin_node), int(node))
                score = _path_distance(adjacency, path) if path else float("inf")
            else:
                # fallback scoring favors more stock
                score = -float(take)
            if best_choice is None or score < best_choice[3]:
                best_choice = (wh_id, take, node, score)
        if best_choice is None:
            return None
        return best_choice[0], best_choice[1], best_choice[2]

    # Order requirements getter
    def get_order_requirements(order_id: str) -> Dict[str, int]:
        try:
            req = env.get_order_requirements(order_id) or {}
        except Exception:
            req = {}
        return {str(k): _safe_int(v) for k, v in req.items() if _safe_int(v) > 0}

    order_requirements: Dict[str, Dict[str, int]] = {oid: get_order_requirements(oid) for oid in order_ids}

    if not order_ids or not vehicle_ids:
        return {"routes": []}

    # Sort orders (greedy: more units first)
    def order_total_units(oid: str) -> int:
        return sum(order_requirements.get(oid, {}).values())

    sorted_orders: List[str] = sorted(order_ids, key=order_total_units, reverse=True)

    # Round-robin vehicles while there is pending demand
    next_vehicle_index = 0

    for order_id in sorted_orders:
        pending = order_requirements.get(order_id, {})
        if not pending:
            continue
        order_node = _get_order_location_node(env, order_id)

        while sum(pending.values()) > 0 and next_vehicle_index < len(vehicle_ids) * 4:
            vid = vehicle_ids[next_vehicle_index % len(vehicle_ids)]
            next_vehicle_index += 1
            st = vehicle_state[vid]

            remaining_w: float = float(st["remaining_weight"])
            remaining_v: float = float(st["remaining_volume"])

            made_progress = False

            # iterate deterministically on sku ids to avoid thrashing
            for sku_id in sorted(list(pending.keys())):
                needed_qty = pending.get(sku_id, 0)
                if needed_qty <= 0:
                    continue
                unit_w, unit_v = _get_sku_stats(env, sku_id)
                if unit_w <= 0 or unit_v <= 0:
                    continue
                max_fit = _units_that_fit(remaining_w, remaining_v, unit_w, unit_v)
                if max_fit <= 0:
                    continue
                target_qty = min(needed_qty, max_fit)

                # choose warehouse for this batch
                choice = choose_next_warehouse(sku_id, target_qty, st["current_node"])
                if choice is None:
                    continue
                wh_id, feasible_qty, wh_node = choice
                if feasible_qty <= 0:
                    continue
                take_qty = min(target_qty, feasible_qty)

                # move -> pickup
                move_vehicle(vid, wh_node)
                _append_pickup(st["steps"], wh_id, sku_id, take_qty)
                get_planned_inventory(wh_id)[sku_id] -= take_qty
                st["load"][sku_id] = st["load"].get(sku_id, 0) + take_qty
                remaining_w -= unit_w * take_qty
                remaining_v -= unit_v * take_qty

                # move -> deliver
                move_vehicle(vid, order_node)
                _append_deliver(st["steps"], order_id, sku_id, take_qty)
                st["load"][sku_id] -= take_qty
                if st["load"][sku_id] <= 0:
                    del st["load"][sku_id]
                pending[sku_id] -= take_qty
                if pending[sku_id] <= 0:
                    del pending[sku_id]
                made_progress = True

                if sum(pending.values()) <= 0:
                    break

            if not made_progress and sum(pending.values()) > 0:
                # If all vehicles stall eventually break to avoid infinite loop
                if next_vehicle_index >= len(vehicle_ids):
                    break

    # Close routes: return to home
    routes: List[Dict[str, Any]] = []
    for vid in vehicle_ids:
        st = vehicle_state[vid]
        if st["home_node"] is not None and st["current_node"] is not None:
            if st["home_node"] != st["current_node"]:
                move_vehicle(vid, st["home_node"])
        routes.append(
            {
                "vehicle_id": vid,
                "start_node": st["home_node"],
                "end_node": st["home_node"],
                "steps": st["steps"],
                "total_distance": round(float(st["distance"]), 3),
            }
        )

    solution: Dict[str, Any] = {"routes": routes}
    return solution


# Backwards-compatible alias so existing scripts using `my_solver` still work
my_solver = solver


if __name__ == "__main__":
    # For local testing only. When submitting to the portal, comment this out.
    try:
        from robin_logistics import LogisticsEnvironment  # type: ignore
    except Exception:  # pragma: no cover
        from robin_logistics_env import LogisticsEnvironment  # type: ignore

    env = LogisticsEnvironment()
    solution = solver(env)

    print("Validating solution ...")
    is_valid, msg, summary = env.validate_solution_complete(solution)
    print(f"Validation: {is_valid} - {msg}")

    if is_valid:
        env.reset_all_state()
        ok, exec_msg = env.execute_solution(solution)
        print(f"Execution: {ok} - {exec_msg}")
        stats = env.get_solution_statistics(solution)
        print("\n--- Metrics ---")
        print(
            f"Orders served: {stats.get('unique_orders_served', 0)}/"
            f"{stats.get('total_orders', 0)}"
        )
        print(f"Total distance: {stats.get('total_distance', 0):.2f} km")
        print("\n--- Summary ---")
        print(summary)
    else:
        print("Solution invalid in this scenario. Iterate on solver(env).")
