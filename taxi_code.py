"""
Conflict-free, time-based airport taxi trajectory planning implementation of the SPPTW-MTTC algorithm.

"""

import heapq
import itertools

import matplotlib.pyplot as plt
import networkx as nx

SEPARATION_BUFFER_S = 15.0      # minimum safety gap between different aircraft
                                 
HORIZON = 100000.0               # a sufficiently large "end of time" for the
                                 # last free time window of any zone

NOMINAL_SPEED = 10.0             # m/s : normal taxiing speed (used for t_min)
MIN_SPEED = 4.0                  # m/s : slowest allowed crawl speed (used for
                                 # t_max = distance / MIN_SPEED)
NODE_MIN_DWELL = 5.0             # s  : unimpeded time to cross/turn at a junction
NODE_MAX_DWELL = 15.0            # s  : maximum time allowed to linger at a
                                 # junction before it counts as blocking traffic

_counter = itertools.count()     # heap tie-breaker


# ============================================================
# 1. RESERVATION TABLE + FREE TIME WINDOWS
# ============================================================
class ReservationTable:
    """
    Tracks, for every zone (a graph node OR a graph edge/segment), which
    time intervals are already claimed by committed aircraft. Free time
    windows
    """

    def __init__(self):
        self.reservations = {}  # zone -> list[(start, end, aircraft_id)]

    @staticmethod
    def edge_zone(u, v):
        return frozenset((u, v))

    def reserve(self, zone, start, end, aircraft_id):
        self.reservations.setdefault(zone, []).append((start, end, aircraft_id))

    def free_time_windows(self, zone, buffer=SEPARATION_BUFFER_S, horizon=HORIZON):
        """Fig. 1 of the paper: the free time windows of a zone are what's
        left after removing every reserved interval (expanded by the
        safety buffer on both sides)."""
        occ = self.reservations.get(zone)
        if not occ:
            return [(0.0, horizon)]
        occ_sorted = sorted(occ, key=lambda r: r[0])
        merged = [[occ_sorted[0][0], occ_sorted[0][1]]]
        for (s, e, _aid) in occ_sorted[1:]:
            if s <= merged[-1][1] + buffer:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        windows = []
        prev_end = 0.0
        for (s, e) in merged:
            window_end = s - buffer
            if window_end > prev_end:
                windows.append((prev_end, window_end))
            prev_end = max(prev_end, e + buffer)
        if prev_end < horizon:
            windows.append((prev_end, horizon))
        return windows


# ============================================================
# 2. ZONE TRAVERSAL PARAMETERS (t_min / t_max, Section 2.2)
# ============================================================
def node_traversal_params():
    return NODE_MIN_DWELL, NODE_MAX_DWELL


def edge_traversal_params(distance):
    t_min = distance / NOMINAL_SPEED
    t_max = distance / MIN_SPEED
    return t_min, t_max


def pass_through_zone(interval, window_end, t_min, t_max):
    """
    Section 2.2 / 3.1: given a feasible arrival interval at a zone and the
    end of the free time window it belongs to, compute the feasible EXIT
    interval, respecting both the unimpeded traversal time (t_min) and the
    maximum traversal time (t_max). Returns None if no exit is feasible
    within the maximum traversal time allowed - this is the maximum
    traversal time constraint actually biting.
    """
    exit_start = interval[0] + t_min
    exit_end = min(interval[1] + t_max, window_end)
    if exit_start > exit_end:
        return None
    return (exit_start, exit_end)


def arrival_options(exit_interval, zone, reservations, buffer=SEPARATION_BUFFER_S):
    """
    For every free time window of `zone` that overlaps the feasible exit
    interval from the previous zone, produce a candidate arrival interval.
    Multiple free time windows can each produce a separate candidate -
    this is exactly why the search needs to track a SET of interval-based
    candidates rather than a single earliest time.
    """
    options = []
    for (w_s, w_e) in reservations.free_time_windows(zone, buffer):
        a_start = max(exit_interval[0], w_s)
        a_end = min(exit_interval[1], w_e)
        if a_start <= a_end:
            options.append(((a_start, a_end), (w_s, w_e)))
    return options


# ============================================================
# 3. SEARCH STATE (a "temporal node": zone + free time window + interval)
# ============================================================
class SearchState:
    __slots__ = ("zone", "window", "interval", "taxi_time", "zone_chain", "alive")

    def __init__(self, zone, window, interval, taxi_time, zone_chain):
        self.zone = zone
        self.window = window          # (w_start, w_end) - the free time window this belongs to
        self.interval = interval      # (a_start, a_end) - feasible arrival interval
        self.taxi_time = taxi_time    # accumulated unimpeded (moving/dwelling) time
        self.zone_chain = zone_chain  # [(zone, interval, t_min), ...] start..here, for the backward pass
        self.alive = True             # False once dominated/superseded

    def cost(self, heuristic):
        # Section 3: cost = completion time so far (ts of interval) + taxi
        # time so far + heuristic estimate of the remaining journey.
        return self.interval[0] + self.taxi_time + heuristic


def dominates(a, b):
    """(a) dominates (b) if a's interval is a superset of b's AND a costs
    less or the same. Only a dominated candidate may be safely discarded."""
    return (a.interval[0] <= b.interval[0] and a.interval[1] >= b.interval[1])


# ============================================================
# 4. THE SPPTW-MTTC SEARCH (Fig. 4 of the paper)
# ============================================================
def dijkstra_lower_bound(graph, goal):
    return nx.single_source_dijkstra_path_length(graph, goal, weight="weight")


def plan_trajectory(graph, reservations, start, goal, ready_time, gate_nodes=frozenset()):
    dist_to_goal = dijkstra_lower_bound(graph, goal)

    def heuristic(zone):
        # cost combines completion time AND taxi time, both of which grow
        # by (at least) the remaining travel time in the best case, so
        # the admissible lower bound on the REMAINING cost is 2x the
        # remaining unimpeded travel time, not 1x.
        d = dist_to_goal.get(zone, float("inf"))
        if d == float("inf"):
            return float("inf")
        return 2.0 * (d / NOMINAL_SPEED)

    start_state = SearchState(
        zone=start, window=(0.0, HORIZON), interval=(ready_time, HORIZON),
        taxi_time=0.0, zone_chain=[(start, (ready_time, HORIZON), 0.0)],
    )

    labels = {}  # (zone, window) -> list[SearchState]
    labels[(start, start_state.window)] = [start_state]

    open_heap = [(start_state.cost(heuristic(start)), next(_counter), start_state)]

    while open_heap:
        f, _, state = heapq.heappop(open_heap)
        if not state.alive:
            continue
        if state.zone == goal:
            return backward_pass(state), state

        for neighbor in graph.neighbors(state.zone):
            _expand(graph, reservations, state, neighbor, goal, gate_nodes, heuristic, labels, open_heap)

    return None, None


def _expand(graph, reservations, state, neighbor, goal, gate_nodes, heuristic, labels, open_heap):
    # 1. exit the CURRENT zone (node dwell, unless it's a gate - gates are
    #    dedicated parking, not a contested/limited-dwell resource)
    if state.zone in gate_nodes:
        exit_from_current = state.interval
    else:
        t_min_node, t_max_node = node_traversal_params()
        exit_from_current = pass_through_zone(state.interval, state.window[1], t_min_node, t_max_node)
        if exit_from_current is None:
            return  # maximum traversal time constraint blocks leaving this zone at all

    # 2. arrive at the EDGE zone (the taxiway segment itself)
    distance = graph[state.zone][neighbor]["weight"]
    t_min_edge, t_max_edge = edge_traversal_params(distance)
    edge_zone = ReservationTable.edge_zone(state.zone, neighbor)

    for (edge_interval, edge_window) in arrival_options(exit_from_current, edge_zone, reservations):
        # 3. traverse the edge zone
        exit_from_edge = pass_through_zone(edge_interval, edge_window[1], t_min_edge, t_max_edge)
        if exit_from_edge is None:
            continue  # would need to wait on the segment longer than allowed

        # 4. arrive at the NEXT node zone
        for (node_interval, node_window) in arrival_options(exit_from_edge, neighbor, reservations):
            new_taxi_time = state.taxi_time + (
                0.0 if state.zone in gate_nodes else NODE_MIN_DWELL
            ) + t_min_edge
            new_chain = state.zone_chain + [
                (edge_zone, edge_interval, t_min_edge),
                (neighbor, node_interval, NODE_MIN_DWELL if neighbor != goal else 0.0),
            ]
            candidate = SearchState(neighbor, node_window, node_interval, new_taxi_time, new_chain)
            _insert_with_dominance(candidate, labels, open_heap, heuristic)


def _insert_with_dominance(candidate, labels, open_heap, heuristic):
    key = (candidate.zone, candidate.window)
    existing = labels.setdefault(key, [])
    cand_cost = candidate.cost(heuristic(candidate.zone))

    for other in existing:
        if not other.alive:
            continue
        other_cost = other.cost(heuristic(other.zone))
        if dominates(other, candidate) and other_cost <= cand_cost:
            return  # candidate is dominated - discard it, matches Fig. 4 lines 14-19
    # candidate survives - remove any existing labels it dominates
    for other in existing:
        if other.alive and dominates(candidate, other) and cand_cost <= other.cost(heuristic(other.zone)):
            other.alive = False  # matches Fig. 4 lines 20-24

    existing.append(candidate)
    heapq.heappush(open_heap, (cand_cost, next(_counter), candidate))


# ============================================================
# 5. BACKWARD PASS (end of Section 3): push every zone's time as LATE
#    as possible while staying feasible, to minimise idle taxi time
# ============================================================
def backward_pass(final_state):
    chain = final_state.zone_chain
    n = len(chain)
    times = [None] * n
    # destination fixed at the earliest feasible completion time
    times[-1] = chain[-1][1][0]
    for k in range(n - 2, -1, -1):
        interval_k = chain[k][1]
        t_min_k = chain[k][2]  # cost to advance FROM zone k TO zone k+1
        candidate = times[k + 1] - t_min_k
        # clamp into the interval: "cannot meet the interval" -> use its ending
        # time when too late; clamping below also absorbs float round-off
        # (e.g. 240.8999... vs 240.9), which must not jump to the interval end
        times[k] = min(max(candidate, interval_k[0]), interval_k[1])
    return [(chain[i][0], times[i]) for i in range(n)]


def commit_trajectory(reservations, aircraft_id, timed_chain, gate_nodes=frozenset()):
    """
    Commits the concrete per-zone times from the backward pass. Each zone
    is reserved from its own time to the NEXT zone's time (i.e. how long
    the aircraft actually occupies that zone), except the destination,
    which needs no further reservation once reached. Gates are excluded,
    matching their treatment as non-contested resources throughout.
    """
    for i in range(len(timed_chain) - 1):
        zone, t_start = timed_chain[i]
        _, t_end = timed_chain[i + 1]
        if zone in gate_nodes:
            continue
        reservations.reserve(zone, t_start, t_end, aircraft_id)


# ============================================================
# 6. VALIDATION
# ============================================================
def validate_conflict_free(reservations, buffer=SEPARATION_BUFFER_S):
    violations = []
    for zone, windows in reservations.reservations.items():
        windows_sorted = sorted(windows, key=lambda w: w[0])
        for i in range(len(windows_sorted)):
            for j in range(i + 1, len(windows_sorted)):
                s1, e1, id1 = windows_sorted[i]
                s2, e2, id2 = windows_sorted[j]
                if id1 == id2:
                    continue
                if s2 >= e1 + buffer:
                    break
                violations.append((zone, id1, id2, s1, e1, s2, e2))
    return violations


# ============================================================
# 7. PLOTS 
# ============================================================
def _zone_label(zone):
    return zone if isinstance(zone, str) else "-".join(sorted(zone))


def plot_network(graph, gate_nodes, goal, save_path=None):
    pos = nx.spring_layout(graph, seed=7)
    node_colors = []
    for n in graph.nodes:
        if n in gate_nodes:
            node_colors.append("#4C72B0")
        elif n == goal:
            node_colors.append("#C44E52")
        else:
            node_colors.append("#8C8C8C")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    nx.draw(graph, pos, ax=ax, with_labels=True, node_color=node_colors, node_size=1600,
             font_size=9, font_color="white", font_weight="bold", edge_color="#555555", width=2)
    edge_labels = {(u, v): f'{d["weight"]:.0f} m' for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title("Taxiway Network (Zones = Nodes + Edges)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")


def plot_gantt(reservations, gate_nodes, save_path=None):
    """Built directly from reservations.reservations - whatever the run
    actually produced, gates excluded (they're non-contested)."""
    zones = [z for z in reservations.reservations.keys() if z not in gate_nodes]
    # sort rows by the earliest reservation start time, for a readable top-to-bottom flow
    zones.sort(key=lambda z: min(s for s, e, a in reservations.reservations[z]))

    aircraft_ids = sorted({a for windows in reservations.reservations.values() for _, _, a in windows})
    palette = plt.cm.tab10.colors
    colors = {aid: palette[i % len(palette)] for i, aid in enumerate(aircraft_ids)}

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(zones) + 2))
    for i, zone in enumerate(zones):
        for (start, end, aid) in reservations.reservations[zone]:
            duration = max(end - start, 0.8)
            ax.barh(i, duration, left=start, height=0.55, color=colors[aid], edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels([_zone_label(z) for z in zones], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title("Committed Time-Window Reservations (SPPTW-MTTC, Conflict-Free)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[aid]) for aid in aircraft_ids]
    ax.legend(handles, aircraft_ids, loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")


def plot_gate_hold_vs_taxi(flight_summaries, save_path=None):
    """flight_summaries: list of dicts with keys
    aircraft_id, ready_time, pushback_time, completion_time."""
    ids = [f["aircraft_id"] for f in flight_summaries]
    gate_wait = [f["pushback_time"] - f["ready_time"] for f in flight_summaries]
    taxi_time = [f["completion_time"] - f["pushback_time"] for f in flight_summaries]
    completion = [f["completion_time"] for f in flight_summaries]

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(ids) + 2))
    y = range(len(ids))
    ax.barh(y, gate_wait, color="#8C8C8C", label="Waiting at gate (engines off)")
    ax.barh(y, taxi_time, left=gate_wait, color="#55A868", label="Actual taxiing (moving)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ids)
    ax.invert_yaxis()
    ax.set_xlabel("Time since ready (s)")
    ax.set_title("Delayed Pushback Absorbs Waiting - Route Choice + Wait Placement")
    for i in range(len(ids)):
        ax.text(gate_wait[i] + taxi_time[i] + 1, i, f"complete @ {completion[i]:.0f}s", va="center", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, max(completion) + 20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")


# ============================================================
# 8. DEMO
# ============================================================
def build_toy_airport():
    G = nx.Graph()
    G.add_edge("Gate_A", "Int_1", weight=100)
    G.add_edge("Gate_B", "Int_1", weight=150)
    G.add_edge("Int_1", "Int_2", weight=200)
    G.add_edge("Int_2", "Int_3", weight=100)
    G.add_edge("Int_3", "Runway", weight=50)
    G.add_edge("Int_1", "Int_3", weight=400)
    return G


def run_demo():
    print("--- SPPTW-MTTC: Shortest Path with Time Windows + Max Traversal Time ---")
    G = build_toy_airport()
    reservations = ReservationTable()
    gate_nodes = {"Gate_A", "Gate_B"}
    goal = "Runway"

    aircraft_list = [
        ("Flight_101", "Gate_A", goal, 0),
        ("Flight_202", "Gate_B", goal, 5),
        ("Flight_303", "Gate_A", goal, 8),
    ]

    flight_summaries = []  # collected live, used by the plots below - never hardcoded
    sim_results = {}       # flight_id -> planned trajectory, used by the live simulation

    for flight_id, start, end, ready_time in aircraft_list:
        print(f"\nRouting {flight_id} (ready at T={ready_time}s)...")
        timed_chain, final_state = plan_trajectory(G, reservations, start, end, ready_time, gate_nodes)
        if timed_chain is None:
            print(f"  [FAILED] no feasible SPPTW-MTTC solution for {flight_id}")
            continue

        route = " -> ".join(z if isinstance(z, str) else "-".join(sorted(z)) for z, _t in timed_chain)
        pushback_time = timed_chain[0][1]
        completion_time = timed_chain[-1][1]
        taxi_time = completion_time - pushback_time
        print(f"  [PLANNED] {route}")
        print(f"  [PLANNED] pushback (leaves gate) at T={pushback_time:.1f}s")
        print(f"  [PLANNED] completion (runway) at T={completion_time:.1f}s")
        print(f"  [PLANNED] taxi time (actual moving time) = {taxi_time:.1f}s")

        commit_trajectory(reservations, flight_id, timed_chain, gate_nodes)
        flight_summaries.append({
            "aircraft_id": flight_id, "ready_time": ready_time,
            "pushback_time": pushback_time, "completion_time": completion_time,
        })
        sim_results[flight_id] = {"ready_time": ready_time, "start": start, "timed_chain": timed_chain}

    print("\n--- Final Reservation Schedule ---")
    for zone, windows in reservations.reservations.items():
        label = zone if isinstance(zone, str) else "-".join(sorted(zone))
        formatted = ", ".join(f"({s:.1f}-{e:.1f}, {a})" for s, e, a in sorted(windows))
        print(f"{label:16}: {formatted}")

    print("\n--- Validation ---")
    violations = validate_conflict_free(reservations)
    if violations:
        print(f"FAILED - {len(violations)} separation violations found:")
        for v in violations:
            print("  ", v)
    else:
        print("PASSED - no separation violations across all committed trajectories.")

    # print("\n--- Generating plots ---")
    # plot_network(G, gate_nodes, goal, save_path="network_diagram.png")
    # plot_gantt(reservations, gate_nodes, save_path="schedule_gantt.png")
    # if flight_summaries:
    #     plot_gate_hold_vs_taxi(flight_summaries, save_path="gate_hold_vs_taxi.png")

    print("\n--- Running live simulation (close the window to exit) ---")
    from taxi_simulation import animate  # imported here: taxi_simulation imports this module
    anim = animate(G, sim_results, gate_nodes, goal)  # keep a reference or the animation stops
    try:
        plt.show()
    except Exception as e:
        print(f"(Could not open an interactive window: {e}.)")


if __name__ == "__main__":
    run_demo()
