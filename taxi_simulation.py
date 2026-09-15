"""
Simulations built on top of the SPPTW-MTTC planner in taxi_code.py.

  1. Animated playback  - aircraft move across the taxiway network along
                          their planned, conflict-free trajectories.
  2. Monte Carlo study  - random traffic scenarios at increasing traffic
                          levels, comparing SPPTW-MTTC against a simple
                          "fixed shortest route + hold at gate" baseline.

Run:
    python taxi_simulation.py              # live animation window
    python taxi_simulation.py montecarlo   # Monte Carlo results table
    python taxi_simulation.py all          # both
"""

import random
import statistics
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from taxi_code import (
    NODE_MIN_DWELL,
    NOMINAL_SPEED,
    ReservationTable,
    build_toy_airport,
    commit_trajectory,
    edge_traversal_params,
    plan_trajectory,
    validate_conflict_free,
)


# ============================================================
# SHARED: plan a batch of aircraft
# ============================================================
def plan_all(graph, aircraft_list, gate_nodes, planner="spptw"):
    """Plans aircraft in order (first-come, first-served priority) and
    commits each one. Returns (reservations, results) where results maps
    flight_id -> dict(ready_time, timed_chain) for successfully planned flights."""
    reservations = ReservationTable()
    results = {}
    for flight_id, start, goal, ready_time in aircraft_list:
        if planner == "spptw":
            timed_chain, _ = plan_trajectory(graph, reservations, start, goal, ready_time, gate_nodes)
        else:
            timed_chain = plan_fixed_route_gate_hold(graph, reservations, start, goal, ready_time)
        if timed_chain is None:
            continue
        commit_trajectory(reservations, flight_id, timed_chain, gate_nodes)
        results[flight_id] = {"ready_time": ready_time, "start": start, "timed_chain": timed_chain}
    return reservations, results


# ============================================================
# BASELINE: fixed shortest route, all waiting done at the gate
# ============================================================
def free_flow_chain(graph, route, pushback_time):
    """Zone chain at nominal speed with minimum dwell - no en-route waiting."""
    chain = [(route[0], pushback_time)]
    t = pushback_time
    for i, (u, v) in enumerate(zip(route, route[1:])):
        if i > 0:
            t += NODE_MIN_DWELL
        chain.append((ReservationTable.edge_zone(u, v), t))
        t += edge_traversal_params(graph[u][v]["weight"])[0]
        chain.append((v, t))
    return chain


def free_flow_time(graph, start, goal):
    route = nx.shortest_path(graph, start, goal, weight="weight")
    chain = free_flow_chain(graph, route, 0.0)
    return chain[-1][1]


def _fits(reservations, zone, s, e):
    return any(w_s <= s and e <= w_e for w_s, w_e in reservations.free_time_windows(zone))


def plan_fixed_route_gate_hold(graph, reservations, start, goal, ready_time,
                               step=1.0, max_hold=3600.0):
    """Delay pushback in `step` increments until the whole shortest route
    is free at nominal speed. No rerouting, no waiting on taxiways."""
    route = nx.shortest_path(graph, start, goal, weight="weight")
    t = ready_time
    while t <= ready_time + max_hold:
        chain = free_flow_chain(graph, route, t)
        # skip the gate (index 0) and the destination (not reserved)
        if all(_fits(reservations, chain[i][0], chain[i][1], chain[i + 1][1])
               for i in range(1, len(chain) - 1)):
            return chain
        t += step
    return None


# ============================================================
# 1. ANIMATED PLAYBACK
# ============================================================
def position_at(t, timed_chain, pos):
    """Aircraft (x, y) at time t. Chain alternates node, edge, node, ...:
    a node entry means 'at this junction until the next entry's time';
    an edge entry means 'moving from the previous node to the next node'."""
    for i in range(0, len(timed_chain) - 2, 2):
        node, _t_node = timed_chain[i]
        _edge, t_edge = timed_chain[i + 1]
        nxt, t_next = timed_chain[i + 2]
        if t < t_edge:
            return pos[node]
        if t < t_next:
            f = (t - t_edge) / max(t_next - t_edge, 1e-9)
            return (1 - f) * pos[node] + f * pos[nxt]
    return pos[timed_chain[-1][0]]


def status_at(t, flight):
    chain = flight["timed_chain"]
    if t < flight["ready_time"]:
        return "not ready"
    if t < chain[0][1]:
        return "holding at gate"
    if t < chain[-1][1]:
        zone = next(z for z, zt in reversed(chain) if zt <= t)
        label = zone if isinstance(zone, str) else "-".join(sorted(zone))
        return f"taxiing ({label})"
    return "at runway"


def animate(graph, results, gate_nodes, goal, fps=20, sim_speed=10.0, save_path=None):
    """sim_speed = simulated seconds per real second."""
    pos = {n: np.array(p) for n, p in nx.spring_layout(graph, seed=7).items()}
    flight_ids = list(results)
    palette = plt.cm.tab10.colors
    colors = {fid: palette[i % len(palette)] for i, fid in enumerate(flight_ids)}

    t_end = max(f["timed_chain"][-1][1] for f in results.values()) + 10
    dt = sim_speed / fps
    frames = np.arange(0.0, t_end + dt, dt)

    fig, (ax, ax_txt) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [3, 1.3]})
    node_colors = ["#4C72B0" if n in gate_nodes else "#C44E52" if n == goal else "#8C8C8C"
                   for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=700)
    # labels sit below the nodes so aircraft markers don't hide them
    label_pos = {n: p + np.array([0.0, -0.13]) for n, p in pos.items()}
    nx.draw_networkx_labels(graph, label_pos, ax=ax, font_size=8, font_weight="bold")
    base_edges = nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#BBBBBB", width=3)
    edge_list = list(graph.edges)
    ax.set_axis_off()
    ax.margins(0.15)

    markers = {fid: ax.plot([], [], "o", ms=16, color=colors[fid], mec="black", mew=1.5, zorder=5)[0]
               for fid in flight_ids}
    tags = {fid: ax.text(0, 0, fid.split("_")[-1], fontsize=7, ha="center", va="center",
                         color="white", fontweight="bold", zorder=6)
            for fid in flight_ids}
    clock = fig.suptitle("", fontsize=12)

    ax_txt.set_axis_off()
    status_texts = {fid: ax_txt.text(0.0, 0.9 - 0.12 * i, "", fontsize=9, color=colors[fid],
                                     fontweight="bold", transform=ax_txt.transAxes)
                    for i, fid in enumerate(flight_ids)}

    def edge_occupant(u, v, t):
        zone = ReservationTable.edge_zone(u, v)
        for fid, f in results.items():
            chain = f["timed_chain"]
            for i in range(len(chain) - 1):
                if chain[i][0] == zone and chain[i][1] <= t < chain[i + 1][1]:
                    return fid
        return None

    def update(t):
        clock.set_text(f"Taxi simulation   T = {t:6.1f} s")
        edge_colors = []
        for u, v in edge_list:
            occ = edge_occupant(u, v, t)
            edge_colors.append(colors[occ] if occ else "#BBBBBB")
        base_edges.set_color(edge_colors)

        for fid, f in results.items():
            visible = f["ready_time"] <= t <= f["timed_chain"][-1][1] + 3
            x, y = position_at(t, f["timed_chain"], pos) if visible else (np.nan, np.nan)
            markers[fid].set_data([x], [y])
            tags[fid].set_position((x, y))
            tags[fid].set_visible(visible)
            status_texts[fid].set_text(f"{fid}: {status_at(t, f)}")
        return [clock, base_edges, *markers.values(), *tags.values(), *status_texts.values()]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    plt.tight_layout()
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved {save_path}")
    return anim


def run_animation(save_path=None):  # pass "taxi_animation.gif" to also save a GIF
    print("--- Animated playback ---")
    G = build_toy_airport()
    gate_nodes = {"Gate_A", "Gate_B"}
    goal = "Runway"
    aircraft_list = [
        ("Flight_101", "Gate_A", goal, 0),
        ("Flight_202", "Gate_B", goal, 5),
        ("Flight_303", "Gate_A", goal, 8),
    ]
    reservations, results = plan_all(G, aircraft_list, gate_nodes)
    violations = validate_conflict_free(reservations)
    print(f"Planned {len(results)}/{len(aircraft_list)} flights, {len(violations)} separation violations")
    return animate(G, results, gate_nodes, goal, save_path=save_path)


# ============================================================
# 2. MONTE CARLO STUDY
# ============================================================
def random_scenario(rng, n_aircraft, gates, goal, time_window):
    ready = sorted(rng.uniform(0, time_window) for _ in range(n_aircraft))
    return [(f"F{i + 1:02d}", rng.choice(gates), goal, round(t, 1)) for i, t in enumerate(ready)]


def summarize(graph, aircraft_list, results):
    delays, holds, taxis = [], [], []
    for fid, start, goal, _ready in aircraft_list:
        if fid not in results:
            continue
        f = results[fid]
        chain = f["timed_chain"]
        pushback, completion = chain[0][1], chain[-1][1]
        holds.append(pushback - f["ready_time"])
        taxis.append(completion - pushback)
        delays.append(completion - f["ready_time"] - free_flow_time(graph, start, goal))
    return {
        "planned": len(results),
        "mean_delay": statistics.mean(delays) if delays else float("nan"),
        "mean_hold": statistics.mean(holds) if holds else float("nan"),
        "mean_taxi": statistics.mean(taxis) if taxis else float("nan"),
    }


def run_monte_carlo(traffic_levels=(2, 4, 6, 8, 10, 12), runs_per_level=30,
                    time_window=300.0, seed=42, save_path="monte_carlo_results.png"):
    print("--- Monte Carlo study ---")
    G = build_toy_airport()
    gate_nodes = {"Gate_A", "Gate_B"}
    gates = sorted(gate_nodes)
    goal = "Runway"
    rng = random.Random(seed)

    planners = {"SPPTW-MTTC": "spptw", "Fixed route + gate hold": "baseline"}
    table = {name: {k: [] for k in ("mean_delay", "mean_hold", "mean_taxi",
                                    "success", "plan_ms", "violations")}
             for name in planners}

    for n in traffic_levels:
        scenarios = [random_scenario(rng, n, gates, goal, time_window) for _ in range(runs_per_level)]
        for name, key in planners.items():
            per_run = []
            violations = 0
            t0 = time.perf_counter()
            for aircraft_list in scenarios:
                reservations, results = plan_all(G, aircraft_list, gate_nodes, planner=key)
                violations += len(validate_conflict_free(reservations))
                per_run.append(summarize(G, aircraft_list, results))
            elapsed_ms = (time.perf_counter() - t0) * 1000 / runs_per_level
            row = table[name]
            row["mean_delay"].append(statistics.mean(r["mean_delay"] for r in per_run))
            row["mean_hold"].append(statistics.mean(r["mean_hold"] for r in per_run))
            row["mean_taxi"].append(statistics.mean(r["mean_taxi"] for r in per_run))
            row["success"].append(100 * sum(r["planned"] for r in per_run) / (n * runs_per_level))
            row["plan_ms"].append(elapsed_ms)
            row["violations"].append(violations)

    header = f"{'aircraft':>8} | {'planner':<24} | {'delay s':>8} | {'hold s':>7} | {'taxi s':>7} | {'planned %':>9} | {'ms/run':>7} | {'viol.':>5}"
    print(header)
    print("-" * len(header))
    for i, n in enumerate(traffic_levels):
        for name in planners:
            r = table[name]
            print(f"{n:>8} | {name:<24} | {r['mean_delay'][i]:8.1f} | {r['mean_hold'][i]:7.1f} | "
                  f"{r['mean_taxi'][i]:7.1f} | {r['success'][i]:9.1f} | {r['plan_ms'][i]:7.1f} | "
                  f"{r['violations'][i]:>5}")

    # fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # styles = {"SPPTW-MTTC": ("#55A868", "o"), "Fixed route + gate hold": ("#8C8C8C", "s")}
    # for name, (color, marker) in styles.items():
    #     r = table[name]
    #     axes[0].plot(traffic_levels, r["mean_delay"], marker=marker, color=color, label=name)
    #     axes[1].plot(traffic_levels, r["mean_hold"], marker=marker, color=color, label=name)
    #     axes[2].plot(traffic_levels, r["mean_taxi"], marker=marker, color=color, label=name)
    # for ax, title in zip(axes, ("Mean delay vs free-flow (s)", "Mean gate hold (s)", "Mean taxi time (s)")):
    #     ax.set_title(title)
    #     ax.set_xlabel(f"Aircraft per {time_window:.0f} s")
    #     ax.grid(linestyle="--", alpha=0.4)
    # axes[0].legend(fontsize=9)
    # fig.suptitle(f"Monte Carlo: {runs_per_level} random scenarios per traffic level")
    # plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=150)
    #     print(f"Saved {save_path}")
    return table


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "animate"
    anim = None
    if mode in ("all", "animate"):
        anim = run_animation()
    if mode in ("all", "montecarlo"):
        run_monte_carlo()
    try:
        plt.show()
    except Exception as e:
        print(f"(Could not open an interactive window: {e}. Output files were still saved.)")
