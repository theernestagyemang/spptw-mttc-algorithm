# Conflict-Free Airport Taxi Planning (SPPTW-MTTC) - spptw-mttc-algorithm
Conflict-free, time-based airport taxi trajectory planning implementation of the SPPTW-MTTC algorithm.


A Python implementation of the **SPPTW-MTTC** algorithm (Shortest Path Problem with Time Windows and Maximum Traversal Time Constraints) for planning airport taxi routes, together with an animated simulation and a Monte Carlo study.

The planner finds, for each aircraft in turn, a route and timing from its gate to the runway that:

- keeps every taxiway segment and junction clear of other aircraft, with a safety buffer of 15 s,
- respects a maximum time an aircraft may wait on a segment or junction, and
- does as much of any waiting as possible **at the gate** (engines off) instead of on the taxiways.

## Files

| File | Purpose |
|---|---|
| `taxi_code.py` | The planner: reservation table, free time windows, time-window A* search, backward pass, validation, and the demo. |
| `taxi_simulation.py` | Simulations built on the planner: live animation and the Monte Carlo study. |

Both files must be in the same folder, because each one imports from the other.

## Requirements

- Python 3.9+
- `networkx`, `matplotlib`, `numpy`
- `pillow` (only needed to save the animation as a GIF)

```bash
pip install networkx matplotlib numpy pillow
```

## Running

Run the scripts from a terminal, or with VS Code's **Run Python File** button. A Jupyter or Interactive window shows the animation as a still image.

```bash
python taxi_code.py                      # plan the 3-flight demo, print the schedule, open the live simulation
python taxi_simulation.py                # open the live simulation only
python taxi_simulation.py montecarlo     # print the Monte Carlo results table
python taxi_simulation.py all            # both
```

### Demo output

`taxi_code.py` plans three flights on a small toy airport (2 gates, 3 junctions, 1 runway). It prints:

- each flight's route, pushback time, runway arrival time and taxi time,
- the final reservation schedule for every zone, and
- a validation result confirming no separation violations.

It then opens a window where the aircraft move along their planned trajectories. Each taxiway segment is highlighted in the colour of the aircraft using it, and a side panel shows each flight's status. Close the window to exit.

## How the algorithm works

1. **Zones.** Every junction (graph node) and every taxiway segment (graph edge) is a *zone* that only one aircraft may use at a time.
2. **Reservation table.** Planned aircraft reserve time intervals on each zone. The remaining gaps, shrunk by the separation buffer, are that zone's **free time windows**.
3. **Traversal times.** Each zone has a minimum time (moving at 10 m/s, or a 5 s turn at a junction) and a maximum time (crawling at 4 m/s, or a 15 s dwell at a junction).
4. **Search.** An A*-style search expands states of the form *(zone, free time window, feasible arrival interval)*. Because one zone can have several free windows, the search keeps a set of candidates and discards only those that are *dominated*: a later-starting, narrower interval with no lower cost.
5. **Backward pass.** Once the runway is reached, times are pushed as late as possible along the route. Waiting therefore moves back to the gate.
6. **Commit.** The resulting trajectory is reserved, and the next aircraft is planned around it (first-come, first-served).

## Monte Carlo study

`python taxi_simulation.py montecarlo` generates 30 random scenarios per traffic level (2–12 aircraft ready within 300 s). It compares SPPTW-MTTC against a baseline:

> **Fixed route + gate hold:** every aircraft takes its shortest route at nominal speed, and waits at the gate until the whole route is free.

Results on the toy airport (seed 42):

| Aircraft | SPPTW-MTTC delay | Baseline delay | SPPTW-MTTC taxi time | Baseline taxi time |
|---:|---:|---:|---:|---:|
| 2  | 1.5 s  | 2.0 s  | 62.3 s | 62.0 s |
| 6  | 9.9 s  | 17.2 s | 63.8 s | 62.8 s |
| 12 | 39.1 s | 71.3 s | 64.0 s | 62.7 s |

At 12 aircraft, SPPTW-MTTC roughly halves delay, at a cost of 1–2 s extra taxi time from occasionally using a longer route. Both methods produced zero separation violations and planned every aircraft. The toy airport is very small, so treat these numbers as illustrative.

## Configuration

Parameters are constants at the top of `taxi_code.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SEPARATION_BUFFER_S` | 15.0 s | Minimum time gap between different aircraft in the same zone |
| `NOMINAL_SPEED` | 10.0 m/s | Normal taxi speed |
| `MIN_SPEED` | 4.0 m/s | Slowest allowed speed |
| `NODE_MIN_DWELL` | 5.0 s | Time to cross or turn at a junction |
| `NODE_MAX_DWELL` | 15.0 s | Longest allowed wait at a junction |
| `HORIZON` | 100000 s | End of the last free time window |

To try a different airport, edit `build_toy_airport()` (edge weights are distances in metres). To try different flights, edit `aircraft_list` in `run_demo()`.

Simulation options:

- `animate(..., sim_speed=10.0)` sets how many simulated seconds pass per real second.
- `run_animation(save_path="taxi_animation.gif")` also saves a GIF.
- `run_monte_carlo(traffic_levels=..., runs_per_level=..., time_window=...)` sets the Monte Carlo study size.

## Optional figures

Static figures are commented out in `run_demo()` (`taxi_code.py`) and in `run_monte_carlo()` (`taxi_simulation.py`). Uncomment them to produce:

- `network_diagram.png`: the taxiway network
- `schedule_gantt.png`: reservations per zone over time
- `gate_hold_vs_taxi.png`: gate waiting vs taxiing per flight
- `monte_carlo_results.png`: delay, gate hold and taxi time vs traffic level

## Limitations

- Aircraft are planned one at a time in order, so earlier flights get priority. The result is conflict-free but not globally optimal.
- The simulation plays back the plan exactly. It does not model speed deviations or replanning.
- The toy network is undirected, and all aircraft go to a single runway.
