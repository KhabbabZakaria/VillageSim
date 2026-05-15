"""
logger.py
---------
Writes simulation statistics to sim_log.csv at a fixed sim-time interval.
Designed for post-run analysis and graph plotting.

Usage (after a run):
    import pandas as pd
    df = pd.read_csv("sim_log.csv", comment="#")
    df.plot(x="sim_time_s", y="population")
"""

from __future__ import annotations

import csv
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import World

LOG_FILE     = "sim_log.csv"
LOG_INTERVAL = 100.0  # sim-seconds between rows (≈25 real-s at default 4× speed)

_COLUMNS = [
    "sim_time_s",
    "real_time_s",
    "generation",
    "population",
    "males",
    "females",
    "children",
    "total_born",
    "total_deaths",
    "deaths_old_age",
    "deaths_starvation",
    "deaths_hunting",
    "deaths_troll",
    "deaths_farming",
]


class SimLogger:

    def __init__(self, path: str = LOG_FILE) -> None:
        self.path      = path
        self._next_log = 0.0

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            f.write(f"# VillageSim stats — started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            csv.writer(f).writerow(_COLUMNS)

    # ── called every frame ────────────────────────────────────────────────────

    def update(self, world: "World") -> None:
        if world.sim_time >= self._next_log:
            self._append(world)
            self._next_log = world.sim_time + LOG_INTERVAL

    # ── called once on quit ───────────────────────────────────────────────────

    def finalize(self, world: "World") -> None:
        self._append(world)

    # ── internals ─────────────────────────────────────────────────────────────

    def _append(self, world: "World") -> None:
        al  = world.alive()
        dbc = world.deaths_by_cause
        row = [
            round(world.sim_time,  1),
            round(world.real_time, 2),
            world.generation,
            len(al),
            sum(1 for v in al if v.gender == "M"),
            sum(1 for v in al if v.gender == "F"),
            sum(1 for v in al if v.is_child),
            world.total_born,
            world.total_deaths,
            dbc.get("old age",            0),
            dbc.get("starvation",         0),
            dbc.get("hunting mishap",     0),
            dbc.get("troll attack",       0),
            dbc.get("farming exhaustion", 0),
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
