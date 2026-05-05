"""
stats_logger.py
---------------
Logs population and death statistics to a CSV file whenever the
simulation generation counter advances.

Load and plot with:
    import pandas as pd
    df = pd.read_csv("stats_log.csv")
    df.plot(x="generation", y=["population", "total_deaths"])
"""

from __future__ import annotations

import csv
import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import World

_LOG_INTERVAL = 100.0  # sim-seconds between rows

_FIELDS = [
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


class StatsLogger:
    """Appends one CSV row every 100 sim-seconds to *path*."""

    def __init__(self, path: str = "stats_log.csv") -> None:
        self.path           = path
        self._next_log_time = 0.0

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"# VillageSim stats — started {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            w.writerow(_FIELDS)

    def update(self, world: "World") -> None:
        """Call once per frame. Writes a row every 100 sim-seconds."""
        if world.sim_time < self._next_log_time:
            return
        self._next_log_time += _LOG_INTERVAL

        stats = world.get_stats()
        dbc   = world.deaths_by_cause

        row = [
            round(world.sim_time,  2),
            round(world.real_time, 2),
            world.generation,
            stats["alive"],
            stats["males"],
            stats["females"],
            stats["children"],
            stats["total_born"],
            stats["total_deaths"],
            dbc.get("old age",            0),
            dbc.get("starvation",         0),
            dbc.get("hunting mishap",     0),
            dbc.get("troll attack",       0),
            dbc.get("farming exhaustion", 0),
        ]

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
