"""
simulation.py
-------------
Core simulation logic.

Classes
~~~~~~~
Villager
    An agent with its own Actor-Critic brain, health / risk attributes,
    lifecycle state (child → adult → death), and reproduction capability.

World
    Owns all Villagers, drives the simulation step, handles reproduction,
    repopulation, zone-based reward/risk, event logging, and particle FX.
"""

from __future__ import annotations

import math
import random
import threading
from typing import List, Optional, Tuple

import numpy as np

import config as cfg
from drl import ActorCritic, ReplayBuffer
from helpers import (
    dist_vv, clamp, jitter, zone_for_point, fmt_time,
    spawn_particles, update_particles, Particle,
)


# ── uid counter ───────────────────────────────────────────────────────────────

_uid_counter: int = 0


def _next_uid() -> int:
    global _uid_counter
    _uid_counter += 1
    return _uid_counter


def reset_uid() -> None:
    global _uid_counter
    _uid_counter = 0


# ── villager ──────────────────────────────────────────────────────────────────

class Villager:
    """
    A single villager agent.

    State vector (9 dims)
    ~~~~~~~~~~~~~~~~~~~~~
    [health/100, risk/100, age/max_age, is_child, has_kid,
     kids_nearby, mate_nearby, sex_cooldown>0, current_action/4]

    Actions
    ~~~~~~~
    0 = Rest (village)
    1 = Farm West
    2 = Farm East
    3 = Hunt South
    4 = Hunt North
    """

    def __init__(
        self,
        gender: str,
        x: float,
        y: float,
        parent_a: Optional["Villager"] = None,
        parent_b: Optional["Villager"] = None,
        is_child: bool = False,
    ) -> None:
        self.uid    = _next_uid()
        self.gender = gender                 # 'M' or 'F'
        self.x      = float(x)
        self.y      = float(y)
        self.tx     = float(x)              # target x
        self.ty     = float(y)              # target y

        # ── lifecycle ─────────────────────────────────────────────────────────
        self.alive     = True
        self.is_child  = is_child
        self.is_adult  = not is_child
        self.age       = 0.0
        self.max_age   = random.uniform(cfg.MAX_AGE_MIN, cfg.MAX_AGE_MAX)
        self.death_reason: str = ""
        self.last_damage_source: str = "starvation"  # tracks dominant damage cause
        self.max_health = 100.0
        self.health = self.max_health

        # ── stats ─────────────────────────────────────────────────────────────
        self.health         = 100.0
        self.risk           = 0.0
        self.total_reward   = 0.0
        self.instant_reward = 0.0

        # ── action state ──────────────────────────────────────────────────────
        self.action       = 0
        self.action_timer = 0.0
        self.action_dur   = random.uniform(cfg.ACTION_DUR_MIN, cfg.ACTION_DUR_MAX)

        # ── name ──────────────────────────────────────────────────────────────
        pool      = cfg.NAMES_M if gender == "M" else cfg.NAMES_F
        self.name = random.choice(pool)

        # ── brain ─────────────────────────────────────────────────────────────
        if parent_a is not None and parent_b is not None:
            self.brain: ActorCritic = parent_a.brain.breed(parent_b.brain)
        else:
            self.brain = ActorCritic()

        # ── RL phase ──────────────────────────────────────────────────────────
        self.train_phase  = True
        self.phase_timer  = 0.0
        self.last_state:  Optional[np.ndarray] = None
        self.last_action: int = 0

        # ── parenting ─────────────────────────────────────────────────────────
        self.parent_ids: List[int] = [parent_a.uid, parent_b.uid] if parent_a else []
        self.has_kid       = False
        self.kid_bond_t    = 0.0

        # ── hunger ────────────────────────────────────────────────────────────
        self.hunger_timer    = 0.0   # sim-seconds since last meal
        self.hunger_eat_timer = 0.0  # continuous sim-seconds spent in farm/hunt

        # ── social ────────────────────────────────────────────────────────────
        self.sex_cooldown = 0.0

        # ── visuals ───────────────────────────────────────────────────────────
        self.trail: List[Tuple[float, float]] = []
        self.emotion       = ""
        self.emotion_timer = 0.0
        self.flash_timer   = 0.0
        self.speed = random.uniform(cfg.VILLAGER_SPEED_MIN, cfg.VILLAGER_SPEED_MAX)

        # ── training metrics ──────────────────────────────────────────────────
        self.last_train_metrics: Optional[dict] = None

        # ── LLM collective memory ─────────────────────────────────────────────
        self.event_log:       List[dict]  = []   # observations this cycle
        self.hp_snapshot:     float       = 100.0
        self.hp_snapshot_timer: float     = 0.0
        self.bubble_text:     str         = ""   # speech bubble shown during discussion
        self.bubble_timer:    float       = 0.0

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return self.name + ("*" if self.is_child else "")

    @property
    def color(self) -> Tuple[int, int, int]:
        if not self.alive:    return cfg.C_DEAD
        if self.is_child:     return cfg.C_CHILD
        if self.gender == "M": return cfg.C_MALE
        return cfg.C_FEMALE

    @property
    def radius(self) -> int:
        return 7 if self.is_child else 10

    # ── state construction ────────────────────────────────────────────────────

    def get_state(self, kids_near: float, mate_near: float, is_night: float) -> np.ndarray:
        # hunger_frac: 0 = just ate, 1 = at threshold, >1 = starving
        hunger_frac = clamp(self.hunger_timer / max(cfg.HUNGER_THRESHOLD, 1.0), 0.0, 2.0) / 2.0
        return np.array([
            self.health / 100.0,
            clamp(self.risk, 0, 100) / 100.0,
            self.age / max(self.max_age, 1.0),
            1.0 if self.is_child  else 0.0,
            1.0 if self.has_kid   else 0.0,
            kids_near,
            mate_near,
            1.0 if self.sex_cooldown > 0 else 0.0,
            self.action / 4.0,
            hunger_frac,   # dim 9 — how urgently the agent needs to eat
            is_night,      # dim 10 — danger signal for troll hours
        ], dtype=np.float32)

    # ── reward / risk per action ──────────────────────────────────────────────

    def action_rr(self, action: int) -> Tuple[float, float]:
        """Return (reward_rate, risk_rate) per sim-second for *action*."""
        pm = cfg.FARM_PARENT_MULT if self.has_kid else 1.0
        hm = cfg.HUNT_PARENT_MULT if self.has_kid else 1.0
        if action == 0:
            return cfg.REST_REWARD, 0.0
        if action in (1, 2):
            risk = cfg.FARM_RISK_BASE * (2.0 if self.has_kid else 1.0)
            return cfg.FARM_REWARD * pm, risk
        # hunt
        risk = cfg.HUNT_PARENT_RISK if self.has_kid else cfg.HUNT_RISK_BASE
        return cfg.HUNT_REWARD * hm, risk


# ── troll ─────────────────────────────────────────────────────────────────────

class Troll:
    """
    Immortal night creature. No health bar.
    Stays in the mountains during the day; roams and attacks during the
    middle 20 seconds of each 30-second night.
    """
    __slots__ = ("x", "y", "home_x", "home_y", "tx", "ty", "target_timer")

    def __init__(self, x: float, y: float) -> None:
        self.x = self.home_x = float(x)
        self.y = self.home_y = float(y)
        self.tx = float(x)
        self.ty = float(y)
        self.target_timer = 0.0


# ── world ─────────────────────────────────────────────────────────────────────

class World:
    """
    Simulation engine.

    Drives per-frame updates, zone effects, RL decisions, reproduction,
    repopulation, particle FX, and the event log.
    """

    def __init__(self) -> None:
        self.villagers:  List[Villager] = []
        self.particles:  List[Particle] = []
        self.log:        List[Tuple[str, str]] = []  # (message, kind)
        self.sim_time    = 0.0
        self.paused      = False
        self.speed       = cfg.SIM_SPEED_DEFAULT
        self.generation  = 1
        self.total_born  = 0
        self.total_deaths = 0
        self.repop_count  = 0
        self.selected_uid: Optional[int] = None
        # Collective experience pool — fed by every villager's full replay buffer on death
        self.species_pool: ReplayBuffer = ReplayBuffer(cfg.AC_SPECIES_POOL_SIZE)
        # Trolls — immortal night creatures
        self.trolls: List[Troll] = []
        self.real_time: float = 0.0
        # LLM discussion phase state
        self.discussion_active:  bool        = False
        self.discussion_timer:   float       = 0.0
        self.discussion_pause:   float       = 0.0
        self._was_night:         bool        = False
        self._was_twilight:      bool        = False
        self.pending_lessons:    List[dict]  = []
        self.last_lessons:       List[dict]  = []
        self._lesson_lock = threading.Lock()
        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        reset_uid()
        self.villagers.clear()
        self.particles.clear()
        self.log.clear()
        self.sim_time     = 0.0
        self.generation   = 1
        self.total_born   = 4
        self.total_deaths = 0
        self.repop_count  = 0
        self.selected_uid    = None
        self.species_pool    = ReplayBuffer(cfg.AC_SPECIES_POOL_SIZE)
        self.real_time       = 0.0
        self.trolls          = [Troll(x, y) for x, y in cfg.TROLL_HOMES]
        self.discussion_active = False
        self.discussion_timer  = 0.0
        self.discussion_pause  = 0.0
        self._was_night        = False
        self._was_twilight     = False
        self._last_discussion_obs: List[dict] = []
        with self._lesson_lock:
            self.pending_lessons.clear()
        self.last_lessons = []
        # Clear the LLM log on every new game
        with open("llm_log.txt", "w") as f:
            f.write("=== VillageSim LLM Controller Log ===\n\n")
        self.deaths_by_cause = {"old age": 0, "starvation": 0, "hunting mishap": 0, "troll attack": 0, "farming exhaustion": 0}
        cx, cy = cfg.ACTION_TARGETS[0]
        offsets = [(-38, -28, "M"), (+38, +28, "M"), (-38, +28, "F"), (+38, -28, "F")]
        for dx, dy, g in offsets:
            self._spawn(g, cx + dx, cy + dy)
        self._log("Village founded. Four souls begin their journey.", "birth")

    # ── public helpers ────────────────────────────────────────────────────────

    def alive(self) -> List[Villager]:
        return [v for v in self.villagers if v.alive]

    def adults(self) -> List[Villager]:
        return [v for v in self.alive() if v.is_adult]

    def fmt_time(self) -> str:
        return fmt_time(self.sim_time)

    def get_stats(self) -> dict:
        al = self.alive()
        return dict(
            alive       = len(al),
            males       = sum(1 for v in al if v.gender == "M"),
            females     = sum(1 for v in al if v.gender == "F"),
            children    = sum(1 for v in al if v.is_child),
            generation  = self.generation,
            total_born  = self.total_born,
            total_deaths= self.total_deaths,
            repops      = self.repop_count,
            sim_time    = self.fmt_time(),
            speed       = self.speed,
        )

    # ── time helpers ──────────────────────────────────────────────────────────

    def _time_of_day(self) -> str:
        period = cfg.DAY_DURATION + cfg.NIGHT_DURATION
        phase  = self.sim_time % period
        if phase >= cfg.DAY_DURATION:
            return "night"
        elif phase >= cfg.DAY_DURATION - cfg.TWILIGHT_DURATION:
            return "twilight"
        elif phase < cfg.DAWN_DURATION:
            return "dawn"
        return "day"

    def _night_danger(self) -> float:
        """Continuous 0.0–1.0 danger level: ramps up during twilight, down during dawn."""
        period = cfg.DAY_DURATION + cfg.NIGHT_DURATION
        phase  = self.sim_time % period
        if phase >= cfg.DAY_DURATION:
            return 1.0
        elif phase >= cfg.DAY_DURATION - cfg.TWILIGHT_DURATION:
            return (phase - (cfg.DAY_DURATION - cfg.TWILIGHT_DURATION)) / cfg.TWILIGHT_DURATION
        elif phase < cfg.DAWN_DURATION:
            return 1.0 - (phase / cfg.DAWN_DURATION)
        return 0.0

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, dt: float) -> None:
        if self.paused:
            return
        real_dt = min(dt, 0.05)
        self.real_time += real_dt
        dt = real_dt * self.speed

        # ── discussion resolution (always checked; timeout uses wall-clock) ──
        if self.discussion_active:
            self.discussion_pause += real_dt
            with self._lesson_lock:
                lessons_ready = len(self.pending_lessons) > 0
            if lessons_ready or self.discussion_pause >= cfg.DISCUSSION_TIMEOUT:
                with self._lesson_lock:
                    lessons = self.pending_lessons[:]
                    self.pending_lessons.clear()
                alive = self.alive()
                self._apply_lessons(lessons, alive)
                self.last_lessons = lessons
                self.discussion_active = False
                for v in alive:
                    v.bubble_timer = 0.0
            else:
                return  # freeze: sim_time and all physics don't advance

        # ── physics (only runs when not in discussion) ────────────────────────
        self.sim_time += dt
        alive = self.alive()

        # Force re-evaluation at twilight start (head home warning) and night start
        tod = self._time_of_day()
        is_night_now    = tod == "night"
        is_twilight_now = tod == "twilight"
        if (is_night_now and not self._was_night) or (is_twilight_now and not self._was_twilight):
            for v in alive:
                if zone_for_point(v.x, v.y) != "village":
                    v.action_timer = 0.0
        self._was_night    = is_night_now
        self._was_twilight = is_twilight_now

        for v in alive:
            self._update_villager(v, dt, alive)

        self._update_trolls(dt, alive)

        # discussion trigger (sim-time so it scales with speed)
        self.discussion_timer += dt
        if self.discussion_timer >= cfg.DISCUSSION_INTERVAL:
            self.discussion_timer = 0.0
            self._start_discussion(alive)

        # Repopulation guard — maintain at least one adult of each gender
        for gender in ("M", "F"):
            adults_g = [v for v in self.adults() if v.gender == gender]
            kids_g   = [v for v in self.alive() if v.gender == gender and v.is_child]
            if not adults_g and not kids_g:
                self._repopulate(gender)

        update_particles(self.particles, dt)

    # ── per-villager update ───────────────────────────────────────────────────

    def _update_villager(self, v: Villager, dt: float, alive: List[Villager]) -> None:
        # Full freeze during discussion — only tick the speech bubble
        if self.discussion_active:
            if v.bubble_timer > 0:
                v.bubble_timer = max(0.0, v.bubble_timer - dt)
                if v.bubble_timer <= 0:
                    v.bubble_text = ""
            return

        v.age           += dt
        v.sex_cooldown   = max(0.0, v.sex_cooldown - dt)
        v.emotion_timer  = max(0.0, v.emotion_timer - dt)
        v.flash_timer    = max(0.0, v.flash_timer - dt)
        v.phase_timer   += dt

        # Kid bond countdown
        if v.kid_bond_t > 0:
            v.kid_bond_t -= dt
            if v.kid_bond_t <= 0:
                v.has_kid = False

        # Adulthood transition
        if v.is_child and v.age >= cfg.ADULTHOOD_AGE:
            v.is_child = False
            v.is_adult = True
            self._log(f"{v.name} has come of age.", "birth")
            v.emotion = "★"
            v.emotion_timer = 4.0
            spawn_particles(self.particles, v.x, v.y, cfg.C_ACCENT, 20)

        # Death conditions
        if v.age >= v.max_age:
            self._kill(v, "old age")
            return
        if v.risk >= 100.0:
            self._kill(v, "hunting mishap")
            return
        if v.health <= 0.0:
            self._kill(v, v.last_damage_source)
            return

        # Passive health drain
        v.health = max(0.0, v.health - dt * cfg.PASSIVE_HEALTH_DRAIN)

        # Hunger — no max(0.0) clamp so health can go negative.
        # Village regen is guarded below and won't rescue a negative-health villager.
        v.hunger_timer += dt
        if v.hunger_timer > cfg.HUNGER_THRESHOLD:
            v.last_damage_source = "starvation"
            v.health -= dt * cfg.HUNGER_DRAIN

        # HP snapshot — detect significant drops and gains for LLM observation log
        v.hp_snapshot_timer += dt
        if v.hp_snapshot_timer >= cfg.HP_SNAPSHOT_INTERVAL:
            drop = v.hp_snapshot - v.health
            gain = v.health - v.hp_snapshot
            if drop >= cfg.HP_DROP_THRESHOLD:
                v.event_log.append({
                    "event":         "sudden_hp_drop",
                    "subject":       "self",
                    "nearby":        self._nearby_context(v, alive),
                    "time":          self._time_of_day(),
                    "action":        cfg.ACTION_NAMES[v.action],
                    "health_before": round(v.hp_snapshot),
                    "health_after":  round(v.health),
                    "hunger":        "hungry" if v.hunger_timer > cfg.HUNGER_THRESHOLD else "fed",
                })
            elif gain >= cfg.HP_GAIN_THRESHOLD:
                v.event_log.append({
                    "event":         "hp_restored",
                    "subject":       "self",
                    "nearby":        self._nearby_context(v, alive),
                    "time":          self._time_of_day(),
                    "action":        cfg.ACTION_NAMES[v.action],
                    "health_before": round(v.hp_snapshot),
                    "health_after":  round(v.health),
                })
            v.hp_snapshot       = v.health
            v.hp_snapshot_timer = 0.0

        # Phase switching (train ↔ test)
        phase_dur = cfg.AC_TRAIN_DURATION if v.train_phase else cfg.AC_TEST_DURATION
        if v.phase_timer >= phase_dur:
            v.phase_timer = 0.0
            if v.train_phase:
                v.last_train_metrics = v.brain.train()
            v.train_phase = not v.train_phase

        # Action decision
        v.action_timer += dt
        if v.action_timer >= v.action_dur:
            v.action_timer = 0.0
            self._decide_action(v, alive)

        # Movement toward target
        dx = v.tx - v.x
        dy = v.ty - v.y
        d  = math.hypot(dx, dy)
        if d > 2.0:
            step = v.speed * dt * 55.0
            v.x += dx / d * min(step, d)
            v.y += dy / d * min(step, d)

        # Trail
        v.trail.append((v.x, v.y))
        if len(v.trail) > cfg.TRAIL_MAX_LEN:
            v.trail.pop(0)

        # Zone-based reward / risk
        self._apply_zone_effects(v, dt)

        # Children gain health from nearby parent
        if v.is_child:
            parent = next(
                (p for p in alive if p.uid in v.parent_ids
                 and dist_vv(v, p) < cfg.CHILD_NEARBY_DIST),
                None,
            )
            if parent:
                v.health = min(100.0, v.health + dt * cfg.PARENT_FEED_RATE)

        # Reproduction (male initiates proximity check) — suppressed at population cap
        if (v.is_adult and v.gender == "M" and v.sex_cooldown <= 0
                and len(alive) < cfg.MAX_VILLAGERS):
            for f in alive:
                if (f.is_adult and f.gender == "F"
                        and f.sex_cooldown <= 0
                        and dist_vv(v, f) < cfg.SEX_PROXIMITY):
                    self._do_sex(v, f)
                    break

    # ── troll update ──────────────────────────────────────────────────────────

    def _update_trolls(self, dt: float, alive: List[Villager]) -> None:
        if self.discussion_active:
            return
        period     = cfg.DAY_DURATION + cfg.NIGHT_DURATION
        phase      = self.sim_time % period
        out_start  = cfg.DAY_DURATION + cfg.TROLL_EMERGE_DELAY   # 65 s
        out_end    = period - cfg.TROLL_RETURN_EARLY              # 85 s
        is_out     = out_start <= phase < out_end

        vx, vy, vw, vh = cfg.ZONES["village"]
        vm      = 80    # exclusion margin for target picking
        vcx     = vx + vw / 2          # village centre x
        vcy     = vy + vh / 2          # village centre y
        v_infl  = vw / 2 + vm + 50     # repulsion influence radius (~190 px)
        world_w = cfg.WINDOW_W - cfg.PANEL_WIDTH

        def _safe_target(nx, ny):
            """True if point is outside the village + margin."""
            return not (vx - vm <= nx <= vx + vw + vm and
                        vy - vm <= ny <= vy + vh + vm)

        for t in self.trolls:
            if is_out:
                t.target_timer -= dt
                dx, dy = t.tx - t.x, t.ty - t.y
                if t.target_timer <= 0 or math.hypot(dx, dy) < 8:
                    for _ in range(60):
                        nx = random.uniform(30, world_w - 30)
                        ny = random.uniform(30, 550)
                        if _safe_target(nx, ny):
                            t.tx, t.ty = nx, ny
                            break
                    t.target_timer = random.uniform(
                        cfg.TROLL_TARGET_DUR_MIN, cfg.TROLL_TARGET_DUR_MAX
                    )
            else:
                t.tx, t.ty = t.home_x, t.home_y

            # Move with smooth village repulsion — no hard redirects
            dx, dy = t.tx - t.x, t.ty - t.y
            d = math.hypot(dx, dy)
            if d > 2.0:
                step = cfg.TROLL_SPEED * dt * 55.0
                mx, my = dx / d, dy / d   # base direction

                # Repulsion from village centre, strengthens as troll gets closer
                rx, ry  = t.x - vcx, t.y - vcy
                rdist   = math.hypot(rx, ry)
                if 0 < rdist < v_infl:
                    strength = ((v_infl - rdist) / v_infl) ** 2 * 3.0
                    mx += (rx / rdist) * strength
                    my += (ry / rdist) * strength
                    nd = math.hypot(mx, my)
                    if nd > 0:
                        mx /= nd
                        my /= nd

                t.x += mx * min(step, d)
                t.y += my * min(step, d)

            # Hard wall: push troll outside village rect if it slipped in
            wall = 6   # pixels of clearance from village edge
            if (vx - wall < t.x < vx + vw + wall and
                    vy - wall < t.y < vy + vh + wall):
                # find nearest edge and eject to it
                d_left  = t.x - (vx - wall)
                d_right = (vx + vw + wall) - t.x
                d_top   = t.y - (vy - wall)
                d_bot   = (vy + vh + wall) - t.y
                m = min(d_left, d_right, d_top, d_bot)
                if m == d_left:   t.x = vx - wall
                elif m == d_right: t.x = vx + vw + wall
                elif m == d_top:   t.y = vy - wall
                else:              t.y = vy + vh + wall

            # Attack villagers that are nearby but NOT inside the village
            if is_out:
                for v in alive:
                    if zone_for_point(v.x, v.y) == "village":
                        continue
                    if math.hypot(t.x - v.x, t.y - v.y) < cfg.TROLL_ATTACK_RANGE:
                        v.last_damage_source = "troll attack"
                        v.health = max(0.0, v.health - dt * cfg.TROLL_ATTACK_DRAIN)

    # ── zone effects ─────────────────────────────────────────────────────────

    def _apply_zone_effects(self, v: Villager, dt: float) -> None:
        zone      = zone_for_point(v.x, v.y)
        rwd, rsk  = v.action_rr(v.action)
        danger    = self._night_danger()   # 0.0 = safe day, 1.0 = full night
        is_night  = danger >= 1.0
        is_hungry = v.hunger_timer > cfg.HUNGER_THRESHOLD

        if zone == "crop":
            v.last_damage_source = "farming exhaustion"
            v.health = max(0.0, v.health - cfg.FARM_HEALTH_DRAIN * dt)
            v.risk   = max(0.0, v.risk   - dt * cfg.FARM_RISK_DRAIN)
            # Farm reward scales down as danger rises: full at day, near-zero at night
            day_mult = 3.0 if is_hungry else 1.0
            night_mult = 0.35 if is_hungry else 0.08
            mult = day_mult * (1.0 - danger) + night_mult * danger
            v.instant_reward = rwd * 0.5 * mult
            v.hunger_eat_timer += dt
            if v.hunger_eat_timer >= cfg.HUNGER_EAT_DURATION:
                if v.hunger_timer > cfg.HUNGER_THRESHOLD:
                    v.event_log.append({
                        "event":   "hunger_satisfied",
                        "subject": "self",
                        "nearby":  self._nearby_context(v, self.alive()),
                        "time":    self._time_of_day(),
                        "action":  cfg.ACTION_NAMES[v.action],
                    })
                v.hunger_timer     = 0.0
                v.hunger_eat_timer = 0.0

        elif zone == "hunt":
            v.last_damage_source = "hunting mishap"
            v.health = max(0.0, v.health - cfg.HUNT_HEALTH_DRAIN * dt)
            v.risk   = min(100.0, v.risk + rsk * dt)
            # Hunt reward scales down as danger rises
            v.instant_reward = rwd * (0.35 * (1.0 - danger) + 0.05 * danger)
            v.hunger_eat_timer += dt
            if v.hunger_eat_timer >= cfg.HUNGER_EAT_DURATION:
                v.hunger_timer     = 0.0
                v.hunger_eat_timer = 0.0

        elif zone == "village":
            if v.health > 0:
                v.health = min(100.0, v.health + dt * cfg.VILLAGE_HEALTH_REGEN)
            v.risk   = max(0.0, v.risk - dt * cfg.REST_RISK_DRAIN)
            # Village reward rises with danger but dampened when hungry — resting hungry at night feels bad
            village_mult = 0.2 if is_hungry else 1.0
            v.instant_reward = cfg.REST_REWARD * (0.4 + 3.1 * danger) * village_mult
            v.hunger_eat_timer = 0.0

        else:   # wild
            if v.health > 0:
                v.health = min(100.0, v.health + dt * cfg.WILD_HEALTH_REGEN)
            v.risk   = max(0.0, v.risk - dt * 1.0)
            v.instant_reward = 0.05 - 0.85 * danger
            v.hunger_eat_timer = 0.0

        # Outside-village penalty scales with danger (0 at midday, full -1.5 at midnight)
        if danger > 0 and zone != "village":
            v.instant_reward -= 1.5 * danger

        # Universal hunger penalty — scales with danger so starvation at night feels worse
        if is_hungry:
            v.instant_reward -= (1.0 + 2.0 * danger)

        # Small survival bonus every tick — living longer is intrinsically rewarded
        v.instant_reward += 0.15

        v.total_reward += v.instant_reward * dt

    # ── RL decision ───────────────────────────────────────────────────────────

    def _decide_action(self, v: Villager, alive: List[Villager]) -> None:
        kids_near, mate_near = self._nearby_info(v, alive)
        state = v.get_state(kids_near, mate_near, self._night_danger())

        probs, _, _ = v.brain.forward(state)
        action = (
            v.brain.sample_action(probs)
            if v.train_phase
            else v.brain.greedy_action(probs)
        )
        v.action = action

        tx, ty = cfg.ACTION_TARGETS[action]
        v.tx, v.ty = jitter(tx, ty, cfg.ACTION_TARGET_JITTER)
        v.tx = clamp(v.tx, 10, cfg.WINDOW_W - 10)
        v.ty = clamp(v.ty, 10, cfg.WINDOW_H - 10)
        v.action_dur = random.uniform(cfg.ACTION_DUR_MIN, cfg.ACTION_DUR_MAX)

        if v.last_state is not None:
            v.brain.store(v.last_state, v.last_action, v.instant_reward, state, False)

        v.last_state  = state
        v.last_action = action

    def _nearby_info(
        self, v: Villager, alive: List[Villager]
    ) -> Tuple[float, float]:
        near     = [o for o in alive if o.uid != v.uid and dist_vv(v, o) < 70]
        kids     = any(o.is_child for o in near)
        has_mate = any(o.is_adult and o.gender != v.gender for o in near)
        return (1.0 if kids else 0.0), (1.0 if has_mate else 0.0)

    # ── reproduction ─────────────────────────────────────────────────────────

    def _do_sex(self, male: Villager, female: Villager) -> None:
        male.sex_cooldown   = cfg.SEX_COOLDOWN
        female.sex_cooldown = cfg.SEX_COOLDOWN
        male.risk    = min(100.0, male.risk   + cfg.SEX_RISK_PENALTY)
        female.risk  = min(100.0, female.risk + cfg.SEX_RISK_PENALTY)
        male.health  = min(100.0, male.health + cfg.SEX_HEALTH_BONUS)
        female.health = min(100.0, female.health + cfg.SEX_HEALTH_BONUS)

        male.has_kid    = True; female.has_kid    = True
        male.kid_bond_t = cfg.KID_BOND_DURATION
        female.kid_bond_t = cfg.KID_BOND_DURATION

        male.emotion   = "♥"; male.emotion_timer   = 6.0
        female.emotion = "♥"; female.emotion_timer = 6.0
        spawn_particles(self.particles, male.x, male.y, cfg.C_LOG_SEX, 25)

        child_gender = random.choice(["M", "F"])
        cx, cy = jitter(male.x, male.y, 25)
        child  = self._spawn(child_gender, cx, cy, male, female, is_child=True)
        # Seed child with recent species wisdom — avoids inheriting stale old-gen behaviors
        if len(self.species_pool) > 0:
            child.brain.memory.seed(
                self.species_pool.sample_recent_random(cfg.AC_INHERIT_SPECIES, cfg.AC_SPECIES_RECENT_WINDOW)
            )
        self.total_born += 1
        self._log(f"{male.name} & {female.name} → child {child.name}!", "sex")
        child.emotion = "★"; child.emotion_timer = 6.0

    # ── repopulation ─────────────────────────────────────────────────────────

    def _repopulate(self, gender: str) -> None:
        cx, cy = cfg.ACTION_TARGETS[0]
        x, y   = jitter(cx, cy, 45)
        # Always spawn as a child — must grow up before reproducing
        v = self._spawn(gender, x, y, is_child=True)
        # Seed wanderer with recent species wisdom — avoids inheriting stale old-gen behaviors
        if len(self.species_pool) > 0:
            v.brain.memory.seed(
                self.species_pool.sample_recent_random(cfg.AC_INHERIT_SPECIES, cfg.AC_SPECIES_RECENT_WINDOW)
            )
        self.total_born   += 1
        self.repop_count  += 1
        self.generation   += 1
        label = "boy" if gender == "M" else "girl"
        self._log(f"A wandering {label} joins the village: {v.name}.", "repop")
        v.emotion = "?"; v.emotion_timer = 5.0
        spawn_particles(self.particles, x, y, cfg.C_LOG_REPOP, 22)

    # ── spawn ─────────────────────────────────────────────────────────────────

    def _spawn(
        self,
        gender: str,
        x: float, y: float,
        parent_a: Optional[Villager] = None,
        parent_b: Optional[Villager] = None,
        is_child: bool = False,
    ) -> Villager:
        v = Villager(gender, x, y, parent_a, parent_b, is_child)
        self.villagers.append(v)
        return v

    # ── LLM collective memory ─────────────────────────────────────────────────

    def _nearby_context(self, v: Villager, alive: List[Villager]) -> List[str]:
        """Return a short list of strings describing what is near villager v."""
        context: List[str] = []
        period   = cfg.DAY_DURATION + cfg.NIGHT_DURATION
        is_night = (self.sim_time % period) >= cfg.DAY_DURATION

        zone = zone_for_point(v.x, v.y)
        context.append(zone if zone else "wild")

        # Only flag troll if villager is outside the village — inside is always safe
        if is_night and zone != "village":
            for t in self.trolls:
                if math.hypot(t.x - v.x, t.y - v.y) < cfg.TROLL_ATTACK_RANGE * 1.5:
                    context.append("troll")
                    break

        for other in alive:
            if other is v:
                continue
            if math.hypot(other.x - v.x, other.y - v.y) < cfg.WITNESS_RANGE:
                context.append(cfg.ACTION_NAMES[other.action])
                break

        return list(dict.fromkeys(context))  # deduplicate, preserve order

    def _start_discussion(self, alive: List[Villager]) -> None:
        """Freeze villagers, collect observations, fire async LLM call."""
        all_obs: List[dict] = []
        seen_deaths: set = set()   # deduplicate witnessed_death by (subject, cause, time)

        for v in alive:
            # Every villager always contributes a status snapshot
            all_obs.append({
                "event":   "status_report",
                "subject": v.name,
                "nearby":  self._nearby_context(v, alive),
                "time":    self._time_of_day(),
                "action":  cfg.ACTION_NAMES[v.action],
                "health":  round(v.health),
                "hunger":  "hungry" if v.hunger_timer > cfg.HUNGER_THRESHOLD else "fed",
            })
            # Plus any critical events they witnessed — deduplicate deaths
            if v.event_log:
                last = v.event_log[-1]
                v.bubble_text  = f"{last['event'].replace('_', ' ')} @ {last['time']}"
                v.bubble_timer = cfg.DISCUSSION_TIMEOUT + 3.0
                for obs in v.event_log:
                    if obs["event"] == "witnessed_death":
                        key = (obs.get("subject"), obs.get("cause"), obs.get("time"))
                        if key in seen_deaths:
                            continue
                        seen_deaths.add(key)
                    all_obs.append(obs)
                v.event_log.clear()
            else:
                v.bubble_text  = f"{cfg.ACTION_NAMES[v.action]} • hp {round(v.health)}"
                v.bubble_timer = cfg.DISCUSSION_TIMEOUT + 3.0

        self.discussion_active      = True
        self.discussion_pause       = 0.0
        self._last_discussion_obs   = all_obs
        self._log("Villagers pause to share their experiences...", "")

        def _llm_thread() -> None:
            from model import call_controller
            lessons = call_controller(all_obs)
            with self._lesson_lock:
                self.pending_lessons.extend(lessons)

        threading.Thread(target=_llm_thread, daemon=True).start()

    def _apply_lessons(self, lessons: List[dict], alive: List[Villager]) -> None:
        """Convert LLM lessons into synthetic transitions and seed into all replay buffers."""
        if not lessons:
            self._log("Discussion ended — no lessons extracted.", "")
            return

        synthetic: List[dict] = []
        lesson_transitions: List[List[dict]] = []  # per-lesson, for logging

        for lesson in lessons:
            bad_actions  = lesson.get("bad_actions",  [])
            good_actions = lesson.get("good_actions", [])
            is_night_raw = lesson.get("is_night", None)
            is_night_pin = float(is_night_raw) if is_night_raw is not None else None
            hunger_pin   = lesson.get("hunger_frac", None)
            health_pin   = lesson.get("health_frac", None)
            reward       = float(lesson.get("reward", -1.0))
            fatal        = bool(lesson.get("fatal",   False))

            # Safety layer: enforce night-action consistency regardless of LLM output.
            # At night, actions 1-4 outside the village are always dangerous.
            if is_night_pin == 1.0:
                good_actions = [a for a in good_actions if a == 0]   # only Rest is safe
                bad_actions  = [a for a in bad_actions  if a != 0]   # never penalise Rest
            # Farming/hunting should never be taught as daytime-agnostic good actions —
            # if is_night is unpinned and good_actions includes outdoor actions, pin to day.
            if is_night_pin is None and any(a in good_actions for a in [1, 2, 3, 4]):
                is_night_pin = 0.0
            # Hunger-based rest penalties must only fire during the day.
            # At night, Rest is the only troll-safe action — penalising it for hunger would
            # directly contradict "rest at night" good_action lessons.
            if is_night_pin is None and 0 in bad_actions and lesson.get("hunger_frac") is not None:
                is_night_pin = 0.0

            # bad_actions always get negative reward, good_actions always positive
            action_reward_pairs = (
                [(a, -abs(reward))   for a in bad_actions]  +
                [(a,  abs(reward))   for a in good_actions]
            )

            this_lesson_transitions: List[dict] = []
            for action, r in action_reward_pairs:
                for _ in range(cfg.LLM_TRANSITIONS_PER_LESSON):
                    state = np.array([
                        health_pin   if health_pin   is not None else np.random.uniform(0.5, 0.9),  # health
                        np.random.uniform(0.1, 0.3),                            # risk
                        np.random.uniform(0.2, 0.7),                            # age
                        np.random.uniform(0.0, 1.0),                            # is_child
                        np.random.uniform(0.0, 1.0),                            # has_kid
                        np.random.uniform(0.0, 1.0),                            # kids_near
                        np.random.uniform(0.0, 1.0),                            # mate_near
                        np.random.uniform(0.0, 1.0),                            # sex_cooldown
                        action / 4.0,                                           # current_action
                        hunger_pin   if hunger_pin   is not None else np.random.uniform(0.0, 1.0),
                        is_night_pin if is_night_pin is not None else float(np.random.random() < 0.5),
                    ], dtype=np.float32)

                    # Add small noise only to unpinned dims
                    noise = np.random.randn(11).astype(np.float32) * 0.04
                    if is_night_pin is not None: noise[10] = 0.0
                    if hunger_pin   is not None: noise[9]  = 0.0
                    noise[8] = 0.0
                    if health_pin   is not None: noise[0]  = 0.0
                    state = np.clip(state + noise, 0.0, 1.0)

                    next_state    = state.copy()
                    next_state[0] = float(np.clip(state[0] + (0.05 if r > 0 else -0.25), 0.0, 1.0))

                    t = dict(s=state, a=int(action), r=r, ns=next_state, done=fatal)
                    synthetic.append(t)
                    this_lesson_transitions.append(t)

            lesson_transitions.append(this_lesson_transitions)

            text = lesson.get("lesson", "")
            if text:
                self._log(f"Lesson: {text}", "")

        n_generated = len(synthetic)
        max_synthetic = int(cfg.AC_MAX_MEMORY * cfg.LLM_MAX_SYNTHETIC_FRAC)
        if len(synthetic) > max_synthetic:
            idxs = np.random.choice(len(synthetic), max_synthetic, replace=False)
            synthetic = [synthetic[i] for i in sorted(idxs)]

        n_synthetic = len(synthetic)
        for v in alive:
            real_before = len(v.brain.memory)
            for t in synthetic:
                v.brain.memory.push(t["s"], t["a"], t["r"], t["ns"], t["done"])
            real_surviving = max(0, cfg.AC_MAX_MEMORY - n_synthetic)

        species_before = len(self.species_pool)
        self.species_pool.seed(synthetic)

        self._log(f"[LLM] {len(lessons)} lessons → {n_generated} generated, {n_synthetic} injected (cap={max_synthetic}).", "")
        self._write_llm_log(lessons, lesson_transitions, n_generated, n_synthetic, max_synthetic, real_before, real_surviving, species_before)

    def _write_llm_log(self, lessons: List[dict], lesson_transitions: List[List[dict]],
                       n_generated: int, n_synthetic: int, max_synthetic: int,
                       real_before: int, real_surviving: int, species_before: int) -> None:
        import json as _json
        STATE_LABELS = [
            "health", "risk", "age", "is_child", "has_kid",
            "kids_near", "mate_near", "sex_cooldown", "current_action",
            "hunger_frac", "is_night",
        ]
        try:
            with open("llm_log.txt", "a") as f:
                f.write(f"--- Discussion @ t={self.real_time:.1f}s  (sim {self.fmt_time()}) ---\n")
                f.write("OBSERVATIONS:\n")
                for obs in self._last_discussion_obs:
                    f.write(f"  {_json.dumps(obs)}\n")
                f.write("LESSONS:\n")
                if lessons:
                    for i, (lesson, transitions) in enumerate(zip(lessons, lesson_transitions), 1):
                        f.write(f"  [{i}] {_json.dumps(lesson)}\n")
                        f.write(f"      VECTORS ({len(transitions)} injected):\n")
                        for t in transitions:
                            vec_str = ", ".join(
                                f"{label}={val:.2f}"
                                for label, val in zip(STATE_LABELS, t["s"])
                            )
                            f.write(f"        action={t['a']} reward={t['r']:+.1f} [{vec_str}]\n")
                else:
                    f.write("  (none returned)\n")
                real_pct = max(0, real_surviving) / cfg.AC_MAX_MEMORY * 100
                overwritten = max(0, real_before + n_synthetic - cfg.AC_MAX_MEMORY)
                capped = n_generated - n_synthetic
                f.write(f"BUFFER STATS (per villager):\n")
                f.write(f"  Real transitions before : {real_before} / {cfg.AC_MAX_MEMORY}\n")
                f.write(f"  Synthetic generated     : {n_generated}  (cap={max_synthetic})\n")
                f.write(f"  Synthetic injected      : {n_synthetic}" + (f"  *** CAPPED ({capped} dropped) ***" if capped > 0 else "") + "\n")
                f.write(f"  Real overwritten        : {overwritten}\n")
                f.write(f"  Real surviving          : {real_surviving} ({real_pct:.0f}% of buffer)\n")
                f.write(f"  Species pool before     : {species_before} / {cfg.AC_SPECIES_POOL_SIZE}\n")
                f.write("\n")
        except OSError as e:
            print(f"[LLM log] Could not write: {e}")

    # ── kill ──────────────────────────────────────────────────────────────────

    def _kill(self, v: Villager, reason: str) -> None:
        # Store terminal transition so the agent learns what led to death
        if v.last_state is not None:
            terminal_reward = (
                cfg.PREMATURE_DEATH_PENALTY if reason != "old age"
                else v.instant_reward
            )
            v.brain.store(v.last_state, v.last_action, terminal_reward, v.last_state, True)

        # Donate the villager's entire lifetime experience to the species pool
        # so future generations can learn from it at birth
        self.species_pool.seed(v.brain.memory._buf)

        # Notify nearby witnesses — only for avoidable deaths, not natural old age
        if reason != "old age":
            for other in self.alive():
                if other is not v and math.hypot(other.x - v.x, other.y - v.y) < cfg.WITNESS_RANGE:
                    other.event_log.append({
                        "event":   "witnessed_death",
                        "subject": v.name,
                        "nearby":  self._nearby_context(v, self.alive()),
                        "time":    self._time_of_day(),
                        "action":  cfg.ACTION_NAMES[v.action],
                        "cause":   reason,
                    })

        v.alive        = False
        v.death_reason = reason
        self.total_deaths += 1
        self.deaths_by_cause[reason] = self.deaths_by_cause.get(reason, 0) + 1
        spawn_particles(self.particles, v.x, v.y, cfg.C_DEAD, 18)
        self._log(f"{v.name} ({v.gender}) died: {reason}.", "death")

    # ── log ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str, kind: str = "") -> None:
        self.log.insert(0, (f"[{self.fmt_time()}] {msg}", kind))
        if len(self.log) > cfg.LOG_KEEP:
            self.log.pop()