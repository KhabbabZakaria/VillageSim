# VillageSim — Actor-Critic DRL Ecosystem

A living village simulation where every villager is an independent **Actor-Critic reinforcement learning agent** built from scratch in pure NumPy. Villagers explore, learn, age, reproduce, and die. Children inherit neural weights and memories from both parents. Knowledge accumulates across generations through a shared **species memory pool**. A background **LLM Controller** (GPT-4o-mini) watches each day/night cycle, extracts survival lessons, and injects them as synthetic training transitions — giving the population collective memory that transcends individual lifespans. Emergent behaviour arises purely from the RL policies — no rules, no scripted AI.

---

## Screenshots

| Day | Night |
|-----|-------|
| ![Day](day.png) | ![Night](night.png) |

During the day villagers farm and rest. At night, trolls emerge from the mountains — villagers that venture outside the village take heavy damage and risk death.

---

## Quick Start

```bash
# 1. Copy and fill in your OpenAI key (required for LLM collective memory)
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

Requires **Python 3.10+** and **pygame 2.6+**.

The simulation runs fine without an `OPENAI_API_KEY` — the LLM Controller is silently skipped and the rest of the RL ecosystem operates normally.

---

## Controls

| Key / Input | Action |
|-------------|--------|
| `Space` | Pause / resume |
| `R` | Full reset |
| `+` / `=` | Increase simulation speed (up to 20×) |
| `-` | Decrease simulation speed |
| Drag speed slider | Set simulation speed 1×–20× directly with the mouse |
| `Left click` | Select a villager — shows live policy tooltip |
| `Esc` / `Q` | Quit |

---

## What's Happening on Screen

The **left panel** is the simulated world. The **right panel** is a live dashboard showing:

- **Population** — alive villager count and generation number
- **Deaths** — total deaths broken down by cause (Old Age, Starvation, Hunting, Troll Attack, Farming Exhaustion)
- **Events log** — births, deaths, reproduction events, and coming-of-age

Each villager displays:
- A coloured dot (blue = male adult, pink = female adult, yellow = child)
- A movement trail
- A health/risk bar on hover
- An emotion symbol for key life events

---

## How It Works

### The World

Five zones define the map:

| Zone | Activity | Health | Risk |
|------|----------|--------|------|
| Village (centre) | Rest — regens health | +3.0 hp/s | None |
| Farm West / East | Farming — feeds hunger | −0.1 hp/s | Low |
| Hunt South / North | Hunting — high reward, high risk | −3.5 hp/s | High |

Villagers outside all zones wander the wild, slowly losing health with no benefit.

### Day / Night Cycle

The cycle has four phases (sim-seconds; divide by 4 to get real-seconds at default 4× speed):

| Phase | Sim-seconds | Real-seconds (4×) | Notes |
|-------|-------------|-------------------|-------|
| Day | 160 | ≈40s | Safe for all activities |
| Twilight | 40 | ≈10s | Trolls not yet active — last safe window to farm before night |
| Night | 120 | ≈30s | Trolls patrol; any villager outside the village takes +12 hp/s damage |
| Dawn | 40 | ≈10s | Trolls retreating — safe to resume farming immediately |

A sun/moon indicator at the top-centre of the map tracks the current phase.
At night, a dark blue overlay fades in over the world.
**Trolls** emerge 20 sim-seconds into night and return 20 sim-seconds before dawn.

> **Twilight** is specifically taught to the LLM Controller as a critical farming window: a hungry villager that rests through twilight will be trapped hungry all night and risk starvation.

### Hunger

- Villagers start getting hungry after **120 sim-seconds** without eating.
- Eating requires spending at least 1 continuous sim-second inside a farm zone.
- A hungry villager loses an additional **3.5 hp/s**. Even inside the village, prolonged starvation is lethal.
- The RL reward for farming is **3× higher** when the villager is hungry, teaching agents to seek food proactively.

### Lifecycle

| Stage | Duration | Notes |
|-------|----------|-------|
| Childhood | 0 – 60 sim-sec | Cannot act independently; gains health from nearby parents |
| Adulthood | 60 sim-sec – death | Full RL agent; can reproduce |
| Old age | 300 – 600 sim-sec | Natural death window |

**Death causes tracked:** Old Age, Starvation, Hunting Mishap (risk ≥ 100%), Troll Attack, Farming Exhaustion.

If a gender goes extinct and no children remain, a wanderer arrives from off-screen and the generation counter increments.

### Simulation Auto-Termination

The simulation stops automatically after **2 000 generations** or **300 000 sim-seconds**, whichever comes first. A summary line is printed to stdout. This is designed for long unattended training runs paired with the CSV stats logger.

---

## The RL Brain

### Architecture (`drl.py`)

Each villager owns one `ActorCritic` instance — a two-head neural network implemented in **pure NumPy** (no PyTorch, no TensorFlow):

```
State (11 dims) → Hidden (24 neurons, ReLU) → Actor head  → softmax over 5 actions
                                             → Critic head → scalar state-value estimate
```

### State Vector

| Index | Feature | Why it matters |
|-------|---------|----------------|
| 0 | `health / 100` | Urgency signal |
| 1 | `risk / 100` | Danger awareness |
| 2 | `age / max_age` | Proximity to death |
| 3 | `is_child` | Capability gate |
| 4 | `has_kid` | Parenting context |
| 5 | Kids nearby | Feeding opportunity |
| 6 | Potential mate nearby | Reproduction signal |
| 7 | Sex cooldown active | Reproduction readiness |
| 8 | `current_action / 4` | Inertia / momentum |
| 9 | `hunger_frac` | Food urgency (0–1, clipped at 2×threshold) |
| 10 | `is_night` | Day/night awareness for troll danger |

### Actions

| Index | Action | Zone | Reward shaping |
|-------|--------|------|----------------|
| 0 | Rest | Village | +1.2× at night, +0.4× during day |
| 1 | Farm West | Crop W | 3× when hungry, baseline otherwise |
| 2 | Farm East | Crop E | 3× when hungry, baseline otherwise |
| 3 | Hunt South | Hunt S | 0.35× reward during day, 0.05× at night |
| 4 | Hunt North | Hunt N | 0.35× reward during day, 0.05× at night |

A universal **−1.0 hp/s penalty** is injected whenever hunger is active, and a **+0.15/s survival bonus** rewards staying alive. This pushes agents toward: *farm when hungry → rest in village at night → avoid hunting at night*.

### Training

Each villager alternates between phases:
- **Training phase** (60 sim-sec) — stochastic policy, experiences stored in replay buffer
- **Testing phase** (30 sim-sec) — greedy (argmax) policy, evaluates learned behaviour

At each phase boundary, one **A2C gradient update** runs over a mini-batch of recent transitions:
- **Critic loss** — mean-squared TD error
- **Actor loss** — policy gradient scaled by advantage

### Knowledge Transfer

On reproduction, the child brain is built by `parent_a.brain.breed(parent_b.brain)`:

1. **Weight crossover** — each weight independently drawn from either parent's matrix
2. **Gaussian mutation** — ~12% of weights perturbed with small random noise
3. **Parent memory seeding** — child inherits the last 20 transitions from each parent
4. **Species pool seeding** — child draws 30 random transitions from the collective species memory pool

### Species Memory Pool

When a villager dies, its **entire replay buffer** is donated to a shared species pool (capacity 2000 transitions). Every newborn draws from this pool at birth, so hard-won survival knowledge — including what killed past villagers — propagates to future generations even without direct parent–child lineage.

---

## LLM Collective Memory (`model.py`)

Every **360 sim-seconds** (≈90 real-seconds at 4×), the simulation pauses villager actions, collects all observations logged during that cycle, and sends them to **GPT-4o-mini**. The model acts as a survival analyst and returns 2–5 structured JSON lessons. These lessons are immediately converted into **synthetic RL transitions** and injected into every living villager's replay buffer.

### What the LLM receives

Each observation is a JSON object describing one noteworthy event:

| Field | Example values | Meaning |
|-------|---------------|---------|
| `event` | `"witnessed_death"`, `"sudden_hp_drop"`, `"hp_restored"` | What happened |
| `subject` | `"self"`, `"Ninian"` | Who it happened to |
| `time` | `"day"`, `"twilight"`, `"night"`, `"dawn"` | Phase of cycle |
| `action` | `"Rest"`, `"Farm W"` | Active action at the time |
| `health_before` / `health_after` | `35`, `68` | HP change |
| `cause` | `"starvation"`, `"troll attack"` | Death cause (on death events) |
| `hunger` | `"hungry"`, `"fed"` | Hunger state |
| `nearby` | `["village", "troll"]` | What was in proximity |

Events are triggered by HP drops ≥ 25 over a 5-second snapshot window, HP gains ≥ 20, deaths within witness range (150 px), and notable actions.

### What the LLM returns

Each lesson is a JSON object:

```json
{
  "lesson": "Avoid hunting at night — troll damage is lethal.",
  "is_night": 1.0,
  "bad_actions": [3, 4],
  "good_actions": [0],
  "reward": -5.0,
  "fatal": true
}
```

Optional fields `hunger_frac` (0.0 = fed, 1.0 = starving) and `health_frac` (0.0 = near death, 1.0 = full) pin the synthetic transitions to the context where the lesson is meaningful.

### How lessons become training data

Each lesson generates `LLM_TRANSITIONS_PER_LESSON` (default 15) synthetic state-action-reward transitions per relevant action. These are injected into every living villager's replay buffer, capped at `LLM_MAX_SYNTHETIC_FRAC` (default 40%) of buffer capacity so real experience always dominates.

A safety layer enforces logical consistency regardless of LLM output — e.g. night lessons can never mark Rest (action 0) as bad, and farming lessons for hunger never fire at hunger_frac=0.

Every discussion is appended to `llm_log.txt` with the raw lessons, injection counts, and buffer stats.

### LLM config constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DISCUSSION_INTERVAL` | 360.0 | Sim-seconds between LLM calls |
| `DISCUSSION_TIMEOUT` | 12.0 | Real-seconds to wait for LLM response |
| `LLM_TRANSITIONS_PER_LESSON` | 15 | Synthetic transitions per action per lesson |
| `LLM_MAX_SYNTHETIC_FRAC` | 0.40 | Max fraction of replay buffer that can be synthetic |
| `HP_DROP_THRESHOLD` | 25.0 | HP lost in a snapshot window to log as an event |
| `HP_GAIN_THRESHOLD` | 20.0 | HP gained in a snapshot window to log as a positive event |
| `HP_SNAPSHOT_INTERVAL` | 5.0 | Sim-seconds between HP snapshots |
| `WITNESS_RANGE` | 150.0 | Pixel radius for witnessing events |

---

## Statistics Logging (`stats_logger.py`)

A `stats_log.csv` is written automatically, appending one row every **100 sim-seconds**. The file is reset on each new run.

### Columns

| Column | Description |
|--------|-------------|
| `sim_time_s` | Simulation time in seconds |
| `real_time_s` | Wall-clock time in seconds |
| `generation` | Current generation counter |
| `population` | Alive villager count |
| `males` / `females` / `children` | Breakdown of alive villagers |
| `total_born` | Cumulative births |
| `total_deaths` | Cumulative deaths |
| `deaths_old_age` | Deaths by cause breakdown |
| `deaths_starvation` | |
| `deaths_hunting` | |
| `deaths_troll` | |
| `deaths_farming` | |

### Quick plot

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stats_log.csv", comment="#")
df.plot(x="generation", y=["population", "total_deaths"])
plt.show()
```

---

## Project Structure

```
VillageSim/
├── main.py          # Entry point, event loop, keyboard / mouse / slider handling
├── simulation.py    # Villager class, Troll class, World engine, LLM observation collection
├── drl.py           # ActorCritic network, ReplayBuffer, breed()
├── renderer.py      # All pygame rendering — world, villagers, trolls, UI panel, speed slider
├── helpers.py       # Geometry utilities, particle system, world-surface builder
├── config.py        # Every tunable constant — tweak here, not in logic files
├── model.py         # LLM Controller — sends observations to GPT-4o-mini, returns survival lessons
├── stats_logger.py  # CSV stats logger — one row per 100 sim-seconds to stats_log.csv
├── requirements.txt
└── assets/          # Sprite icons (villagers, farm, troll, animals)
```

Output files written at runtime:

| File | Contents |
|------|----------|
| `stats_log.csv` | Per-100-sim-second population and death statistics |
| `llm_log.txt` | LLM lesson log — raw JSON lessons and injection counts per discussion |

---

## Tuning (`config.py`)

| Constant | Default | Effect |
|----------|---------|--------|
| `HUNT_RISK_BASE` | 7.0 | How fast hunting risk accumulates |
| `TROLL_ATTACK_DRAIN` | 12.0 | HP/sec damage from trolls |
| `HUNGER_THRESHOLD` | 120.0 | Sim-seconds before hunger kicks in |
| `HUNGER_DRAIN` | 3.5 | HP/sec lost while hungry |
| `AC_LEARNING_RATE` | 0.018 | Neural network update step size |
| `AC_MUTATION_RATE` | 0.12 | Fraction of weights mutated in each child |
| `AC_MUTATION_STD` | 0.08 | Noise magnitude per mutation |
| `AC_INHERIT_SPECIES` | 30 | Species pool samples drawn at birth |
| `MAX_VILLAGERS` | 20 | Hard cap on simultaneous population |
| `DAY_DURATION` | 160.0 | Sim-seconds of daylight per cycle |
| `NIGHT_DURATION` | 120.0 | Sim-seconds of night per cycle |
| `TWILIGHT_DURATION` | 40.0 | Sim-seconds of twilight (pre-night warning window) |
| `DAWN_DURATION` | 40.0 | Sim-seconds of dawn (post-night safe window) |
| `SIM_SPEED_DEFAULT` | 4 | Sim-seconds per real-second at startup |
| `DISCUSSION_INTERVAL` | 360.0 | Sim-seconds between LLM collective memory calls |
| `LLM_MAX_SYNTHETIC_FRAC` | 0.40 | Max fraction of replay buffer that can be synthetic |

---

## Dependencies

```
pygame==2.6.1
numpy>=1.26.0
openai>=1.0.0
python-dotenv>=1.0.0
```

The neural network is plain NumPy — no ML framework required. The only external service is the OpenAI API (used by the LLM Controller). The simulation runs fully offline without an API key; only the collective memory feature is disabled.
