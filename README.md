# VillageSim — Actor-Critic DRL Ecosystem

A living village simulation where every villager is an independent **Actor-Critic reinforcement learning agent** built from scratch in pure NumPy. Villagers explore, learn, age, reproduce, and die. Children inherit neural weights and memories from both parents. Knowledge accumulates across generations through a shared **species memory pool**. Emergent behaviour arises purely from the RL policies — no rules, no scripted AI.

---

## Screenshots

| Day | Night |
|-----|-------|
| ![Day](day.png) | ![Night](night.png) |

During the day villagers farm and rest. At night, trolls emerge from the mountains — villagers that venture outside the village take heavy damage and risk death.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Requires **Python 3.10+** and **pygame 2.6+**.

---

## Controls

| Key / Input | Action |
|-------------|--------|
| `Space` | Pause / resume |
| `R` | Full reset |
| `+` / `=` | Increase simulation speed (up to 20×) |
| `-` | Decrease simulation speed |
| `Left click` | Select a villager — shows live policy tooltip |
| `Drag speed slider` | Adjust simulation speed (right panel, top bar) |
| `Esc` / `Q` | Quit |

---

## What's Happening on Screen

The **left panel** is the simulated world. The **right panel** is a live dashboard showing:

- **Population** — alive villager count and generation number
- **Deaths** — total deaths broken down by cause (Old Age, Starvation, Hunting, Troll Attack, Farming Exhaustion)
- **Events log** — births, deaths, reproduction events, and coming-of-age
- **Speed slider** — drag to adjust simulation speed from 1× to 20×; a tick marks the default speed

Each villager displays:
- A coloured dot (blue = male adult, pink = female adult, yellow = child)
- A movement trail
- A health bar above the sprite
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

The cycle has four distinct phases. At default speed (4×), divide sim-seconds by 4 to get real-world seconds.

| Phase | Duration | What happens |
|-------|----------|--------------|
| **Day** | 160 sim-s (≈40 real-s) | Full daylight; farming and hunting reward at full rate |
| **Twilight** | 40 sim-s (≈10 real-s) | Transition day → night; danger ramps up; villagers outside the village re-evaluate their action immediately |
| **Night** | 120 sim-s (≈30 real-s) | Full darkness; trolls active; rest reward peaks, hunt reward near zero |
| **Dawn** | 40 sim-s (≈10 real-s) | Transition night → day; danger ramps down; trolls return home |

A sun/moon indicator at the top-centre of the map tracks the cycle. At night, a dark blue overlay fades in over the world.

**Trolls** emerge from their mountain homes 20 sim-seconds into each night and return 20 sim-seconds before dawn. Any villager caught outside the village takes **12 hp/s** of troll damage on top of normal drains.

#### Action reset on phase transitions

When **twilight** or **night** begins, every villager that is not inside the village has its action timer zeroed. This forces an immediate policy re-evaluation so agents react to rising danger and head home rather than finishing a long action committed to during full daylight.

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

---

## The RL Brain

### Architecture (`drl.py`)

Each villager owns one `ActorCritic` instance — a two-head neural network implemented in **pure NumPy** (no PyTorch, no TensorFlow):

```
State (11 dims) → Hidden (24 neurons, ReLU) → Actor head  → softmax over 5 actions
                                             → Critic head → scalar state-value estimate
```

**Numerical stability:** the softmax is guarded against zero or non-finite sums — if the exponential total collapses, the policy falls back to a uniform distribution. `sample_action` similarly validates probabilities before sampling.

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

## CSV Logging (`logger.py`)

`SimLogger` automatically records simulation statistics to **`sim_log.csv`** at a fixed interval (every 100 sim-seconds, ≈25 real-seconds at default 4× speed). On quit, `finalize()` writes one final row capturing the end state.

### Columns

| Column | Description |
|--------|-------------|
| `sim_time_s` | Simulation time in seconds |
| `real_time_s` | Wall-clock time in seconds |
| `generation` | Current generation counter |
| `population` | Alive villager count |
| `males` | Alive adult males |
| `females` | Alive adult females |
| `children` | Alive children |
| `total_born` | Cumulative births |
| `total_deaths` | Cumulative deaths |
| `deaths_old_age` | Deaths by old age |
| `deaths_starvation` | Deaths by starvation |
| `deaths_hunting` | Deaths by hunting mishap |
| `deaths_troll` | Deaths by troll attack |
| `deaths_farming` | Deaths by farming exhaustion |

### Post-run analysis

```python
import pandas as pd
df = pd.read_csv("sim_log.csv", comment="#")
df.plot(x="sim_time_s", y="population")
df.plot(x="sim_time_s", y=["deaths_troll", "deaths_starvation", "deaths_hunting"])
```

---

## Auto-Stop / Run Limits

Two hard limits are set in `main.py` to allow unattended runs:

| Constant | Default | Effect |
|----------|---------|--------|
| `MAX_GENERATIONS` | 2000 | Stops the simulation after this many repopulation generations |
| `MAX_SIM_TIME` | 300 000 sim-s | Stops the simulation after this much simulated time |

When either limit is reached the simulation exits cleanly, calls `logger.finalize()` to flush the final CSV row, and closes the window. Adjust these values directly in `main.py` for longer or shorter batch runs.

---

## Project Structure

```
VillageSim/
├── main.py          # Entry point, event loop, keyboard / mouse / slider handling
├── simulation.py    # Villager class, Troll class, World engine
├── drl.py           # ActorCritic network, ReplayBuffer, breed()
├── renderer.py      # All pygame rendering — world, villagers, trolls, UI panel
├── helpers.py       # Geometry utilities, particle system, world-surface builder
├── logger.py        # SimLogger — writes sim_log.csv for post-run analysis
├── config.py        # Every tunable constant — tweak here, not in logic files
├── requirements.txt
└── assets/          # Sprite icons (villagers, farm, troll, animals)
```

---

## Tuning (`config.py`)

| Constant | Default | Effect |
|----------|---------|--------|
| `HUNT_RISK_BASE` | 7.0 | How fast hunting risk accumulates |
| `TROLL_ATTACK_DRAIN` | 12.0 | HP/sec damage from trolls |
| `TROLL_EMERGE_DELAY` | 20.0 | Sim-seconds into night before trolls emerge |
| `TROLL_RETURN_EARLY` | 20.0 | Sim-seconds before dawn that trolls go home |
| `HUNGER_THRESHOLD` | 120.0 | Sim-seconds before hunger kicks in (~30 real-sec at 4×) |
| `HUNGER_DRAIN` | 3.5 | HP/sec lost while hungry |
| `AC_LEARNING_RATE` | 0.018 | Neural network update step size |
| `AC_MUTATION_RATE` | 0.12 | Fraction of weights mutated in each child |
| `AC_MUTATION_STD` | 0.08 | Noise magnitude per mutation |
| `AC_INHERIT_SPECIES` | 30 | Species pool samples drawn at birth |
| `MAX_VILLAGERS` | 20 | Hard cap on simultaneous population |
| `DAY_DURATION` | 160.0 | Sim-seconds of daylight per cycle (≈40 real-s at 4×) |
| `NIGHT_DURATION` | 120.0 | Sim-seconds of night per cycle (≈30 real-s at 4×) |
| `TWILIGHT_DURATION` | 40.0 | Sim-seconds of twilight (day → night transition) |
| `DAWN_DURATION` | 40.0 | Sim-seconds of dawn (night → day transition) |
| `NIGHT_FADE` | 16.0 | Sim-seconds over which the darkness overlay fades in/out |
| `SIM_SPEED_DEFAULT` | 4 | Sim-seconds per real-second at startup |

---

## Dependencies

```
pygame==2.6.1
numpy>=1.26.0
```

No other dependencies. The neural network is plain NumPy — no ML framework required.
