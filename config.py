"""
config.py
---------
Central configuration for the Village simulation.
Tweak constants here without touching simulation logic.
"""

# ── Window ────────────────────────────────────────────────────────────────────
WINDOW_W    = 1150
WINDOW_H    = 700
FPS         = 60
TITLE       = "Village — Actor-Critic DRL Ecosystem"

# ── Simulation ────────────────────────────────────────────────────────────────
SIM_SPEED_DEFAULT  = 4      # sim-seconds per real-second
SIM_SPEED_MIN      = 1
SIM_SPEED_MAX      = 20

# ── Villager lifecycle ────────────────────────────────────────────────────────
MAX_AGE_MIN        = 300.0  # sim-seconds  (~5 min)
MAX_AGE_MAX        = 600.0  # sim-seconds  (~10 min)
ADULTHOOD_AGE      = 60.0   # sim-seconds  (~1 min)
KID_BOND_DURATION  = 120.0  # sim-seconds parents stay bonded to kid
SEX_COOLDOWN       = 90.0   # sim-seconds between reproduction events

# ── Rewards / risks ───────────────────────────────────────────────────────────
REST_REWARD        = 0.5
REST_RISK_DRAIN    = 3.0    # risk reduced per second while resting

FARM_REWARD        = 1.8
FARM_RISK_DRAIN    = 2.0
FARM_RISK_BASE     = 1.5    # risk added per second while farming
FARM_PARENT_MULT   = 0.55   # reward multiplier when caring for kid

HUNT_REWARD        = 2.5   # reduced from 6.0 — hunt should not dominate farm
HUNT_RISK_BASE     = 7.0    # risk added per second while hunting
HUNT_PARENT_MULT   = 0.50
HUNT_PARENT_RISK   = 14.0   # higher risk while distracted by kid

VILLAGE_HEALTH_REGEN = 3.0  # hp/second inside village
WILD_HEALTH_REGEN    = 0.3  # hp/second in the wild (slight drain net)
PASSIVE_HEALTH_DRAIN = 0.8  # hp/second always draining (aging/hunger)
FARM_HEALTH_DRAIN    = 0.1  # hp/second lost while farming (exertion)
HUNT_HEALTH_DRAIN    = 3.5  # hp/second lost while hunting (danger + exertion)

# ── Day / Night cycle (sim-seconds — scales with World.speed) ────────────────
# At default speed (SIM_SPEED_DEFAULT=4) divide by 4 to get real-world seconds.
DAY_DURATION        = 160.0  # sim-seconds of daylight    (≈40 real-s at 4×)
NIGHT_DURATION      = 120.0  # sim-seconds of night       (≈30 real-s at 4×)
TWILIGHT_DURATION   =  40.0  # sim-seconds of twilight    (≈10 real-s at 4×)
DAWN_DURATION       =  40.0  # sim-seconds of dawn        (≈10 real-s at 4×)
NIGHT_FADE          =  16.0  # sim-seconds fade per transition (≈4 real-s at 4×)
TROLL_EMERGE_DELAY  =  20.0  # sim-seconds into night before trolls emerge
TROLL_RETURN_EARLY  =  20.0  # sim-seconds before dawn trolls go home

# ── Trolls ────────────────────────────────────────────────────────────────────
TROLL_COUNT          = 5
TROLL_SPEED          = 1.6     # slightly faster than villagers
TROLL_ATTACK_RANGE   = 80.0    # pixels — proximity that triggers damage
TROLL_ATTACK_DRAIN   = 12.0    # extra hp/second lost near a troll
TROLL_TARGET_DUR_MIN = 4.0     # sim-seconds per roam waypoint
TROLL_TARGET_DUR_MAX = 10.0
# Mountain home positions (x, y) — spread across the bottom mountain strip
TROLL_HOMES = [
    (105, 665), (228, 672), (402, 659), (578, 668), (722, 673),
]

# ── Hunger ────────────────────────────────────────────────────────────────────
HUNGER_THRESHOLD   = 120.0  # sim-seconds without eating before hunger kicks in (~30 real-sec at 4x)
HUNGER_DRAIN       = 3.5    # extra hp/second lost while hungry
HUNGER_EAT_DURATION = 1.0   # continuous sim-seconds in farm/hunt to reset hunger

PARENT_FEED_RATE   = 2.5    # hp/second child gains from nearby parent
CHILD_NEARBY_DIST  = 90.0   # pixels

SEX_HEALTH_BONUS   = 18.0
SEX_RISK_PENALTY   = 8.0

# Penalty injected into the replay buffer when a villager dies before old age.
# A large negative value trains the agent to avoid the action that led to death.
PREMATURE_DEATH_PENALTY = -60.0

# Hard cap on simultaneous villagers.
# Reproduction is suppressed when alive count >= this value.
MAX_VILLAGERS = 20

# ── Actor-Critic hyper-parameters ─────────────────────────────────────────────
AC_STATE_SIZE   = 11   # +hunger_frac, +is_night vs original 9
AC_ACTION_SIZE  = 5
AC_HIDDEN_SIZE  = 24
AC_LEARNING_RATE = 0.018
AC_GAMMA         = 0.93
AC_MAX_MEMORY    = 300
AC_BATCH_SIZE    = 64
AC_MUTATION_RATE = 0.12     # fraction of weights mutated in child
AC_MUTATION_STD  = 0.08     # std of gaussian noise on mutated weights
AC_INHERIT_MEM   = 20       # memories inherited from each parent
AC_INHERIT_SPECIES   = 30   # memories drawn from species pool at birth
AC_SPECIES_POOL_SIZE = 2000 # capacity of the collective species experience pool

AC_TRAIN_DURATION = 60.0    # sim-seconds in training phase
AC_TEST_DURATION  = 30.0    # sim-seconds in testing phase

# ── World geometry ────────────────────────────────────────────────────────────
# Zones: (x, y, width, height)
ZONES = {
    "village": (390, 260, 230, 175),
    "crop_w":  ( 40,  55, 245, 155),
    "crop_e":  (700, 345, 235, 195),
    "hunt_s":  ( 40, 310, 205, 220),
    "hunt_n":  (700,  40, 225, 225),
}

# Action index → zone target (world coords)  0=rest 1=cropW 2=cropE 3=huntS 4=huntN
ACTION_TARGETS = [
    (505, 350),   # 0 — village / rest
    (162, 132),   # 1 — crop west
    (818, 442),   # 2 — crop east
    (142, 420),   # 3 — hunt south
    (812, 152),   # 4 — hunt north
]

ACTION_NAMES = ["Rest", "Farm W", "Farm E", "Hunt S", "Hunt N"]

ACTION_TARGET_JITTER = 45   # ± pixel jitter on destination

# ── Villager visuals ──────────────────────────────────────────────────────────
TRAIL_MAX_LEN      = 14
VILLAGER_SPEED_MIN = 1.0
VILLAGER_SPEED_MAX = 1.8
ACTION_DUR_MIN     = 3.0
ACTION_DUR_MAX     = 7.0
SEX_PROXIMITY      = 30     # pixels — triggers reproduction

# ── Names ─────────────────────────────────────────────────────────────────────
NAMES_M = [
    "Aldric","Borin","Cedwyn","Dafyd","Emrys","Faolan","Gareth",
    "Hafoc","Idwal","Jorvald","Keiron","Leofric","Madawc","Ninian","Oswin",
]
NAMES_F = [
    "Aelwen","Brida","Caera","Dwyn","Elowen","Fflur","Gwenith",
    "Hafwen","Isolde","Jarnsaxa","Keira","Luned","Morwenna","Nesta","Olwen",
]

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG          = ( 28,  45,  18)
C_GRASS       = ( 55,  95,  40)
C_CROP        = (160, 130,  40)
C_HUNT        = ( 30,  65,  20)
C_VILLAGE     = (120,  85,  50)
C_MALE        = ( 60, 110, 190)
C_FEMALE      = (190,  80, 120)
C_CHILD       = (210, 165,  30)
C_DEAD        = ( 90,  90,  90)
C_ACCENT      = (200, 165,  70)
C_TEXT        = (240, 225, 195)
C_MUTED       = (150, 130, 100)
C_HEALTH      = ( 80, 190,  80)
C_RISK        = (210,  70,  60)
C_TRAIN_DOT   = ( 70, 140, 220)
C_TEST_DOT    = (220, 150,  40)

C_LOG_BIRTH   = (180, 220,  80)
C_LOG_DEATH   = (210, 100,  80)
C_LOG_SEX     = (230, 140, 180)
C_LOG_REPOP   = (120, 200, 230)
C_LOG_DEFAULT = (180, 165, 140)

LOG_COLORS = {
    "birth":  C_LOG_BIRTH,
    "death":  C_LOG_DEATH,
    "sex":    C_LOG_SEX,
    "repop":  C_LOG_REPOP,
    "":       C_LOG_DEFAULT,
}

# ── Panel layout ──────────────────────────────────────────────────────────────
PANEL_WIDTH     = 300       # right-side UI panel
PANEL_MAX_VILLS = 18        # max villager cards shown
LOG_MAX_LINES   = 7
LOG_KEEP        = 80