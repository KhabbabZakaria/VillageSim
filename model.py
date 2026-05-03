"""
model.py
--------
LLM Controller for the Village simulation.

Villagers accumulate observations during each day/night cycle.
At the end of each cycle the Controller receives all observations,
extracts survival lessons, and returns structured JSON that the
simulation converts into synthetic RL training transitions.
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── system context ────────────────────────────────────────────────────────────

_SYSTEM = """You are a survival analyst for a medieval village simulation.
Villagers are reinforcement learning agents that choose between 5 actions:
  0 = Rest (safe inside the village, health regenerates but hunger is NOT reset)
  1 = Farm West  (resets hunger, low danger DURING THE DAY; places villager outside — troll danger at night)
  2 = Farm East  (resets hunger, low danger DURING THE DAY; places villager outside — troll danger at night)
  3 = Hunt South (high risk accumulation, heavy health drain; troll danger at night)
  4 = Hunt North (high risk accumulation, heavy health drain; troll danger at night)

Critical rules you must never violate:
- ONLY actions 1 and 2 (farming) reset hunger. Action 0 (Rest) does NOT cure hunger.
- A villager that rests while hungry will continue to lose HP and eventually starve.
- If hunger is the concern, the correct lesson is always to farm (1 or 2), never to rest (0). Therefore: when hunger_frac is set, NEVER put 0 in good_actions unless 1 or 2 are also in good_actions.
- "hunting mishap" as a death cause means accumulated hunt risk hit 100% — this can happen even if the villager recently switched away from hunting.
- If a sudden_hp_drop observation shows hunger="hungry", the HP loss is primarily from starvation, not from the action itself — do not blame the action in that case.
- A troll in "nearby" only means danger if the villager is OUTSIDE the village. If zone is "village", the troll cannot attack — do not treat it as a threat.
- The time field in observations has four values: "day", "twilight", "night", "dawn".
  * "twilight" = the 10 seconds before night — danger rising, but trolls are NOT yet active. Farming is still safe; hungry villagers MUST farm now before night forces them home.
  * "night"    = full night — trolls are active everywhere outside the village.
  * "dawn"     = the 10 seconds after night ends — trolls are retreating, safe to resume farming immediately.
  * "day"      = safe daytime — normal farming and hunting.
- At night (time="night"), the ONLY safe action is Rest (0) — it always moves the villager to the village. Actions 1, 2, 3, 4 all place the villager outside where trolls patrol. Therefore: NEVER put 1, 2, 3, or 4 in good_actions when is_night=1.0, and NEVER put 0 in bad_actions when is_night=1.0.
- If a villager is hungry during twilight or dawn, farming (1 or 2) is the correct action — it is the last/first safe window before/after night. Generate a lesson with good_actions=[1,2], hunger_frac>=0.5, is_night=0.0.

Your job: read villager observations and extract concise, actionable survival lessons."""

# ── observation schema (told to LLM, not hardcoded in code) ──────────────────

_OBS_SCHEMA_DESC = (
    "Each observation is a JSON object. "
    "Possible keys: event, subject, nearby, time, action, "
    "health_before, health_after, cause. "
    "Not all keys are present in every observation. "
    "The 'time' field is one of: 'day', 'twilight', 'night', 'dawn'."
)

# ── lesson output schema ──────────────────────────────────────────────────────

_LESSON_SCHEMA = {
    "lesson":      "One sentence survival lesson starting with 'We should...' or 'Avoid...'",
    "is_night":    "1.0 if this lesson applies ONLY at night, 0.0 if ONLY during day. Omit entirely if the lesson applies at any time.",
    "hunger_frac": "Float 0.0–1.0 pinned in the training state. SCALE: 0.0 = perfectly fed (no hunger at all); 1.0 = starving (maximum hunger). HIGHER = MORE HUNGRY. Examples: 0.1 = well-fed, 0.5 = moderately hungry, 0.9 = near starvation. Set to the hunger level at which this lesson is meaningful. NEVER omit when good_actions includes 0 (rest) and is_night is not 1.0 — even if the lesson text does not mention hunger, omitting it lets the training vector fire at hunger=0.9 where resting is lethal. For any health-recovery rest lesson, always pin hunger LOW (0.1–0.3). NEVER set to 0.0.",
    "health_frac": "Float 0.0–1.0 pinned in the training state. SCALE: 0.0 = near death; 1.0 = full health. HIGHER = HEALTHIER. Examples: 0.2 = very low health, 0.8 = healthy. Set to the health level at which this lesson is meaningful. Omit if health level is not central to the lesson.",
    "bad_actions": "List of action indices (integers 0–4) to AVOID based on this lesson. Can be empty.",
    "good_actions":"List of action indices (integers 0–4) to PREFER based on this lesson. Can be empty.",
    "reward":      "Float. How severe is violating / how good is following this lesson. Range: -6.0 (lethal) to +4.0 (great).",
    "fatal":       "true if this pattern is likely to cause death, false otherwise.",
}


# ── few-shot examples ─────────────────────────────────────────────────────────

_FEW_SHOT = """
PARAMETER SCALE REFERENCE
  hunger_frac : 0.0 = perfectly fed  ↔  1.0 = starving.   HIGHER = MORE HUNGRY.
  health_frac : 0.0 = near death     ↔  1.0 = full health. HIGHER = HEALTHIER.
  is_night    : 0.0 = daytime        ↔  1.0 = full night.

WORKED EXAMPLES

Example A — villager resting with low health recovers HP (hunger: fed):
  Observation: {"event": "hp_restored", "subject": "self", "nearby": ["village"],
                "time": "day", "action": "Rest", "health_before": 35, "health_after": 68,
                "hunger": "fed"}
  Correct lesson:
    {"lesson": "We should rest when health is low and we are well-fed.",
     "hunger_frac": 0.1, "health_frac": 0.3, "good_actions": [0], "reward": 3.0, "fatal": false}
  Key points:
    • hunger_frac=0.1, NOT 0.8. "Fed" = low hunger = LOW hunger_frac.
      hunger_frac=0.8 would mean "nearly starving" — the opposite of well-fed.
    • hunger_frac must NOT be omitted even though the lesson text does not mention hunger.
      Omitting it lets the vector fire at hunger=0.9 where resting causes starvation.
      Always pin hunger_frac LOW (0.1–0.3) for health-recovery rest lessons.

Example B — villager dies of starvation while resting during the day:
  Observation: {"event": "witnessed_death", "subject": "Ninian",
                "nearby": ["village", "Rest"], "time": "day",
                "action": "Rest", "cause": "starvation"}
  Correct lesson:
    {"lesson": "Avoid resting when hungry — farm to reset hunger instead.",
     "hunger_frac": 0.5, "bad_actions": [0], "good_actions": [1, 2],
     "reward": -4.0, "fatal": true}
  Key points:
    • hunger_frac=0.5 because this lesson only applies when hunger is elevated.
      High hunger_frac = hungry villager. Do NOT set hunger_frac low here.
    • is_night is omitted — the lesson applies whenever hunger is high, day or night.
      The safety layer separates day/night vectors automatically.
"""

# ── public API ────────────────────────────────────────────────────────────────

def build_prompt(observations: list[dict]) -> str:
    obs_json = json.dumps(observations, indent=2)
    lesson_schema_json = json.dumps(_LESSON_SCHEMA, indent=2)

    return f"""Here are survival observations collected from villagers during the last day/night cycle.

{_OBS_SCHEMA_DESC}

Observations:
{obs_json}

Analyze these observations and extract 2–5 key survival lessons.

Return a JSON object with a single key "lessons" containing an array.
Each lesson must follow this schema exactly:
{lesson_schema_json}

Rules:
- If a lesson applies at ALL times regardless of day or night, omit is_night entirely AND do not mention "day" or "night" in the lesson text.
- If a lesson is specific to daytime only, you MUST set is_night=0.0 AND the lesson text must reflect that.
- If a lesson is specific to night only, you MUST set is_night=1.0 AND the lesson text must reflect that.
- Never mention "day" or "night" in lesson text without setting the matching is_night value. Inconsistency between lesson text and is_night is an error.
- If a lesson mentions low health (e.g. health restored, near death, low HP), set health_frac to a low value (0.1–0.3).
- If a lesson mentions high health or healthy villagers, set health_frac to a high value (0.7–0.9).
- Omit health_frac entirely if health level is not central to the lesson.
- "bad_actions" and "good_actions" must contain integers 0–4 only.
- Be specific: if trolls caused damage at night, bad_actions should be non-village actions.
- If a lesson recommends resting (good_actions includes 0) for health recovery (NOT because it is nighttime), you MUST also set hunger_frac to a LOW value (0.1–0.3). A hungry villager cannot recover by resting — they will continue to starve. This lesson should only fire when the villager is already well-fed. Do NOT omit hunger_frac on a health-recovery rest lesson unless is_night=1.0.
{_FEW_SHOT}
- Return raw JSON only — no markdown, no explanation outside the JSON."""


def parse_lessons(response_text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", response_text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("lessons", [])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        match = re.search(r'\{.*"lessons".*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()).get("lessons", [])
            except json.JSONDecodeError:
                pass
    return []


def call_controller(observations: list[dict]) -> list[dict]:
    """
    Send all villager observations to the LLM Controller.
    Returns a list of structured lesson dicts (may be empty on error).
    Runs synchronously — call from a background thread.
    """
    if not observations:
        return []

    prompt = build_prompt(observations)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return parse_lessons(response.choices[0].message.content)
    except Exception as exc:
        print(f"[LLM Controller] Error: {exc}")
        return []
