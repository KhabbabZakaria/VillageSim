"""
helpers.py
----------
Pure utility functions used across the Village simulation.

Sections
~~~~~~~~
- Geometry / math helpers
- Time formatting
- Pygame drawing utilities
- World surface builder (static background rendered once)
- Particle system
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, TYPE_CHECKING

import pygame

import config as cfg

if TYPE_CHECKING:
    from simulation import Villager


# ── geometry ──────────────────────────────────────────────────────────────────

def dist(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points."""
    return math.hypot(ax - bx, ay - by)


def dist_vv(a: "Villager", b: "Villager") -> float:
    """Euclidean distance between two Villager objects."""
    return math.hypot(a.x - b.x, a.y - b.y)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def jitter(x: float, y: float, amount: float) -> Tuple[float, float]:
    """Return (x, y) displaced by ±amount in each axis."""
    return (
        x + random.uniform(-amount, amount),
        y + random.uniform(-amount, amount),
    )


def zone_for_point(x: float, y: float) -> str:
    """
    Return which zone a world-coord point falls in.
    Returns: 'crop' | 'hunt' | 'village' | 'wild'
    """
    for name, (zx, zy, zw, zh) in cfg.ZONES.items():
        if zx <= x <= zx + zw and zy <= y <= zy + zh:
            if "crop"    in name: return "crop"
            if "hunt"    in name: return "hunt"
            if "village" in name: return "village"
    return "wild"


# ── time / format ─────────────────────────────────────────────────────────────

def fmt_time(sim_seconds: float) -> str:
    """Format simulation seconds as  M:SS."""
    m   = int(sim_seconds) // 60
    sec = int(sim_seconds) % 60
    return f"{m}:{sec:02d}"


def fmt_float(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}"


# ── particle system ───────────────────────────────────────────────────────────

class Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "life", "max_life")

    def __init__(
        self,
        x: float, y: float,
        color: Tuple[int, int, int],
        speed_min: float = 0.4,
        speed_max: float = 2.2,
    ) -> None:
        angle      = random.uniform(0, math.tau)
        speed      = random.uniform(speed_min, speed_max)
        self.x     = x
        self.y     = y
        self.vx    = math.cos(angle) * speed
        self.vy    = math.sin(angle) * speed
        self.color = color
        self.life  = 1.0
        self.max_life = 1.0 + random.random() * 0.5

    def update(self, dt: float) -> None:
        self.x    += self.vx * dt * 55
        self.y    += self.vy * dt * 55
        self.life -= dt

    @property
    def alive(self) -> bool:
        return self.life > 0


def spawn_particles(
    particles: List[Particle],
    x: float, y: float,
    color: Tuple[int, int, int],
    count: int = 15,
) -> None:
    """Append *count* new particles to an existing list."""
    for _ in range(count):
        particles.append(Particle(x, y, color))


def update_particles(particles: List[Particle], dt: float) -> None:
    """Update all particles in-place and remove dead ones."""
    for p in particles:
        p.update(dt)
    particles[:] = [p for p in particles if p.alive]


# ── pygame drawing helpers ────────────────────────────────────────────────────

def draw_arc(
    surface: pygame.Surface,
    color: Tuple[int, int, int],
    rect: Tuple[int, int, int, int],
    start_angle: float,
    end_angle: float,
    width: int,
) -> None:
    """Safe wrapper around pygame.draw.arc that skips degenerate arcs."""
    if end_angle <= start_angle:
        return
    try:
        pygame.draw.arc(surface, color, rect, start_angle, end_angle, width)
    except Exception:
        pass


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int, y: int,
    color: Tuple[int, int, int] = cfg.C_TEXT,
) -> pygame.Rect:
    """Blit text at (x, y) and return the bounding Rect."""
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))
    return rendered.get_rect(topleft=(x, y))


def draw_bar(
    surface: pygame.Surface,
    x: int, y: int, w: int, h: int,
    fraction: float,
    fill_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Draw a horizontal progress bar."""
    fraction = clamp(fraction, 0.0, 1.0)
    pygame.draw.rect(surface, bg_color, (x, y, w, h))
    if fraction > 0:
        pygame.draw.rect(surface, fill_color, (x, y, int(w * fraction), h))


def draw_semitransparent_rect(
    surface: pygame.Surface,
    x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int],
    alpha: int,
    border_radius: int = 0,
) -> None:
    """Draw a rect with alpha onto *surface*."""
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*color, alpha))
    if border_radius:
        pygame.draw.rect(s, (*color, alpha), (0, 0, w, h), border_radius=border_radius)
    surface.blit(s, (x, y))


def draw_dashed_line(
    surface: pygame.Surface,
    color: Tuple[int, int, int],
    start: Tuple[float, float],
    end: Tuple[float, float],
    dash: int = 6,
    gap: int = 4,
    width: int = 1,
    alpha: int = 80,
) -> None:
    """Draw an alpha-blended dashed line between two points."""
    tmp  = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    x1, y1 = start; x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    ux = (x2 - x1) / length
    uy = (y2 - y1) / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        end_pos = min(pos + seg, length)
        if drawing:
            sx, sy = x1 + ux * pos,     y1 + uy * pos
            ex, ey = x1 + ux * end_pos, y1 + uy * end_pos
            pygame.draw.line(tmp, (*color, alpha), (int(sx), int(sy)), (int(ex), int(ey)), width)
        pos += seg
        drawing = not drawing
    surface.blit(tmp, (0, 0))


# ── static world surface ──────────────────────────────────────────────────────

def _wheat_stalk(surf: pygame.Surface, x: int, y: int) -> None:
    """Draw a single wheat stalk icon centred at (x, y-base)."""
    s  = (168, 142, 52)   # stem
    g  = (215, 185, 65)   # grain head
    gl = (200, 172, 56)   # side grain
    # stem
    pygame.draw.line(surf, s, (x, y), (x, y - 11), 1)
    pygame.draw.line(surf, s, (x, y - 11), (x - 1, y - 14), 1)
    # leaf blade
    pygame.draw.line(surf, s, (x, y - 6), (x - 4, y - 10), 1)
    # grain head (small ellipse)
    pygame.draw.ellipse(surf, g,  (x - 3, y - 22, 5, 9))
    # flanking side grains
    pygame.draw.ellipse(surf, gl, (x - 6, y - 19, 4, 6))
    pygame.draw.ellipse(surf, gl, (x + 1, y - 19, 4, 6))


def _pine_tree(surf: pygame.Surface, x: int, y: int, sz: float = 1.0) -> None:
    """Draw a stylised pine tree with trunk + three canopy tiers."""
    tw  = int(sz * 15)
    # trunk
    trunks = int(sz * 3)
    pygame.draw.rect(surf, (78, 50, 20), (x - trunks // 2, y, trunks, int(sz * 6)))
    # three tiers, darkest at bottom
    tiers = [
        (tw,               y - int(sz * 9),  (14, 46,  9)),
        (int(tw * 0.72),   y - int(sz * 15), (20, 58, 14)),
        (int(tw * 0.44),   y - int(sz * 21), (28, 70, 18)),
    ]
    for tier_w, tier_y, col in tiers:
        pts = [
            (x,               tier_y - int(sz * 7)),
            (x - tier_w // 2, tier_y),
            (x + tier_w // 2, tier_y),
        ]
        pygame.draw.polygon(surf, col, pts)
    # snow-tip highlight
    pygame.draw.circle(surf, (235, 240, 228), (x, y - int(sz * 24)), max(1, int(sz * 2)))



def scatter_trees(seed: int = 42) -> List[Tuple[int, int, float, int]]:
    """
    Return a list of (x, y, scale, variant) tuples for decorative world trees.
    Trees are placed outside zone rects, using a fixed seed for repeatability.
    """
    rng    = random.Random(seed)
    trees  = []
    margin = 18
    for _ in range(80):
        for _ in range(30):
            tx = rng.randint(20, cfg.WINDOW_W - cfg.PANEL_WIDTH - 20)
            ty = rng.randint(25, cfg.WINDOW_H - 25)
            ok = True
            for _, (zx, zy, zw, zh) in cfg.ZONES.items():
                if (zx - margin <= tx <= zx + zw + margin and
                        zy - margin <= ty <= zy + zh + margin):
                    ok = False
                    break
            if ok:
                trees.append((tx, ty, rng.uniform(0.55, 1.05), rng.randint(0, 2)))
                break
    return trees


def draw_trees(
    surface: pygame.Surface,
    trees: List[Tuple[int, int, float, int]],
) -> None:
    """Draw world-scatter pine trees using the same _pine_tree style."""
    TRUNK  = (78, 50, 20)
    COLS   = [(14, 44, 9), (20, 55, 14), (28, 68, 18)]
    for tx, ty, sz, variant in trees:
        tw   = int(sz * 14)
        trunks = int(sz * 3)
        pygame.draw.rect(surface, TRUNK, (tx - trunks//2, ty, trunks, int(sz*6)))
        tiers = [
            (tw,             ty - int(sz*8),  COLS[variant % 3]),
            (int(tw*0.7),    ty - int(sz*14), COLS[(variant+1) % 3]),
            (int(tw*0.42),   ty - int(sz*20), COLS[(variant+2) % 3]),
        ]
        for tier_w, tier_y, col in tiers:
            pts = [(tx, tier_y - int(sz*6)), (tx - tier_w//2, tier_y), (tx + tier_w//2, tier_y)]
            pygame.draw.polygon(surface, col, pts)
        # snow tip
        pygame.draw.circle(surface, (232, 238, 225), (tx, ty - int(sz*23)), max(1, int(sz*1.5)))


def draw_zone_labels(surface: pygame.Surface, font: pygame.font.Font) -> None:
    labels = {
        "village": ("⌂ Village",       cfg.C_ACCENT),
        "crop_w":  ("⚜ Cropland",      (220, 200, 100)),
        "crop_e":  ("⚜ Cropland",      (220, 200, 100)),
        "hunt_s":  ("⚔ Hunting Ground", (140, 210, 100)),
        "hunt_n":  ("⚔ Hunting Ground", (140, 210, 100)),
    }
    for name, (zx, zy, _zw, _zh) in cfg.ZONES.items():
        if name in labels:
            text, color = labels[name]
            draw_text(surface, font, text, zx + 5, zy + 4, color)



def blit_tiled(surface, tiles, rect):
    zx, zy, zw, zh = rect
    tw = tiles[0].get_width()
    th = tiles[0].get_height()

    i = 0
    for y in range(zy, zy + zh, th):
        for x in range(zx, zx + zw, tw):
            surface.blit(tiles[i % len(tiles)], (x, y))
            i += 1