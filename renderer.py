"""
renderer.py
-----------
All pygame rendering for the Village simulation.

Fonts (loaded from actual TTF files):
  Luminari        — decorative/medieval for title
  Georgia Bold    — elegant serif for section headers & tooltip title
  Verdana         — crisp sans for labels / body text
  Trebuchet Bold  — strong sans for HUD stats

Design language:
  - Deep parchment panel with pre-rendered vertical gradient
  - HUD: top gradient bar with separator-dot stat row
  - Villager cards: colour top-strip, inner highlight, rounded fancy bars
  - Tooltip: alternating zebra rows, full policy bar chart
  - World zone labels: pill with coloured border + icon
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pygame
import os
import random

import config as cfg

from helpers import (
    draw_arc,
    draw_trees,
    scatter_trees,
    _pine_tree,
    _wheat_stalk,
    blit_tiled,
    zone_for_point,
)

if TYPE_CHECKING:
    from simulation import Villager, World


# ── font paths ────────────────────────────────────────────────────────────────
_SYS  = "/System/Library/Fonts/Supplemental/"
_ROOT = "/System/Library/Fonts/"

FONT_PATHS = {
    "title":     _SYS  + "Luminari.ttf",          # medieval / decorative
    "serif_b":   _SYS  + "Georgia Bold.ttf",       # elegant bold serif
    "serif":     _SYS  + "Georgia.ttf",
    "sans":      _SYS  + "Verdana.ttf",            # very readable sans
    "sans_b":    _SYS  + "Verdana Bold.ttf",
    "hud_b":     _SYS  + "Trebuchet MS Bold.ttf",  # strong HUD stats
    "hud":       _SYS  + "Trebuchet MS.ttf",
}


def _font(key: str, size: int) -> pygame.font.Font:
    """Load font from TTF file; fall back to pygame default with a size bump."""
    path = FONT_PATHS.get(key, "")
    if path:
        try:
            return pygame.font.Font(path, size)
        except Exception:
            pass
    return pygame.font.Font(None, size + 6)


# ── palette ───────────────────────────────────────────────────────────────────
PANEL_TOP       = (16,  11,   6)
PANEL_BOT       = (10,   7,   4)
CARD_BG         = (26,  18,  10)
DIVIDER         = (80,  57,  28)
DIVIDER_BRIGHT  = (122, 90,  44)
ACCENT_GOLD     = (222, 182,  78)
ACCENT_WARM     = (248, 205, 108)
TEXT_PRIMARY    = (245, 230, 200)
TEXT_SECONDARY  = (172, 152, 115)
TEXT_DIM        = ( 100, 87,  64)
TEXT_MUTED      = (  68, 58,  42)

COL_HEALTH = ( 62, 200,  82)
COL_RISK   = (220,  62,  48)
COL_TRAIN  = ( 70, 148, 238)
COL_TEST   = (230, 160,  48)
COL_CHILD  = (220, 178,  40)

ACTION_COLS = [
    (108, 158, 218),   # Rest   – slate blue
    (128, 200,  90),   # Farm W – grass green
    (105, 178,  82),   # Farm E – forest green
    (218, 108,  52),   # Hunt S – ember orange
    (198,  80,  48),   # Hunt N – deep red
]

GENDER_SYM = {"M": "♂", "F": "♀"}

LOG_COLORS = {
    "birth": (165, 228,  76),
    "death": (222,  90,  70),
    "sex":   (235, 138, 190),
    "repop": ( 95, 200, 235),
    "":      TEXT_SECONDARY,
}

HUD_H = 40      # pixels — top bar height


def load_image(path: str, size: Tuple[int, int] | None = None):
    img = pygame.image.load(path).convert_alpha()
    if size:
        img = pygame.transform.smoothscale(img, size)
    return img


# ── drawing helpers ───────────────────────────────────────────────────────────

def _txt(surf, font, text, x, y, color=TEXT_PRIMARY) -> int:
    r = font.render(text, True, color)
    surf.blit(r, (x, y))
    return r.get_width()

def _txt_r(surf, font, text, rx, y, color=TEXT_PRIMARY) -> None:
    r = font.render(text, True, color)
    surf.blit(r, (rx - r.get_width(), y))

def _hline(surf, y, x0, x1, color=DIVIDER, w=1) -> None:
    pygame.draw.line(surf, color, (x0, y), (x1, y), w)

def _pill(surf, font, text, x, y, bg, fg, pad_x=6, pad_y=2) -> int:
    r  = font.render(text, True, fg)
    bw = r.get_width() + pad_x * 2
    bh = r.get_height() + pad_y * 2
    s  = pygame.Surface((bw, bh), pygame.SRCALPHA)
    pygame.draw.rect(s, (*bg, 222), (0, 0, bw, bh), border_radius=5)
    hl = tuple(min(255, c + 50) for c in bg)
    pygame.draw.rect(s, (*hl, 75), (1, 1, bw-2, 3), border_radius=4)
    s.blit(r, (pad_x, pad_y))
    surf.blit(s, (x, y))
    return bw

def _bar_fancy(surf, x, y, w, h, frac, fill_col,
               track=(30, 22, 12), brad=3) -> None:
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(surf, track, (x, y, w, h), border_radius=brad)
    fw = max(0, int(w * frac))
    if fw > 0:
        pygame.draw.rect(surf, fill_col, (x, y, fw, h), border_radius=brad)
        hl = tuple(min(255, c + 75) for c in fill_col)
        pygame.draw.rect(surf, hl, (x+1, y+1, fw-2, max(1, h//3)), border_radius=brad)
    pygame.draw.rect(surf, (0, 0, 0), (x, y, w, h), 1, border_radius=brad)

def _sparkbar(surf, x, y, w, h, values, colors, gap=2) -> None:
    total = sum(values)
    if total <= 0:
        return
    cx   = x
    used = w - gap * (len(values) - 1)
    for v, c in zip(values, colors):
        sw = max(1, int(used * v / total))
        pygame.draw.rect(surf, c, (cx, y, sw, h), border_radius=2)
        cx += sw + gap

def _sec_header(surf, font, title, x, y, w, color=ACCENT_GOLD) -> int:
    """Centred title flanked by a double-line divider (bright + shadow)."""
    r   = font.render(title, True, color)
    rw  = r.get_width()
    mid = x + w // 2
    surf.blit(r, (mid - rw//2, y))
    mg = 12
    for x0, x1 in [(x + mg, mid - rw//2 - 8), (mid + rw//2 + 8, x + w - mg)]:
        if x1 > x0:
            my = y + r.get_height() // 2
            _hline(surf, my,     x0, x1, DIVIDER_BRIGHT)
            _hline(surf, my + 1, x0, x1, DIVIDER)
    return r.get_height() + 7


# ── renderer ──────────────────────────────────────────────────────────────────

class Renderer:

    def __init__(self, screen: pygame.Surface) -> None:
        BASE = "assets/icons/" 
        self.img_villager_male   = pygame.image.load(BASE + "villagerMale.png").convert_alpha()
        self.img_villager_female = pygame.image.load(BASE + "villagerFemale.png").convert_alpha()
        self.img_boy             = pygame.image.load(BASE + "villagerBoy.png").convert_alpha()
        self.img_girl            = pygame.image.load(BASE + "villagerGirl.png").convert_alpha()
        self.village_tiles = [
            self._load_tile(BASE + "village.png"),
            self._load_tile(BASE + "village2.png"),
            self._load_tile(BASE + "village3.png"),
            self._load_tile(BASE + "village4.png"),
        ]
        self.farm1_tiles = [
            self._load_tile(BASE + "farm.png", size=(50, 50))
            for _ in range(4)
        ]

        self.farm2_tiles = [
            self._load_tile(BASE + "farm2.png", size=(50, 50))
            for _ in range(4)
        ]

        self.hunt_deer  = self._load_icon(BASE + "deer.png")
        self.hunt_lion  = self._load_icon(BASE + "lion.png")
        self.hunt_eleph = self._load_icon(BASE + "elephant.png")
        self.troll_img  = self._load_icon(BASE + "troll.png", size=34)


        self.screen = screen
        self.W = screen.get_width()
        self.H = screen.get_height()

        # real fonts from TTF files
        self.f_title  = _font("title",  24)   # Luminari  — medieval title
        self.f_head   = _font("serif_b", 13)  # Georgia Bold — section headers
        self.f_label  = _font("sans",    11)  # Verdana — card labels
        self.f_small  = _font("sans",    10)  # Verdana — small text
        self.f_tiny   = _font("sans",     9)  # Verdana — HP/RK etc.
        self.f_hud    = _font("hud_b",   13)  # Trebuchet Bold — HUD stats
        self.f_hud_sm = _font("hud",      9)  # Trebuchet — hint text
        self.f_tip_h  = _font("serif_b", 14)  # Georgia Bold — tooltip title
        self.f_tip    = _font("sans",    10)  # Verdana — tooltip body

        # world
        self.world_surf = self._build_world_surface(self.W, self.H)
        self.trees      = scatter_trees(seed=42)

        self.pw = cfg.PANEL_WIDTH
        self.px = self.W - self.pw

        # pre-render panel gradient (dark top → slightly lighter bottom)
        self._panel_bg = pygame.Surface((self.pw, self.H))
        for i in range(self.H):
            t = i / self.H
            c = (
                int(PANEL_TOP[0] + (PANEL_BOT[0] - PANEL_TOP[0]) * t),
                int(PANEL_TOP[1] + (PANEL_BOT[1] - PANEL_TOP[1]) * t),
                int(PANEL_TOP[2] + (PANEL_BOT[2] - PANEL_TOP[2]) * t),
            )
            pygame.draw.line(self._panel_bg, c, (0, i), (self.pw, i))

        # pre-render HUD gradient (dark → slightly lighter)
        self._hud_bg = pygame.Surface((self.W, HUD_H))
        for i in range(HUD_H):
            t = i / HUD_H
            r = int(20 - t * 12)
            g = int(14 - t * 8)
            b = int(7  - t * 4)
            pygame.draw.line(self._hud_bg, (r, g, b), (0, i), (self.W, i))

        # reusable SRCALPHA surfaces
        self._trail_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._bond_surf  = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._night_surf = pygame.Surface((self.px, self.H), pygame.SRCALPHA)

        self._sprite_cache = {}

    WORLD_ICON_SIZE = 28  # tweak 24–40 depending on style

    def _load_icon(self, path, size=WORLD_ICON_SIZE):
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, (size, size))

    def _load_tile(self, path, size=(48, 48)):
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size)
        
    def _get_sprite(self, img, size):
        size = max(12, min(64, size))
        key = (id(img), size)

        if key not in self._sprite_cache:
            self._sprite_cache[key] = pygame.transform.smoothscale(img, (size, size))

        return self._sprite_cache[key]

    def _sample_spaced_positions(self, rng, zx, zy, zw, zh, count, min_dist=28):
        positions = []
        attempts = 0
        max_attempts = count * 30

        while len(positions) < count and attempts < max_attempts:
            attempts += 1

            x = zx + rng.randint(20, zw - 20)
            y = zy + rng.randint(20, zh - 20)

            ok = True
            for px, py in positions:
                if (px - x) ** 2 + (py - y) ** 2 < min_dist ** 2:
                    ok = False
                    break

            if ok:
                positions.append((x, y))

        return positions

    # ── main ──────────────────────────────────────────────────────────────────
    def _build_world_surface(self, width: int, height: int) -> pygame.Surface:
        """
        Render the static background (terrain, zones, cabins, well, river, vignette)
        onto a Surface that is blitted once per frame without regeneration.
        """
        surf = pygame.Surface((width, height)).convert_alpha()

        # ── base grass — lush gradient + noise (FAST) ─────────────────────────────
        rng = random.Random(12345)

        for y in range(height):
            t = y / height

            base_r = int(40 + t * 10)
            base_g = int(130 + t * 50)
            base_b = int(40 + t * 10)

            noise = rng.randint(-10, 10)

            col = (
                max(0, min(255, base_r + noise)),
                max(0, min(255, base_g + noise)),
                max(0, min(255, base_b + noise)),
            )

            pygame.draw.line(surf, col, (0, y), (width, y))

        # ── hunting grounds ───────────────────────────────────────────────────────
        for name, (zx, zy, zw, zh) in cfg.ZONES.items():
            if "hunt" not in name:
                continue

            pygame.draw.rect(surf, (18, 48, 11), (zx, zy, zw, zh))
            rng = random.Random(hash(name))

            hunt_positions = []

            # spawn moss patches + animals (spaced)
            hunt_positions = self._sample_spaced_positions(rng, zx, zy, zw, zh, 35, min_dist=30)

            for bx, by in hunt_positions:
                pygame.draw.circle(
                    surf,
                    (32, 66, 17),
                    (bx, by),
                    rng.randint(3, 7)
                )

            # animals (spread across hunt_positions)
            animals = [self.hunt_deer, self.hunt_lion, self.hunt_eleph] * 3

            for i, img in enumerate(animals):
                if i < len(hunt_positions):
                    ax, ay = hunt_positions[i]
                    rect = img.get_rect(center=(ax, ay))
                    surf.blit(img, rect)


            # sunlight patches
            for _ in range(10):
                bx = zx + rng.randint(10, zw - 10)
                by = zy + rng.randint(10, zh - 10)
                pygame.draw.circle(surf, (32, 66, 17), (bx, by), rng.randint(4, 9))

            # pine trees
            tree_rng = random.Random(hash(name) + 77)
            for _ in range(22):
                tx = zx + tree_rng.randint(18, zw - 18)
                ty = zy + tree_rng.randint(26, zh - 8)
                tsz = tree_rng.uniform(0.65, 1.15)
                _pine_tree(surf, tx, ty, tsz)

            pygame.draw.rect(surf, (10, 30, 5), (zx, zy, zw, zh), 2)

        # ── croplands ─────────────────────────────────────────────────────────────
        for name, (zx, zy, zw, zh) in cfg.ZONES.items():
            if "crop" not in name:
                continue

            pygame.draw.rect(surf, (145, 115, 32), (zx, zy, zw, zh))

            # alternating ploughed rows
            for i, row in enumerate(range(zy + 12, zy + zh - 4, 11)):
                rc = (172, 145, 50) if i % 2 == 0 else (125, 100, 25)
                pygame.draw.line(surf, rc, (zx + 6, row), (zx + zw - 6, row), 2)

            # wheat stalks
            wheat_rng = random.Random(hash(name) + 42)
            for _ in range(55):
                wx_s = zx + wheat_rng.randint(14, zw - 14)
                wy_s = zy + wheat_rng.randint(28, zh - 8)
                _wheat_stalk(surf, wx_s, wy_s)

            # ── FIX: FARM ICONS / STRUCTURES ON FARMLAND ───────────────────────────
            farm_rng = random.Random(hash(name) + 99)

            farm_tiles = self.farm1_tiles if "w" in name else self.farm2_tiles

            farm_positions = self._sample_spaced_positions(
                farm_rng, zx, zy, zw, zh,
                count=4,          # adjust density
                min_dist=80
            )

            for i, (fx, fy) in enumerate(farm_positions):
                tile = farm_tiles[i % len(farm_tiles)]
                rect = tile.get_rect(center=(fx, fy))
                surf.blit(tile, rect)

            # border
            pygame.draw.rect(surf, (100, 74, 14), (zx, zy, zw, zh), 2)

        # -- vilage 
        zx, zy, zw, zh = cfg.ZONES["village"]
        blit_tiled(surf, self.village_tiles, (zx, zy, zw, zh))
        pygame.draw.rect(surf, (80, 55, 30), (zx, zy, zw, zh), 2)

        # outer border
        pygame.draw.rect(surf, (84, 55, 24), (zx, zy, zw, zh), 2)

        # Dirt path between zones (runs through centre of village)
        mid_x = zx + zw // 2
        pygame.draw.line(surf, (98, 68, 34), (mid_x, zy), (mid_x, zy+zh), 6)
        pygame.draw.line(surf, (112, 80, 44), (zx, zy+zh//2), (zx+zw, zy+zh//2), 6)

        # Well (draw fresh on top)
        wx, wy = cfg.ACTION_TARGETS[0]
        # shadow
        ov = pygame.Surface((30, 12), pygame.SRCALPHA)
        pygame.draw.ellipse(ov, (0, 0, 0, 60), (0, 0, 30, 12))
        surf.blit(ov, (wx-15, wy+10))
        pygame.draw.circle(surf, (68, 40, 18), (wx, wy), 14)
        pygame.draw.circle(surf, (45, 98, 152), (wx, wy), 9)
        pygame.draw.circle(surf, (88, 62, 28), (wx, wy), 9, 2)
        pygame.draw.circle(surf, (68, 40, 18), (wx, wy), 14, 2)


        # ── mountains — layered silhouettes along the bottom edge ────────────────
        world_w  = width - cfg.PANEL_WIDTH
        mrng     = random.Random(77)
        # (base_y, max_peak_height, rgb_color)
        m_layers = [
            (height - 22, 88, ( 50,  68,  58)),   # back  — tallest, lightest
            (height -  8, 62, ( 36,  50,  41)),   # mid
            (height,      44, ( 24,  36,  28)),   # front — shortest, darkest
        ]
        for base_y, max_peak, color in m_layers:
            pts = [(0, height), (0, base_y)]
            x   = 0
            while x < world_w:
                peak_h     = mrng.randint(max_peak // 2, max_peak)
                half_w     = mrng.randint(28, 62)
                tip_x      = min(x + half_w, world_w)
                right_x    = min(x + half_w * 2, world_w)
                pts.append((tip_x, base_y - peak_h))
                pts.append((right_x, base_y))
                x = right_x + mrng.randint(4, 18)
            pts += [(world_w, base_y), (world_w, height)]
            pygame.draw.polygon(surf, color, pts)
            # subtle highlight on each ridge crest
            hl = tuple(min(255, c + 28) for c in color)
            for i in range(2, len(pts) - 3, 2):
                ax, ay = pts[i]
                if ay < base_y - 10:   # only actual peaks, not base corners
                    pygame.draw.circle(surf, hl, (ax, ay), 2)

        # ── river ─────────────────────────────────────────────────────────────────
        river_pts = [
            (0, 590), (130, 572), (270, 585), (420, 568),
            (560, 580), (700, 562), (840, 575), (width, 560),
        ]
        rv = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.lines(rv, (38, 98, 148, 145), False, river_pts, 16)  # deep body
        pygame.draw.lines(rv, (62, 130, 182, 100), False, river_pts,  9)  # mid tone
        pygame.draw.lines(rv, (102, 168, 212,  60), False, river_pts, 4)  # surface glint
        surf.blit(rv, (0, 0))

        # ── vignette (darker edges for atmosphere) ────────────────────────────────
        vw = width - cfg.PANEL_WIDTH
        vig = pygame.Surface((width, height), pygame.SRCALPHA)
        edge = 90   # vignette width in pixels
        for i in range(edge):
            t = 1.0 - i / edge
            a = int(95 * t * t)
            if a > 0:
                pygame.draw.line(vig, (0, 0, 0, a), (i, 0), (i, height))              # left
                pygame.draw.line(vig, (0, 0, 0, a), (0, i), (vw, i))                  # top
                pygame.draw.line(vig, (0, 0, 0, a), (0, height-i), (vw, height-i))    # bottom
        surf.blit(vig, (0, 0))

        return surf


    def draw(self, world: "World") -> None:
        self.screen.set_clip(pygame.Rect(0, 0, self.px, self.H))
        self.screen.blit(self.world_surf, (0, 0))
        draw_trees(self.screen, self.trees)
        self._draw_zone_labels()
        self._draw_particles(world)
        self._draw_trails(world)
        self._draw_bond_lines(world)
        for v in world.alive():
            self._draw_villager(v, world.selected_uid)
        night_alpha = self._night_alpha(world.real_time)
        self._draw_night_overlay(night_alpha)
        self._draw_trolls(world, night_alpha)
        self._draw_day_night_indicator(night_alpha)
        self._draw_speech_bubbles(world)
        self._draw_discussion_banner(world)
        self._draw_hud(world)
        self._draw_panel(world)
        if world.selected_uid is not None:
            sel = next((v for v in world.villagers if v.uid == world.selected_uid), None)
            if sel and sel.alive:
                self._draw_tooltip(sel, world)

    # ── day / night cycle ─────────────────────────────────────────────────────

    def _night_alpha(self, real_time: float) -> float:
        """
        Returns darkness alpha 0–150 for the current moment.
        Uses world.real_time so renderer and simulation stay in sync.
        Cycle: DAY_DURATION day → NIGHT_FADE fade → NIGHT_DURATION night → repeat.
        """
        period = cfg.DAY_DURATION + cfg.NIGHT_DURATION
        phase  = real_time % period
        fade   = cfg.NIGHT_FADE

        if phase < cfg.DAY_DURATION - fade:             # full day
            return 0.0
        elif phase < cfg.DAY_DURATION:                  # fade day → night
            return 150.0 * (phase - (cfg.DAY_DURATION - fade)) / fade
        elif phase < period - fade:                     # full night
            return 150.0
        else:                                           # fade night → day
            return 150.0 * (1.0 - (phase - (period - fade)) / fade)

    def _draw_night_overlay(self, alpha: float) -> None:
        """Apply dark blue tint over the world area only."""
        a = int(alpha)
        if a < 3:
            return
        self._night_surf.fill((4, 10, 38, a))
        self.screen.blit(self._night_surf, (0, 0))

    def _draw_trolls(self, world: "World", night_alpha: float) -> None:
        """Draw trolls only during the night portion of the cycle.
        Also draws a red aura on any villager being attacked."""
        period    = cfg.DAY_DURATION + cfg.NIGHT_DURATION
        phase     = world.real_time % period
        is_night  = phase >= cfg.DAY_DURATION
        if not is_night:
            return

        out_start = cfg.DAY_DURATION + cfg.TROLL_EMERGE_DELAY
        out_end   = period - cfg.TROLL_RETURN_EARLY
        is_out    = out_start <= phase < out_end

        # Troll icon opacity: fade in over emerge delay, fade out over return delay
        if phase < out_start:
            t_vis = (phase - cfg.DAY_DURATION) / cfg.TROLL_EMERGE_DELAY
        elif phase >= out_end:
            t_vis = (period - phase) / cfg.TROLL_RETURN_EARLY
        else:
            t_vis = 1.0
        icon_alpha = max(0, min(255, int(t_vis * 255)))

        vx, vy, vw, vh = cfg.ZONES["village"]

        # Red aura on attacked villagers
        if is_out:
            for troll in world.trolls:
                for v in world.alive():
                    if zone_for_point(v.x, v.y) == "village":
                        continue
                    if math.hypot(troll.x - v.x, troll.y - v.y) < cfg.TROLL_ATTACK_RANGE:
                        aura = pygame.Surface((40, 40), pygame.SRCALPHA)
                        pygame.draw.circle(aura, (220, 30, 30, 70), (20, 20), 18)
                        self.screen.blit(aura, (int(v.x) - 20, int(v.y) - 20))

        # Troll sprites
        troll_surf = pygame.Surface(self.troll_img.get_size(), pygame.SRCALPHA)
        troll_surf.blit(self.troll_img, (0, 0))
        troll_surf.set_alpha(icon_alpha)
        for troll in world.trolls:
            rect = troll_surf.get_rect(center=(int(troll.x), int(troll.y)))
            self.screen.blit(troll_surf, rect)
            # eerie green glow beneath troll
            glow = pygame.Surface((48, 20), pygame.SRCALPHA)
            ga = max(0, int(icon_alpha * 0.45))
            pygame.draw.ellipse(glow, (30, 200, 60, ga), (0, 0, 48, 20))
            self.screen.blit(glow, (int(troll.x) - 24, int(troll.y) + 12))

    def _draw_day_night_indicator(self, alpha: float) -> None:
        """Sun (day) or moon (night) icon + label at top-centre of the world."""
        cx = self.px // 2          # horizontally: centre of world area
        cy = HUD_H + 30            # vertically:   just below the HUD bar

        is_night = alpha > 75.0
        t_frac   = alpha / 150.0   # 0=day, 1=night

        if t_frac < 0.98:          # ── sun ──────────────────────────────────
            sun_a = max(0, min(255, int((1.0 - t_frac) * 255)))
            # glow halo
            pygame.draw.circle(self.screen, (255, 240, 100), (cx, cy), 15)
            pygame.draw.circle(self.screen, (255, 215,  40), (cx, cy), 11)
            # 8 rays
            for i in range(8):
                angle = i * math.pi / 4
                x1 = cx + int(14 * math.cos(angle))
                y1 = cy + int(14 * math.sin(angle))
                x2 = cx + int(19 * math.cos(angle))
                y2 = cy + int(19 * math.sin(angle))
                pygame.draw.line(self.screen, (255, 200, 50), (x1, y1), (x2, y2), 2)

        if t_frac > 0.02:          # ── moon ─────────────────────────────────
            # Full moon circle
            pygame.draw.circle(self.screen, (215, 218, 195), (cx, cy), 11)
            # Shadow circle offset right-upward to carve crescent
            pygame.draw.circle(self.screen, (15, 22, 52), (cx + 5, cy - 4), 8)
            # Faint star dots near moon
            for sx, sy in [(cx + 20, cy - 14), (cx + 28, cy + 4), (cx + 18, cy + 16)]:
                pygame.draw.circle(self.screen, (210, 215, 200), (sx, sy), 1)

        # Label
        label  = "Night" if is_night else "Day"
        lcolor = (170, 185, 230) if is_night else (255, 230, 100)
        lsurf  = self.f_hud_sm.render(label, True, lcolor)
        self.screen.blit(lsurf, (cx - lsurf.get_width() // 2, cy + 18))

    # ── particles ─────────────────────────────────────────────────────────────

    def _draw_particles(self, world: "World") -> None:
        for p in world.particles:
            r = max(1, int(3 * p.life / p.max_life))
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), r)

    # ── trails ────────────────────────────────────────────────────────────────

    def _draw_trails(self, world: "World") -> None:
        self._trail_surf.fill((0, 0, 0, 0))
        for v in world.alive():
            if len(v.trail) < 2:
                continue
            pts  = [(int(x), int(y)) for x, y in v.trail]
            base = cfg.C_MALE if v.gender == "M" else cfg.C_FEMALE
            pygame.draw.lines(self._trail_surf, (*base, 50), False, pts, 2)
        self.screen.blit(self._trail_surf, (0, 0))

    # ── bond lines ────────────────────────────────────────────────────────────

    def _draw_bond_lines(self, world: "World") -> None:
        self._bond_surf.fill((0, 0, 0, 0))
        alive = world.alive()
        for v in alive:
            if not (v.has_kid and v.kid_bond_t > 0):
                continue
            for k in alive:
                if v.uid not in k.parent_ids:
                    continue
                if math.hypot(v.x - k.x, v.y - k.y) < 240:
                    a = max(25, int(85 * v.kid_bond_t / cfg.KID_BOND_DURATION))
                    pygame.draw.line(self._bond_surf, (245, 205, 65, a),
                                     (int(v.x), int(v.y)), (int(k.x), int(k.y)), 1)
        self.screen.blit(self._bond_surf, (0, 0))

    # ── zone labels (styled pills) ────────────────────────────────────────────

    def _draw_zone_labels(self) -> None:
        labels = {
            "village": ("[ The Village ]",   cfg.C_ACCENT,    (72, 50, 12)),
            "crop_w":  ("[ West Farmland ]", (228, 210, 108), (75, 68, 8)),
            "crop_e":  ("[ East Farmland ]", (228, 210, 108), (75, 68, 8)),
            "hunt_s":  ("[ South Forest ]",  (138, 215, 100), (16, 58, 10)),
            "hunt_n":  ("[ North Forest ]",  (138, 215, 100), (16, 58, 10)),
        }
        for name, (zx, zy, _zw, _zh) in cfg.ZONES.items():
            if name not in labels:
                continue
            text, fg, bg = labels[name]
            r  = self.f_label.render(text, True, fg)   # f_label = Verdana 11
            bw = r.get_width() + 14
            bh = r.get_height() + 8
            s  = pygame.Surface((bw, bh), pygame.SRCALPHA)
            pygame.draw.rect(s, (*bg, 175), (0, 0, bw, bh), border_radius=5)
            pygame.draw.rect(s, (*fg, 140), (0, 0, bw, bh), 1, border_radius=5)
            # top-edge highlight
            pygame.draw.rect(s, (255, 255, 255, 30), (2, 1, bw-4, 3), border_radius=4)
            s.blit(r, (7, 4))
            self.screen.blit(s, (zx + 6, zy + 7))

        # Troll Land label — centred across the bottom mountain strip
        tl_fg = (148, 200, 148)
        tl_bg = (18, 32, 20)
        r  = self.f_label.render("⛰  Troll Land", True, tl_fg)
        bw = r.get_width() + 16
        bh = r.get_height() + 8
        s  = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(s, (*tl_bg, 180), (0, 0, bw, bh), border_radius=5)
        pygame.draw.rect(s, (*tl_fg, 120), (0, 0, bw, bh), 1, border_radius=5)
        s.blit(r, (8, 4))
        self.screen.blit(s, (self.px // 2 - bw // 2, self.H - 42))

    # ── villager sprite ───────────────────────────────────────────────────────
    #
    # Layout (r = radius, adults=10, children=7):
    #   top-of-head:  iy - r - head_r + 1
    #   body centre:  (ix, iy + r//5)   — shifted down so head floats above
    #   shadow:       iy + r
    #
    _ACTION_BADGE = ["~z", "WW", "WW", ">>", ">>"]   # rest / farm / hunt

    def _draw_villager(self, v, selected_uid):
        ix, iy = int(v.x), int(v.y)

        if v.is_child:
            img = self.img_boy if v.gender == "M" else self.img_girl
        else:
            img = self.img_villager_male if v.gender == "M" else self.img_villager_female

        size = int(v.radius * 3)
        img = self._get_sprite(img, size)

        rect = img.get_rect(center=(ix, iy))
        self.screen.blit(img, rect)

        # ── HP life line above villager ──
        bar_w = int(v.radius * 3)
        bar_h = 4

        x = ix - bar_w // 2
        y = iy - v.radius - 10

        max_hp = getattr(v, "max_health", 100.0)

        hp = v.health / max(1.0, max_hp)
        hp = max(0.0, min(1.0, hp))

        # background
        pygame.draw.rect(self.screen, (30, 22, 12), (x, y, bar_w, bar_h), border_radius=2)

        # ── color interpolation: green → yellow → red ──
        if hp > 0.5:
            # green → yellow
            t = (1.0 - hp) * 2
            r = int(62 + (220 - 62) * t)
            g = int(200 + (220 - 200) * t)
            b = int(82 + (48 - 82) * t)
        else:
            # yellow → red
            t = (0.5 - hp) * 2
            r = int(220 + (220 - 220) * t)
            g = int(220 + (62 - 220) * t)
            b = int(48 + (48 - 48) * t)

        fill_col = (r, g, b)

        fill_w = int(bar_w * hp)

        if fill_w > 0:
            pygame.draw.rect(
                self.screen,
                fill_col,
                (x, y, fill_w, bar_h),
                border_radius=2
            )

            # subtle highlight (gives “gloss”)
            hl = tuple(min(255, c + 60) for c in fill_col)
            pygame.draw.rect(
                self.screen,
                hl,
                (x + 1, y + 1, max(0, fill_w - 2), max(1, bar_h // 2)),
                border_radius=2
            )

        # outline
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, bar_w, bar_h), 1, border_radius=2)

    # ── LLM speech bubbles ────────────────────────────────────────────────────

    def _draw_speech_bubbles(self, world: "World") -> None:
        for v in world.alive():
            if not v.bubble_text:
                continue
            text_surf = self.f_small.render(v.bubble_text, True, (240, 230, 170))
            tw, th = text_surf.get_size()
            bx = int(v.x) - tw // 2 - 5
            by = int(v.y) - 38 - th
            bx = max(2, min(bx, self.px - tw - 12))
            by = max(2, by)
            pygame.draw.rect(self.screen, (35, 28, 12), (bx, by, tw + 10, th + 6), border_radius=4)
            pygame.draw.rect(self.screen, (180, 155, 80), (bx, by, tw + 10, th + 6), 1, border_radius=4)
            self.screen.blit(text_surf, (bx + 5, by + 3))

    def _draw_discussion_banner(self, world: "World") -> None:
        if not world.discussion_active:
            return
        bw, bh = self.px, 26
        surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        surf.fill((15, 10, 5, 200))
        self.screen.blit(surf, (0, self.H // 2 - bh // 2))
        label = self.f_head.render("⬤  Villagers discussing...  waiting for wisdom", True, (220, 195, 90))
        self.screen.blit(label, (bw // 2 - label.get_width() // 2, self.H // 2 - label.get_height() // 2))

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, world: "World") -> None:
        # gradient background
        self.screen.blit(self._hud_bg, (0, 0))
        # bottom border — bright + shadow
        _hline(self.screen, HUD_H-2, 0, self.W, DIVIDER_BRIGHT)
        _hline(self.screen, HUD_H-1, 0, self.W, DIVIDER)

        # title
        _txt(self.screen, self.f_title, "THE VILLAGE", 12, 6, ACCENT_GOLD)

        stats = world.get_stats()
        items = [
            (f"  {stats['sim_time']}",       TEXT_DIM),
            (f"  {stats['alive']}",          TEXT_PRIMARY),
            (f"♂ {stats['males']}",          (148, 188, 238)),
            (f"♀ {stats['females']}",        (238, 142, 182)),
            (f"✦ {stats['children']}",       (232, 185, 68)),
            (f"Gen {stats['generation']}",   ACCENT_GOLD),
            (f"↑ {stats['total_born']}",     (142, 222, 102)),
            (f"↓ {stats['total_deaths']}",   (218, 100, 80)),
            (f"× {stats['speed']}",          TEXT_SECONDARY),
        ]
        if world.paused:
            items.insert(0, ("  ■ PAUSED ", (235, 80, 60)))

        cx    = 168
        dot_r = self.f_hud_sm.render("·", True, TEXT_MUTED)
        for txt, col in items:
            if cx > 168:
                self.screen.blit(dot_r, (cx - 10, 14))
            r = self.f_hud.render(txt, True, col)
            self.screen.blit(r, (cx, 11))
            cx += r.get_width() + 20
            if cx > self.px - 150:
                break

        hint = "SPACE pause  ·  R reset  ·  +/- speed  ·  click to select"
        hr   = self.f_hud_sm.render(hint, True, TEXT_MUTED)
        self.screen.blit(hr, (self.px - hr.get_width() - 10, 14))

    # ── panel ─────────────────────────────────────────────────────────────────
    def _draw_panel(self, world: "World") -> None:
        px, pw, H = self.px, self.pw, self.H

        # HARD CLIP (real fix)
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(px, 0, pw, H))

        # background
        self.screen.blit(self._panel_bg, (px, 0))

        # borders
        pygame.draw.line(self.screen, DIVIDER_BRIGHT, (px, 0), (px, H), 1)
        pygame.draw.line(self.screen, DIVIDER, (px + 1, 0), (px + 1, H), 1)

        y = HUD_H + 8

        # ── title ──────────────────────────────────────────────────────────
        title = self.f_title.render("The Village", True, ACCENT_GOLD)
        self.screen.blit(title, (px + pw // 2 - title.get_width() // 2, y))
        y += title.get_height() + 6

        _hline(self.screen, y, px + 8, self.W - 8, DIVIDER_BRIGHT)
        _hline(self.screen, y + 2, px + 18, self.W - 18, DIVIDER)
        y += 12

        # ── CENSUS ─────────────────────────────────────────────────────────
        y += _sec_header(self.screen, self.f_head, "CENSUS", px, y, pw)

        stats = world.get_stats()
        rows = [
            ("Generation", str(stats["generation"]), ACCENT_GOLD),
            ("Population", str(stats["alive"]), TEXT_PRIMARY),
            ("  ♂ Males", str(stats["males"]), (152, 190, 238)),
            ("  ♀ Females", str(stats["females"]), (238, 152, 190)),
            ("  ✦ Kids", str(stats["children"]), (232, 190, 85)),
            ("Born", str(stats["total_born"]), (145, 220, 105)),
            ("Deaths", str(stats["total_deaths"]), (222, 108, 85)),
        ]

        for i, (label, val, col) in enumerate(rows):
            if i % 2 == 1:
                rb = pygame.Surface((pw - 14, 16), pygame.SRCALPHA)
                pygame.draw.rect(rb, (255, 255, 255, 9), (0, 0, pw - 14, 16), border_radius=2)
                self.screen.blit(rb, (px + 7, y))

            _txt(self.screen, self.f_small, label, px + 13, y + 1, TEXT_SECONDARY)
            _txt_r(self.screen, self.f_small, val, self.W - 10, y + 1, col)
            y += 16

        y += 10
        _hline(self.screen, y, px + 8, self.W - 8, DIVIDER)
        y += 10

        # ── LEGEND ─────────────────────────────────────────────────────────
        y += _sec_header(self.screen, self.f_head, "LEGEND", px, y, pw)

        col_w = (pw - 20) // 3
        legend = [
            (cfg.C_MALE, "♂", "Male"),
            (cfg.C_FEMALE, "♀", "Female"),
            (cfg.C_CHILD, "✦", "Child"),
        ]

        for ci, (c, sym, label) in enumerate(legend):
            ax = px + 10 + ci * col_w
            pygame.draw.circle(self.screen, c, (ax + 6, y + 6), 7)
            pygame.draw.circle(self.screen, (0, 0, 0), (ax + 6, y + 6), 7, 1)
            _txt(self.screen, self.f_tiny, f"{sym} {label}", ax + 17, y + 2, TEXT_SECONDARY)

        y += 18

        for ci, (dc, label) in enumerate([(COL_TRAIN, "Training"), (COL_TEST, "Testing")]):
            ax = px + 10 + ci * (col_w + col_w // 2)
            pygame.draw.circle(self.screen, (0, 0, 0), (ax + 6, y + 6), 6)
            pygame.draw.circle(self.screen, dc, (ax + 6, y + 6), 5)
            _txt(self.screen, self.f_tiny, label, ax + 16, y + 2, TEXT_SECONDARY)

        y += 18
        _hline(self.screen, y, px + 8, self.W - 8, DIVIDER)
        y += 10

        # ── ACTIONS ────────────────────────────────────────────────────────
        y += _sec_header(self.screen, self.f_head, "ACTIONS", px, y, pw)

        half_w = (pw - 20) // 2
        ax = px + 10

        for i, (name, col) in enumerate(zip(cfg.ACTION_NAMES, ACTION_COLS)):
            pygame.draw.rect(self.screen, col, (ax, y + 4, 10, 9), border_radius=2)
            _txt(self.screen, self.f_tiny, name, ax + 14, y + 2, TEXT_SECONDARY)

            ax += half_w + 5
            if i % 2 == 1:
                ax = px + 10
                y += 15

        if len(cfg.ACTION_NAMES) % 2 == 1:
            y += 15

        y += 10
        _hline(self.screen, y, px + 8, self.W - 8, DIVIDER)
        y += 10

        # ── DEATHS ────────────────────────────────────────────────────────
        log_h   = 104
        log_top = H - log_h
        y += _sec_header(self.screen, self.f_head, "DEATHS", px, y, pw)

        total = world.total_deaths
        # total row
        _txt(self.screen, self.f_small, "Total", px + 13, y + 1, TEXT_SECONDARY)
        _txt_r(self.screen, self.f_small, str(total), self.W - 10, y + 1, (222, 108, 85))
        y += 18
        _hline(self.screen, y, px + 14, self.W - 14, DIVIDER)
        y += 8

        cause_rows = [
            ("old age",           "☽ Old Age",       (200, 180, 100)),
            ("starvation",        "☠ Starvation",    (220,  90,  70)),
            ("hunting mishap",    "⚔ Hunting",       (210, 130,  50)),
            ("troll attack",      "👹 Troll",         (190,  60, 200)),
            ("farming exhaustion","🌾 Farming",       ( 90, 180,  80)),
        ]
        bar_w = pw - 80

        for key, label, col in cause_rows:
            count = world.deaths_by_cause.get(key, 0)
            frac  = count / max(1, total)

            _txt(self.screen, self.f_small, label, px + 13, y + 2, col)
            _txt_r(self.screen, self.f_small, str(count), self.W - 10, y + 2, TEXT_PRIMARY)

            # thin proportion bar
            by = y + 14
            pygame.draw.rect(self.screen, (40, 30, 18), (px + 13, by, bar_w, 4), border_radius=2)
            if frac > 0:
                fw = max(2, int(bar_w * frac))
                pygame.draw.rect(self.screen, col, (px + 13, by, fw, 4), border_radius=2)
            y += 26

        y += 6
        _hline(self.screen, y, px + 8, self.W - 8, DIVIDER)
        y += 6

        # ── EVENTS (FIXED, ALWAYS ANCHORED) ───────────────────────────────
        _hline(self.screen, log_top - 4, px + 8, self.W - 8, DIVIDER_BRIGHT)

        ly = log_top
        ly += _sec_header(self.screen, self.f_head, "EVENTS", px, ly, pw)
        ly += 4

        for msg, kind in world.log[:5]:
            col = LOG_COLORS.get(kind, TEXT_SECONDARY)
            pygame.draw.rect(self.screen, col, (px + 10, ly + 2, 3, 12), border_radius=1)
            r = self.f_small.render(msg[:40], True, TEXT_SECONDARY)
            self.screen.blit(r, (px + 17, ly + 1))
            ly += 17

        self.screen.set_clip(prev_clip)

    def _draw_villager_card(self, v: "Villager", y: int) -> None:
        px, pw = self.px, self.pw
        cw = pw - 14
        cx = px + 7
        ch = 60

        tcol = TEXT_PRIMARY if v.alive else TEXT_DIM

        # card body
        cs = pygame.Surface((cw, ch), pygame.SRCALPHA)
        pygame.draw.rect(cs, (*CARD_BG, 235), (0, 0, cw, ch), border_radius=5)
        # top strip
        strip = v.color if v.alive else (55, 48, 42)
        pygame.draw.rect(cs, (*strip, 218), (0, 0, cw, 5), border_radius=5)
        # border
        pygame.draw.rect(cs, (*strip, 80 if v.alive else 45), (0, 0, cw, ch), 1, border_radius=5)
        # inner highlight beneath strip
        pygame.draw.rect(cs, (255, 255, 255, 15), (2, 6, cw-4, 8), border_radius=3)
        self.screen.blit(cs, (cx, y))

        # name + gender
        sym = GENDER_SYM.get(v.gender, "")
        _txt(self.screen, self.f_label, f"{sym} {v.display_name}", cx+9, y+8, tcol)

        # phase badge
        bx = cx + cw - 58
        if not v.alive:
            _pill(self.screen, self.f_tiny, "dead",  bx+6, y+8, (50, 30, 20), (175, 112, 92))
        elif v.is_child:
            _pill(self.screen, self.f_tiny, "child", bx+2, y+8, (70, 50,  8), COL_CHILD)
        elif v.train_phase:
            _pill(self.screen, self.f_tiny, "train", bx+2, y+8, (15, 38, 75), COL_TRAIN)
        else:
            _pill(self.screen, self.f_tiny, "test",  bx+6, y+8, (62, 38,  8), COL_TEST)

        # current action + age
        if v.alive:
            acol = ACTION_COLS[v.action]
            pygame.draw.rect(self.screen, acol, (cx+9, y+24, 8, 8), border_radius=2)
            _txt(self.screen,   self.f_tiny, cfg.ACTION_NAMES[v.action], cx+20, y+24, acol)
            _txt_r(self.screen, self.f_tiny,
                   f"age {v.age:.0f}/{v.max_age:.0f}",
                   cx+cw-5, y+24, TEXT_DIM)

        # HP / Risk bars
        bar_y = y + 38
        half  = (cw - 24) // 2
        _txt(self.screen, self.f_tiny, "HP", cx+9,        bar_y,   TEXT_DIM)
        _bar_fancy(self.screen, cx+24, bar_y+1, half-2, 9, v.health / max(v.max_health, 1.0), COL_HEALTH)
        _txt(self.screen, self.f_tiny, "RK", cx+28+half,  bar_y,   TEXT_DIM)
        _bar_fancy(self.screen, cx+42+half, bar_y+1, half-4, 9,
                   min(v.risk, 100)/100, COL_RISK)

        # policy sparkbar
        if v.alive and v.last_state is not None:
            try:
                probs, _, _ = v.brain.forward(v.last_state)
                _sparkbar(self.screen, cx+9, y+52, cw-18, 4,
                          list(probs), ACTION_COLS, gap=2)
            except Exception:
                pass

    # ── tooltip ────────────────────────────────────────────────────────────────

    def _draw_tooltip(self, v: "Villager", world: "World") -> None:
        from helpers import dist_vv
        alive     = world.alive()
        near      = [o for o in alive if o.uid != v.uid and dist_vv(v, o) < 70]
        kids_near = 1.0 if any(o.is_child for o in near) else 0.0
        mate_near = 1.0 if any(o.is_adult and o.gender != v.gender for o in near) else 0.0
        state     = v.get_state(kids_near, mate_near)
        probs, value, _ = v.brain.forward(state)

        tw, th = 255, 282
        tx = int(v.x) - tw - 26 if v.x > self.W // 2 else int(v.x) + 26
        ty = max(HUD_H + 4, min(self.H - th - 8, int(v.y) - th // 2))

        # card background
        ts = pygame.Surface((tw, th), pygame.SRCALPHA)
        pygame.draw.rect(ts, (*PANEL_TOP, 252), (0, 0, tw, th), border_radius=8)
        pygame.draw.rect(ts, (*DIVIDER_BRIGHT, 218), (0, 0, tw, th), 1, border_radius=8)
        # colour strip
        pygame.draw.rect(ts, (*v.color, 238), (0, 0, tw, 5), border_radius=8)
        # inner highlight
        pygame.draw.rect(ts, (255, 255, 255, 18), (2, 6, tw-4, 12), border_radius=6)
        self.screen.blit(ts, (tx, ty))

        iy = ty + 12
        sym   = GENDER_SYM.get(v.gender, "")
        title = f"{sym}  {v.name}" + (" (child)" if v.is_child else "")
        _txt(self.screen, self.f_tip_h, title, tx+12, iy, ACCENT_GOLD)
        iy += 22
        _hline(self.screen, iy,     tx+10, tx+tw-10, DIVIDER_BRIGHT)
        _hline(self.screen, iy + 1, tx+18, tx+tw-18, DIVIDER)
        iy += 8

        stat_lines = [
            ("Age",    f"{v.age:.0f}s / {v.max_age:.0f}s",  TEXT_SECONDARY),
            ("Health", f"{v.health:.1f}%",                   COL_HEALTH),
            ("Risk",   f"{v.risk:.1f}%",                     COL_RISK),
            ("Phase",  "Training" if v.train_phase else "Testing",
             COL_TRAIN if v.train_phase else COL_TEST),
            ("Action", cfg.ACTION_NAMES[v.action],           ACTION_COLS[v.action]),
            ("Reward", f"{v.total_reward:.1f}",              ACCENT_WARM),
            ("Value",  f"{value:.3f}",                       TEXT_SECONDARY),
            ("Memory", str(len(v.brain.memory)),             TEXT_DIM),
        ]
        for i, (label, val, col) in enumerate(stat_lines):
            if i % 2 == 1:
                rb = pygame.Surface((tw-16, 15), pygame.SRCALPHA)
                pygame.draw.rect(rb, (255, 255, 255, 9), (0, 0, tw-16, 15), border_radius=2)
                self.screen.blit(rb, (tx+8, iy))
            _txt(self.screen,   self.f_tip, label, tx+12, iy+1, TEXT_DIM)
            _txt_r(self.screen, self.f_tip, val,   tx+tw-12, iy+1, col)
            iy += 15

        _hline(self.screen, iy+2, tx+10, tx+tw-10, DIVIDER_BRIGHT)
        _hline(self.screen, iy+3, tx+18, tx+tw-18, DIVIDER)
        iy += 10

        # policy bars
        _txt(self.screen, self.f_tip, "POLICY DISTRIBUTION", tx+12, iy, TEXT_DIM)
        iy += 14
        bw = tw - 24
        for i, (name, prob) in enumerate(zip(cfg.ACTION_NAMES, probs)):
            col  = ACTION_COLS[i]
            fill = max(2, int(bw * prob))
            pygame.draw.rect(self.screen, (30, 22, 11), (tx+12, iy, bw, 13), border_radius=3)
            pygame.draw.rect(self.screen, col,          (tx+12, iy, fill, 13), border_radius=3)
            hl = tuple(min(255, c + 70) for c in col)
            if fill > 4:
                pygame.draw.rect(self.screen, hl, (tx+13, iy+1, fill-2, 4), border_radius=3)
            lbl = self.f_tiny.render(f"{name[:7]} {prob*100:.0f}%", True,
                                     TEXT_PRIMARY if fill > 54 else TEXT_DIM)
            self.screen.blit(lbl, (tx+14, iy+2))
            iy += 17
