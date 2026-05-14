"""
main.py
-------
Entry point for the Village Actor-Critic DRL Ecosystem.

Usage
~~~~~
    python main.py

Controls
~~~~~~~~
    Space       pause / resume
    R           full reset
    +  /  =     increase simulation speed
    -           decrease simulation speed
    Click       select a villager (shows live policy tooltip)
    Esc / Q     quit
"""

import math
import sys
from typing import TYPE_CHECKING

import pygame

import config as cfg
from simulation import World
from renderer import Renderer
from stats_logger import StatsLogger

if TYPE_CHECKING:
    pass

# ── slider state ──────────────────────────────────────────────────────────────

_slider_dragging = False


def _apply_slider_x(world: World, mx: int, slider: tuple) -> None:
    xl, xr, _ = slider
    frac = max(0.0, min(1.0, (mx - xl) / (xr - xl)))
    world.speed = max(
        cfg.SIM_SPEED_MIN,
        min(cfg.SIM_SPEED_MAX, round(cfg.SIM_SPEED_MIN + frac * (cfg.SIM_SPEED_MAX - cfg.SIM_SPEED_MIN))),
    )


def _hit_slider(mx: int, my: int, slider: tuple) -> bool:
    xl, xr, cy = slider
    return (xl - 8) <= mx <= (xr + 8) and abs(my - cy) <= 10


# ── event handling ────────────────────────────────────────────────────────────

def handle_events(world: World, renderer: Renderer) -> bool:
    """
    Process all pending pygame events.

    Returns
    -------
    bool : False if the application should quit, True otherwise.
    """
    global _slider_dragging

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False

            elif event.key == pygame.K_SPACE:
                world.paused = not world.paused

            elif event.key == pygame.K_r:
                world.reset()

            elif event.key in (
                pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS
            ):
                world.speed = min(cfg.SIM_SPEED_MAX, world.speed + 1)

            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                world.speed = max(cfg.SIM_SPEED_MIN, world.speed - 1)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            sl = renderer._slider
            if sl and _hit_slider(mx, my, sl):
                _slider_dragging = True
                _apply_slider_x(world, mx, sl)
            elif mx < cfg.WINDOW_W - cfg.PANEL_WIDTH:
                # villager selection (world area only)
                found = next(
                    (
                        v for v in world.alive()
                        if math.hypot(mx - v.x, my - v.y) < 16
                    ),
                    None,
                )
                world.selected_uid = found.uid if found else None

        elif event.type == pygame.MOUSEMOTION:
            if _slider_dragging:
                sl = renderer._slider
                if sl:
                    _apply_slider_x(world, event.pos[0], sl)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            _slider_dragging = False

    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.display.set_caption(cfg.TITLE)

    screen = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
    clock  = pygame.time.Clock()

    world    = World()
    renderer = Renderer(screen)
    logger   = StatsLogger("stats_log.csv")

    running = True
    while running:
        dt = clock.tick(cfg.FPS) / 1000.0   # real-time delta in seconds

        running = handle_events(world, renderer)
        world.step(dt)
        logger.update(world)
        renderer.draw(world)
        pygame.display.flip()

        if world.generation >= 2000 or world.sim_time >= 300_000:
            print(
                f"Simulation ended — generation {world.generation}, "
                f"sim_time {world.sim_time:.1f}s"
            )
            running = False

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
