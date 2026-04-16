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

import pygame

import config as cfg
from simulation import World
from renderer import Renderer


# ── event handling ────────────────────────────────────────────────────────────

def handle_events(world: World) -> bool:
    """
    Process all pending pygame events.

    Returns
    -------
    bool : False if the application should quit, True otherwise.
    """
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
            # Only consider clicks in the game world (left of panel)
            if mx < cfg.WINDOW_W - cfg.PANEL_WIDTH:
                found = next(
                    (
                        v for v in world.alive()
                        if math.hypot(mx - v.x, my - v.y) < 16
                    ),
                    None,
                )
                world.selected_uid = found.uid if found else None

    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.display.set_caption(cfg.TITLE)

    screen = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
    clock  = pygame.time.Clock()

    world    = World()
    renderer = Renderer(screen)

    running = True
    while running:
        dt = clock.tick(cfg.FPS) / 1000.0   # real-time delta in seconds

        running = handle_events(world)
        world.step(dt)
        renderer.draw(world)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
