"""
main.py – Einstiegspunkt für das Neuro-Ökosystem.

Stufe 0-3: Config-Menü → Hall of Fame → NEAT-Training.
Roboter lernen durch Neuroevolution, Batterien zu finden.
Die besten Gehirne werden in der Hall of Fame gespeichert.
"""

import sys
import ctypes
import warnings

# Unterdrücke die harmlose CuPy-Warnung bzgl. CUDA_PATH, da die Pip-Version ihre eigenen Binaries nutzt
warnings.filterwarnings("ignore", message="CUDA path could not be detected.*")

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1) # Behebt falsche Fenstergrößen bei Windows-Skalierung > 100%
except Exception:
    pass

import pygame
from config.config_manager import SimConfig, ConfigMenu
from ai.hall_of_fame import HallOfFame, HallOfFameMenu
from ai.neat_ai import run_neat_training


def main() -> None:
    """Hauptfunktion: Config-Menü → Hall of Fame → NEAT-Training.

    Flow:
    1. Config-Menü (Parameter anpassen)
    2. Hall of Fame Menü (Top-20 anzeigen, Genome auswählen)
    3. NEAT-Training starten (mit/ohne injizierte Genome)
    """
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
    pygame.init()

    # ── Config laden ──────────────────────────────────────────────────
    config = SimConfig.load()

    # ── Config-Menü ───────────────────────────────────────────────────
    menu = ConfigMenu(config)
    result = menu.run()

    if result is None:
        print("[ENGINE] Beendet (Config-Menü geschlossen)")
        pygame.quit()
        sys.exit(0)

    mode, config = result
    print(f"[ENGINE] Konfiguration uebernommen: "
          f"{config.grid_size}x{config.grid_size}, "
          f"{config.obstacle_count} Hindernisse, "
          f"{config.battery_count} Batterien")

    # ── Hall of Fame laden ────────────────────────────────────────────
    hall = HallOfFame()

    # ── Hall of Fame Menü (wenn Einträge vorhanden) ───────────────────
    injected_entries = []
    if hall.entries:
        screen = pygame.display.set_mode(
            (HallOfFameMenu.MENU_WIDTH, HallOfFameMenu.MENU_HEIGHT))
        hof_menu = HallOfFameMenu(hall)
        injected_entries = hof_menu.run(screen)

        if hof_menu.result is None:
            print("[ENGINE] Beendet (Hall of Fame geschlossen)")
            pygame.quit()
            sys.exit(0)

        if hof_menu.result == 'back':
            # Zurück zum Config-Menü – für jetzt einfach starten
            pass

        if injected_entries:
            print(f"[ENGINE] {len(injected_entries)} Genome aus Hall of Fame ausgewaehlt")
    else:
        print("[ENGINE] Hall of Fame leer - starte frisches Training")

    # ── Pygame-Fenster für Training ───────────────────────────────────
    if mode == 'test':
        screen = pygame.display.set_mode(config.window_size)
        pygame.display.set_caption("Neuro-Oekosystem - Test Modus")
    else:
        screen = pygame.display.set_mode((100, 100), pygame.HIDDEN)
        pygame.display.set_caption("Neuro-Oekosystem - Headless Training")
    clock = pygame.time.Clock()

    # ── NEAT-Training starten ─────────────────────────────────────────
    print(f"[ENGINE] Starte NEAT-Evolution in {mode} Modus...")
    run_neat_training(config, screen, clock, hall,
                      injected_entries if injected_entries else None,
                      mode=mode)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
