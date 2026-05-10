"""
world.py – Spielwelt für das Neuro-Ökosystem.

Enthält:
- World: Konfigurierbare Welt mit Wänden, zufälligen Hindernissen,
  Batterie-Pool mit Respawn-System und Kollisionserkennung.
"""

import random
import math
import pygame
from config.config_manager import SimConfig
from core.entities import Battery, Robot


# ─── Farb-Konstanten ────────────────────────────────────────────────────────────
COLOR_BG = (12, 12, 18)
COLOR_WALL = (60, 60, 80)
COLOR_OBSTACLE = (45, 45, 65)
COLOR_OBSTACLE_BORDER = (70, 70, 100)
COLOR_GRID = (20, 20, 28)


class World:
    """Spielwelt mit Wänden, Hindernissen und Batterien.

    Die Welt besteht aus einem konfigurierbaren N×N Raster.
    Hindernisse werden als Rechtecke zufällig platziert.
    Batterien spawnen mit einem Timer-System.

    Attributes:
        config: Simulationskonfiguration
        width: Breite der Welt in Pixeln
        height: Höhe der Welt in Pixeln
        walls: Liste von Wandsegmenten [(x1,y1,x2,y2), ...]
        obstacles: Liste von Hindernis-Rechtecken [pygame.Rect, ...]
        batteries: Liste aller Batterien
    """

    WALL_THICKNESS = 4
    MIN_OBSTACLE_GAP = 60  # Mindestabstand zwischen Hindernissen

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.width = config.window_width
        self.height = config.window_height
        self.walls: list[tuple[int, int, int, int]] = []
        self.obstacles: list[pygame.Rect] = []
        self.batteries: list[Battery] = []

        self._create_walls()
        self._create_obstacles()
        self._create_batteries()

    def _create_walls(self) -> None:
        """Erstellt die 4 Außenwände als Liniensegmente."""
        w, h = self.width, self.height
        self.walls = [
            (0, 0, w, 0),       # Oben
            (w, 0, w, h),       # Rechts
            (w, h, 0, h),       # Unten
            (0, h, 0, 0),       # Links
        ]

    def _create_obstacles(self) -> None:
        """Erstellt zufällige Rechteck-Hindernisse.

        Hindernisse sind 2-4 Felder breit/hoch, haben einen Mindestabstand
        voneinander und zum Spielfeldrand.
        """
        self.obstacles.clear()
        cell = self.config.cell_pixel_size
        margin = cell * 3  # Mindestabstand zum Rand (3 Felder)

        for _ in range(self.config.obstacle_count):
            placed = False
            for attempt in range(50):  # Max 50 Versuche pro Hindernis
                # Zufällige Größe: 2-4 Felder
                obs_w = random.randint(2, 4) * cell
                obs_h = random.randint(2, 4) * cell

                # Zufällige Position (auf Raster ausgerichtet)
                max_x = self.width - margin - obs_w
                max_y = self.height - margin - obs_h
                if max_x <= margin or max_y <= margin:
                    continue
                obs_x = random.randint(margin // cell, max_x // cell) * cell
                obs_y = random.randint(margin // cell, max_y // cell) * cell

                new_rect = pygame.Rect(obs_x, obs_y, obs_w, obs_h)

                # Prüfe Mindestabstand zu bestehenden Hindernissen
                too_close = False
                for existing in self.obstacles:
                    expanded = existing.inflate(self.MIN_OBSTACLE_GAP,
                                                self.MIN_OBSTACLE_GAP)
                    if expanded.colliderect(new_rect):
                        too_close = True
                        break

                if not too_close:
                    self.obstacles.append(new_rect)
                    # Hindernis-Kanten als Wand-Segmente hinzufügen
                    self._add_obstacle_walls(new_rect)
                    placed = True
                    break

            if not placed:
                print(f"[WORLD] Hindernis konnte nicht platziert werden "
                      f"(nach 50 Versuchen)")

    def _add_obstacle_walls(self, rect: pygame.Rect) -> None:
        """Fügt die Kanten eines Hindernisses als Wandsegmente hinzu."""
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        self.walls.extend([
            (x, y, x + w, y),         # Oben
            (x + w, y, x + w, y + h), # Rechts
            (x + w, y + h, x, y + h), # Unten
            (x, y + h, x, y),         # Links
        ])

    def _create_batteries(self) -> None:
        """Erstellt den initialen Batterie-Pool."""
        self.batteries.clear()
        for _ in range(self.config.battery_count):
            x, y = self._random_free_position(Battery.RADIUS)
            self.batteries.append(Battery(x, y, self.config))

    def _random_free_position(self, radius: float) -> tuple[float, float]:
        """Findet eine zufällige Position, die nicht in einem Hindernis liegt.

        Args:
            radius: Radius des zu platzierenden Objekts.

        Returns:
            (x, y) Tuple mit gültiger Position.
        """
        margin = 30
        for _ in range(200):
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - margin)
            point_rect = pygame.Rect(x - radius, y - radius,
                                     radius * 2, radius * 2)
            valid = True
            for obs in self.obstacles:
                if obs.colliderect(point_rect):
                    valid = False
                    break
            if valid:
                return (float(x), float(y))
        # Fallback
        return (float(random.randint(margin, self.width - margin)),
                float(random.randint(margin, self.height - margin)))

    def spawn_robot_position(self) -> tuple[float, float]:
        """Gibt eine gültige Spawn-Position für einen Roboter zurück."""
        return self._random_free_position(self.config.robot_radius)

    def update(self) -> None:
        """Aktualisiert die Welt (Batterie-Respawn-Timer)."""
        for battery in self.batteries:
            battery.update(self.width, self.height, self.obstacles)

    def check_battery_collision(self, robot: Robot) -> bool:
        """Prüft ob ein Roboter eine aktive Batterie berührt.

        Args:
            robot: Der zu prüfende Roboter.

        Returns:
            True wenn eine Batterie eingesammelt wurde.
        """
        if not robot.alive:
            return False

        for battery in self.batteries:
            if not battery.active:
                continue
            dx = robot.x - battery.x
            dy = robot.y - battery.y
            dist_sq = dx * dx + dy * dy
            collect_dist = robot.radius + Battery.RADIUS
            if dist_sq < collect_dist * collect_dist:
                battery.collect()
                robot.add_energy(self.config.battery_energy)
                return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        """Zeichnet die Welt: Hintergrund, Grid, Hindernisse, Batterien."""
        # Hintergrund
        screen.fill(COLOR_BG)

        # Subtiles Grid
        cell = self.config.cell_pixel_size
        for x in range(0, self.width, cell):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, cell):
            pygame.draw.line(screen, COLOR_GRID, (0, y), (self.width, y))

        # Hindernisse
        for obs in self.obstacles:
            pygame.draw.rect(screen, COLOR_OBSTACLE, obs)
            pygame.draw.rect(screen, COLOR_OBSTACLE_BORDER, obs, 2)

        # Außenwände (dicker Rahmen)
        pygame.draw.rect(screen, COLOR_WALL,
                         (0, 0, self.width, self.height),
                         self.WALL_THICKNESS)

        # Batterien
        for battery in self.batteries:
            battery.draw(screen)
