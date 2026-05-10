"""
entities.py – Spielentitäten für das Neuro-Ökosystem.

Enthält:
- Battery: Sammelbare Energiequelle mit Respawn-Timer
- Robot: Basisklasse mit Differential-Drive Kinematik und Energiesystem
- Collector: Sammler-Roboter (grün)
- Hunter: Jäger-Roboter (rot, Platzhalter für Stufe 4)
"""

import math
import random
import pygame
from config.config_manager import SimConfig


# ─── Entity-Typ-Konstanten (ersetzt isinstance() in Hot-Loops) ──────────────────
ENTITY_BATTERY = 0
ENTITY_COLLECTOR = 1
ENTITY_HUNTER = 2


# ─── Farb-Konstanten ────────────────────────────────────────────────────────────
COLOR_BATTERY_ACTIVE = (255, 230, 50)
COLOR_BATTERY_GLOW = (255, 255, 100, 80)
COLOR_BATTERY_INACTIVE = (80, 75, 30)
COLOR_COLLECTOR = (0, 230, 120)
COLOR_COLLECTOR_DIM = (0, 160, 80)
COLOR_HUNTER = (230, 50, 50)
COLOR_HUNTER_DIM = (160, 30, 30)
COLOR_DIRECTION = (255, 255, 255)
COLOR_ENERGY_BG = (40, 40, 50)
COLOR_ENERGY_HIGH = (0, 230, 120)
COLOR_ENERGY_MED = (230, 200, 50)
COLOR_ENERGY_LOW = (230, 50, 50)


class Battery:
    """Sammelbare Batterie mit Respawn-Timer-System.

    Batterien sind statische Objekte, die von Robotern eingesammelt werden können.
    Nach dem Einsammeln werden sie für eine konfigurierbare Zeitspanne inaktiv
    und spawnen dann an einer neuen Zufallsposition.
    """

    RADIUS = 12

    def __init__(self, x: float, y: float, config: SimConfig) -> None:
        self.x = x
        self.y = y
        self.config = config
        self.active = True
        self.respawn_timer = 0
        self.glow_phase = random.uniform(0, math.pi * 2)  # Für Glow-Animation
        self.entity_type = ENTITY_BATTERY  # Typ-Flag für schnelle Abfragen

    def collect(self) -> None:
        """Markiert die Batterie als eingesammelt und startet den Respawn-Timer."""
        self.active = False
        self.respawn_timer = self.config.battery_respawn_delay

    def update(self, world_width: int, world_height: int,
               obstacles: list[pygame.Rect]) -> None:
        """Aktualisiert den Respawn-Timer und spawnt ggf. an neuer Position.

        Args:
            world_width: Breite der Welt in Pixeln.
            world_height: Höhe der Welt in Pixeln.
            obstacles: Liste der Hindernis-Rechtecke zum Vermeiden.
        """
        if not self.active:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self._respawn(world_width, world_height, obstacles)
        self.glow_phase += 0.05  # Glow-Animation fortschreiten

    def _respawn(self, world_width: int, world_height: int,
                 obstacles: list[pygame.Rect]) -> None:
        """Spawnt die Batterie an einer neuen, gültigen Zufallsposition."""
        margin = 30
        for _ in range(100):  # Max 100 Versuche
            new_x = random.randint(margin, world_width - margin)
            new_y = random.randint(margin, world_height - margin)
            # Prüfe ob Position in einem Hindernis liegt
            point_rect = pygame.Rect(new_x - self.RADIUS, new_y - self.RADIUS,
                                     self.RADIUS * 2, self.RADIUS * 2)
            collision = False
            for obs in obstacles:
                if obs.colliderect(point_rect):
                    collision = True
                    break
            if not collision:
                self.x = new_x
                self.y = new_y
                self.active = True
                return
        # Fallback: einfach irgendwo spawnen
        self.x = random.randint(margin, world_width - margin)
        self.y = random.randint(margin, world_height - margin)
        self.active = True

    def draw(self, screen: pygame.Surface) -> None:
        """Zeichnet die Batterie mit Glow-Effekt."""
        if not self.active:
            # Inaktive Batterie: gedimmt und kleiner zeichnen
            pygame.draw.circle(screen, COLOR_BATTERY_INACTIVE,
                               (int(self.x), int(self.y)), self.RADIUS - 2)
            return

        # Glow-Effekt (pulsierend)
        glow_intensity = 0.6 + 0.4 * math.sin(self.glow_phase)
        glow_radius = int(self.RADIUS + 6 * glow_intensity)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_alpha = int(40 * glow_intensity)
        pygame.draw.circle(glow_surf, (255, 255, 100, glow_alpha),
                           (glow_radius, glow_radius), glow_radius)
        screen.blit(glow_surf,
                    (int(self.x) - glow_radius, int(self.y) - glow_radius))

        # Hauptkörper
        pygame.draw.circle(screen, COLOR_BATTERY_ACTIVE,
                           (int(self.x), int(self.y)), self.RADIUS)
        # Blitz-Symbol (vereinfacht als heller Punkt)
        pygame.draw.circle(screen, (255, 255, 200),
                           (int(self.x), int(self.y)), 3)


class Robot:
    """Basisklasse für alle Roboter mit Differential-Drive Kinematik.

    Verwendet ein Panzersteuerungs-Modell: Linker und rechter Motor
    können unabhängig gesteuert werden. Die resultierende Bewegung
    ergibt sich aus der Kombination beider Motorwerte.

    Attribute:
        x, y: Position in Pixeln
        angle: Blickrichtung in Radiant
        speed: Basis-Geschwindigkeit
        radius: Kollisionsradius
        energy: Aktueller Energielevel
        alive: Ob der Roboter noch lebt
        color: Hauptfarbe (wird von Subklassen gesetzt)
        fitness: Akkumulierte Fitness (für NEAT)
    """

    WHEEL_BASE = 20.0  # Abstand zwischen den Rädern (für Differential Drive)

    def __init__(self, x: float, y: float, config: SimConfig,
                 color: tuple[int, int, int] = (200, 200, 200)) -> None:
        self.x = x
        self.y = y
        self.config = config
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = config.collector_speed
        self.radius = config.robot_radius
        self.energy = config.energy_start
        self.alive = True
        self.color = color
        self.fitness = 0.0

        # Referenzen für NEAT (werden in Stufe 3 gesetzt)
        self.genome = None
        self.net = None

        # Motor-Werte der letzten Aktion
        self.motor_left = 0.0
        self.motor_right = 0.0

        # Sensor-Daten: Liste von (dist_norm, obj_type, hit_x, hit_y)
        self.sensor_data: list[tuple[float, int, float, float]] = []

        # Flag für Hall of Fame Injektion
        self.is_injected = False

        # Funk-Antenne (Stufe 5)
        self.radio_out = 0.0
        self.radio_in = 0.0
        self.entity_type = -1  # Wird von Subklassen gesetzt
        
        # Fitness-Tracking fuer Debugging/Analytics
        self.fit_battery = 0.0
        self.fit_prox = 0.0
        self.fit_hunter_pen = 0.0
        self.fit_escape = 0.0
        self.fit_approach = 0.0
        self.fit_death = 0.0
        self.fit_surv = 0.0
        self.fit_idle = 0.0

    def update_sensors(self, walls: list, batteries: list, robots: list) -> None:
        """Aktualisiert die Sensordaten durch Raycasting.

        Args:
            walls: Liste der Wandsegmente.
            batteries: Liste aller Batterien.
            robots: Liste aller Roboter.
        """
        if not self.alive:
            self.sensor_data = []
            return
        from core.sensors import cast_rays
        self.sensor_data = cast_rays(self, walls, batteries, robots, self.config)

    def get_sensor_inputs(self) -> list[float]:
        """Gibt die Sensordaten als flache Liste für das neuronale Netz zurück.

        Returns:
            Liste mit [dist, is_battery, is_hunter, is_wall, is_collector] pro Strahl.
        """
        n = len(self.sensor_data)
        inputs = [0.0] * (n * 5)  # Pre-allokiert statt append()
        idx = 0
        for dist_norm, obj_type, _, _ in self.sensor_data:
            inputs[idx] = dist_norm
            inputs[idx + 1] = 1.0 if obj_type == 2 else 0.0  # Batterie?
            inputs[idx + 2] = 1.0 if obj_type == 4 else 0.0  # Jäger? (GEFAHR!)
            inputs[idx + 3] = 1.0 if obj_type == 1 else 0.0  # Wand?
            inputs[idx + 4] = 1.0 if obj_type == 3 else 0.0  # Sammler? (Schwarm!)
            idx += 5
        return inputs

    def move(self, left_motor: float, right_motor: float, radio_out: float = 0.0, dt: float = 1.0) -> None:
        """Bewegt den Roboter basierend auf Differential-Drive Kinematik.

        Args:
            left_motor: Motorwert links [-1.0, 1.0]
            right_motor: Motorwert rechts [-1.0, 1.0]
            dt: Zeitschritt (normalerweise 1.0 für frame-basiert)
        """
        if not self.alive:
            return

        # Motor-Werte auf [-1, 1] klemmen
        self.motor_left = max(-1.0, min(1.0, left_motor))
        self.motor_right = max(-1.0, min(1.0, right_motor))
        
        # Funksignal klemmen (nur positive Werte [0, 1])
        self.radio_out = max(0.0, min(1.0, radio_out))

        # Differential Drive Kinematik
        # Lineare Geschwindigkeit: v = speed * (left + right) / 2
        # Winkelgeschwindigkeit:   ω = speed * (right - left) / wheel_base
        v = self.speed * (self.motor_left + self.motor_right) / 2.0
        omega = self.speed * (self.motor_right - self.motor_left) / self.WHEEL_BASE

        # Position und Winkel aktualisieren
        self.angle += omega * dt
        # Winkel normalisieren auf [0, 2π]
        self.angle = self.angle % (2 * math.pi)

        self.x += v * math.cos(self.angle) * dt
        self.y += v * math.sin(self.angle) * dt

    def drain_energy(self) -> None:
        """Verringert die Energie und prüft auf Tod."""
        if not self.alive:
            return
        self.energy -= self.config.energy_drain_per_frame
        if self.energy <= self.config.energy_death_threshold:
            self.energy = 0.0
            self.alive = False

    def add_energy(self, amount: float) -> None:
        """Fügt Energie hinzu (z.B. durch Batterie-Einsammlung)."""
        self.energy = min(self.config.energy_start, self.energy + amount)

    def clamp_to_bounds(self, width: int, height: int) -> None:
        """Hält den Roboter innerhalb der Spielfeldgrenzen.

        Args:
            width: Breite des Spielfelds in Pixeln.
            height: Höhe des Spielfelds in Pixeln.
        """
        margin = self.radius
        if self.x < margin:
            self.x = margin
        elif self.x > width - margin:
            self.x = width - margin
        if self.y < margin:
            self.y = margin
        elif self.y > height - margin:
            self.y = height - margin

    def check_obstacle_collision_fast(self, obstacles_tuples: list[tuple[float, float, float, float]]) -> None:
        """Prüft und korrigiert Kollision mit Hindernissen (Performance-optimiert).

        Schiebt den Roboter aus überlappenden Hindernissen heraus.
        Nutzt vorberechnete Tupel (left, top, right, bottom) und if/else statt min/max
        um Millionen von Funktionsaufrufen zu sparen.

        Args:
            obstacles_tuples: Liste der Hindernisse als (left, top, right, bottom).
        """
        r_sq = self.radius * self.radius
        rx = self.x
        ry = self.y
        
        for left, top, right, bottom in obstacles_tuples:
            # Schneller Inline-Ersatz für:
            # closest_x = max(left, min(rx, right))
            # closest_y = max(top, min(ry, bottom))
            if rx < left:
                closest_x = left
            elif rx > right:
                closest_x = right
            else:
                closest_x = rx
                
            if ry < top:
                closest_y = top
            elif ry > bottom:
                closest_y = bottom
            else:
                closest_y = ry
                
            dx = rx - closest_x
            dy = ry - closest_y
            dist_sq = dx * dx + dy * dy

            if dist_sq < r_sq:
                # Kollision! Roboter herausschieben
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.01
                overlap = self.radius - dist
                if dist > 0:
                    rx += (dx / dist) * overlap
                    ry += (dy / dist) * overlap
                else:
                    rx += overlap
                    ry += overlap
                    
        self.x = rx
        self.y = ry

    def check_obstacle_collision(self, obstacles: list[pygame.Rect]) -> None:
        """Prüft und korrigiert Kollision mit Hindernissen.

        Schiebt den Roboter aus überlappenden Hindernissen heraus.

        Args:
            obstacles: Liste der Hindernis-Rechtecke.
        """
        for obs in obstacles:
            # Nächster Punkt auf dem Rechteck zum Roboter-Zentrum
            closest_x = max(obs.left, min(self.x, obs.right))
            closest_y = max(obs.top, min(self.y, obs.bottom))
            dx = self.x - closest_x
            dy = self.y - closest_y
            dist_sq = dx * dx + dy * dy

            if dist_sq < self.radius * self.radius:
                # Kollision! Roboter herausschieben
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.01
                overlap = self.radius - dist
                if dist > 0:
                    self.x += (dx / dist) * overlap
                    self.y += (dy / dist) * overlap
                else:
                    # Roboter exakt auf dem Rand: in zufällige Richtung schieben
                    self.x += overlap

    def draw(self, screen: pygame.Surface) -> None:
        """Zeichnet den Roboter als Kreis mit Richtungsanzeige und Energieleiste."""
        if not self.alive:
            return

        ix, iy = int(self.x), int(self.y)
        r = int(self.radius)

        # Körper
        pygame.draw.circle(screen, self.color, (ix, iy), r)
        # Rand (etwas dunkler)
        pygame.draw.circle(screen, tuple(max(0, c - 60) for c in self.color),
                           (ix, iy), r, 2)

        # Markierung für Hall-of-Fame Roboter (Blauer Punkt in der Mitte)
        if self.is_injected:
            pygame.draw.circle(screen, (0, 150, 255), (ix, iy), r // 2)

        # Richtungs-Linie
        dir_x = ix + int(math.cos(self.angle) * r * 1.5)
        dir_y = iy + int(math.sin(self.angle) * r * 1.5)
        pygame.draw.line(screen, COLOR_DIRECTION, (ix, iy), (dir_x, dir_y), 2)

        # Markierung für injizierte Hall-of-Fame Roboter (blauer Punkt)
        if getattr(self, 'is_injected', False):
            pygame.draw.circle(screen, (0, 100, 255), (ix, iy), int(r * 0.4))

        # Energie-Leiste über dem Roboter
        bar_width = r * 2
        bar_height = 3
        bar_x = ix - r
        bar_y = iy - r - 6
        energy_ratio = max(0.0, self.energy / self.config.energy_start)

        # Hintergrund
        pygame.draw.rect(screen, COLOR_ENERGY_BG,
                         (bar_x, bar_y, bar_width, bar_height))
        # Füllung (Farbe abhängig von Energie)
        if energy_ratio > 0.6:
            bar_color = COLOR_ENERGY_HIGH
        elif energy_ratio > 0.3:
            bar_color = COLOR_ENERGY_MED
        else:
            bar_color = COLOR_ENERGY_LOW
        fill_width = int(bar_width * energy_ratio)
        if fill_width > 0:
            pygame.draw.rect(screen, bar_color,
                             (bar_x, bar_y, fill_width, bar_height))


class Collector(Robot):
    """Sammler-Roboter: Sucht und sammelt Batterien.

    Grüne Farbe. Wird in Stufe 3 durch ein NEAT-Netz gesteuert.
    In Stufe 1 fährt er mit zufälligen Motorwerten.
    """

    def __init__(self, x: float, y: float, config: SimConfig) -> None:
        super().__init__(x, y, config, color=COLOR_COLLECTOR)
        self.speed = config.collector_speed
        self.batteries_collected = 0
        self.entity_type = ENTITY_COLLECTOR


class Hunter(Robot):
    """Jäger-Roboter: Jagt und fängt Sammler.

    Rote Farbe. Wird in Stufe 4 aktiviert und durch ein
    separates NEAT-Netz gesteuert.
    """

    def __init__(self, x: float, y: float, config: SimConfig) -> None:
        super().__init__(x, y, config, color=COLOR_HUNTER)
        self.speed = config.hunter_speed
        self.kills = 0
        self.entity_type = ENTITY_HUNTER

    def drain_energy(self) -> None:
        """Jäger verbrauchen Energie, sterben aber nie daran.
        
        So bleiben sie als Konstante Gefahrenquelle erhalten,
        damit die Sammler durchgehend einen Fluchtdruck haben.
        """
        if not self.alive:
            return
        self.energy -= self.config.energy_drain_per_frame
        if self.energy <= self.config.energy_death_threshold:
            self.energy = self.config.energy_death_threshold + 1.0
