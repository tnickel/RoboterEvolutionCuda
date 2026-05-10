"""
config_manager.py – Zentrales Konfigurationssystem für das Neuro-Ökosystem.

Enthält:
- SimConfig: Dataclass mit allen Simulationsparametern
- ConfigMenu: Pygame-basiertes UI-Menü zur Konfigurationsanpassung
"""

import json
import os
import math
import pygame
from dataclasses import dataclass, field, fields, asdict


# ─── Farb-Konstanten für das Config-Menü ───────────────────────────────────────
COLOR_BG = (18, 18, 24)
COLOR_PANEL = (28, 28, 38)
COLOR_PANEL_HOVER = (35, 35, 48)
COLOR_ACCENT = (0, 200, 150)
COLOR_ACCENT_DIM = (0, 120, 90)
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (120, 120, 140)
COLOR_SLIDER_BG = (50, 50, 65)
COLOR_SLIDER_FILL = (0, 200, 150)
COLOR_SLIDER_KNOB = (255, 255, 255)
COLOR_BUTTON = (0, 180, 130)
COLOR_BUTTON_HOVER = (0, 220, 160)
COLOR_BUTTON_TEXT = (18, 18, 24)
COLOR_CATEGORY = (0, 200, 150)
COLOR_INPUT_BG = (40, 40, 55)
COLOR_INPUT_ACTIVE = (60, 60, 80)
COLOR_SCROLLBAR = (50, 50, 65)
COLOR_SCROLLBAR_THUMB = (100, 100, 120)


@dataclass
class SimConfig:
    """Zentrale Konfiguration für alle Simulationsparameter.

    Alle numerischen Werte der Simulation werden hier verwaltet.
    Die Config kann als JSON gespeichert und geladen werden.
    """

    # ── Welt ──
    grid_size: int = 40              # N×N Felder (1600x1600 Pixel)
    cell_pixel_size: int = 40        # Pixel pro Feld (Fenster wird deutlich größer!)
    obstacle_count: int = 25         # Anzahl zufälliger Hindernisse

    # ── Batterien ──
    battery_count: int = 100         # Gesamtzahl Batterien
    battery_respawn_delay: int = 0   # Frames bis Respawn (0 = Sofortiger Respawn an neuer Stelle)
    battery_energy: float = 30.0     # Energie pro aufgesammelter Batterie

    # ── Energie ──
    energy_start: float = 50.0       # Start-Energie jedes Roboters (Halbiert, erzwingt sofortiges Handeln)
    energy_drain_per_frame: float = 0.08  # Energieverlust pro Frame (Verdoppelt, Stillstand = Tod)
    energy_death_threshold: float = 0.0  # Tod bei diesem Wert

    # ── Roboter ──
    collector_speed: float = 6.5     # Geschwindigkeit Sammler (Deutlich schneller zur Flucht)
    hunter_speed: float = 3.8        # Geschwindigkeit Jäger (Aggressiver, erzwingt Fluchtreflex)
    robot_radius: float = 18.0       # Kollisionsradius Roboter
    sensor_ray_count: int = 7        # Anzahl Sensor-Strahlen (Erweitert für besseres Fluchtverhalten)
    collector_sensor_ray_length: float = 400.0 # Sammler sehen sehr weit (maximale Vorwarnzeit)
    collector_sensor_fov: float = 240.0        # Sammler haben seeehr weites Sichtfeld (peripheres Sehen)
    hunter_sensor_ray_length: float = 200.0    # Jäger sind kurzsichtiger
    hunter_sensor_fov: float = 90.0            # Jäger haben Tunnelblick
    radio_range: float = 200.0                 # Reichweite der Kommunikation (Antenne)

    # ── NEAT ──
    collector_pop_size: int = 100     # Populationsgröße Sammler
    hunter_pop_size: int = 20        # Populationsgröße Jäger (Wieder reduziert, damit Sammler atmen können)
    simulation_frames: int = 2500    # Frames pro Evaluierung (laenger = besseres Ueberlebens-Signal)

    # ── Fitness ──
    fitness_battery_collected: float = 100.0  # Punkte pro Batterie (stark erhöht: klares Hauptziel)
    fitness_idle_penalty: float = -0.2        # MASSIVE Strafe pro Frame (Bestraft Kreisdreher extrem!)
    fitness_survival_bonus: float = 0.5       # MASSIVER Bonus pro überlebtem Frame (Belohnt Flucht & Überleben!)
    fitness_hunter_kill: float = 200.0        # Punkte pro gefangenem Sammler
    fitness_eaten_penalty: float = -10000.0   # EXTREME Strafe wenn gefressen (Überleben > alles!)

    # ── Fitness-Gradient (Proximity) ──
    fitness_battery_proximity: float = 0.1   # Bonus/Frame wenn nahe an Batterie (Gradient zum Futter)
    fitness_danger_zone: float = 200.0       # Radius der Gefahrenzone (Reduziert auf 200, um Phantom-Panik zu vermeiden)
    fitness_hunter_danger: float = 0.15      # Strafe/Frame in Jaeger-Naehe (Gradient: naeher = schlimmer)
    fitness_hunter_approach_penalty: float = 15.0  # Extra-Strafe wenn Abstand zum Jaeger kleiner wird

    @property
    def window_width(self) -> int:
        """Fensterbreite in Pixeln."""
        return self.grid_size * self.cell_pixel_size

    @property
    def window_height(self) -> int:
        """Fensterhöhe in Pixeln."""
        return self.grid_size * self.cell_pixel_size

    @property
    def window_size(self) -> tuple[int, int]:
        """Fenstergröße als (Breite, Höhe) Tuple."""
        return (self.window_width, self.window_height)

    def save(self, path: str = "sim_config.json") -> None:
        """Speichert die Konfiguration als JSON-Datei."""
        data = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] Konfiguration gespeichert: {path}")

    @classmethod
    def load(cls, path: str = "sim_config.json") -> 'SimConfig':
        """Lädt die Konfiguration aus einer JSON-Datei."""
        if not os.path.exists(path):
            print(f"[CONFIG] Keine Config-Datei gefunden ({path}), verwende Defaults.")
            return cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Filtere alte Parameter heraus, die nicht mehr in der Klasse existieren
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        config = cls(**filtered_data)
        print(f"[CONFIG] Konfiguration geladen: {path}")
        return config


# ─── Metadaten für das Config-Menü ──────────────────────────────────────────────
# Definiert Kategorien, Felder, Wertebereiche und Schrittweiten für das UI.
CONFIG_FIELDS = [
    {
        "category": "🌍 Welt",
        "fields": [
            ("grid_size", "Gittergröße (N×N)", 10, 60, 1, int),
            ("cell_pixel_size", "Pixel pro Feld", 15, 50, 1, int),
            ("obstacle_count", "Hindernisse", 0, 50, 1, int),
        ]
    },
    {
        "category": "🔋 Batterien",
        "fields": [
            ("battery_count", "Anzahl Batterien", 5, 100, 1, int),
            ("battery_respawn_delay", "Respawn-Delay (Frames)", 0, 600, 10, int),
            ("battery_energy", "Energie pro Batterie", 5.0, 100.0, 5.0, float),
        ]
    },
    {
        "category": "⚡ Energie",
        "fields": [
            ("energy_start", "Start-Energie", 10.0, 500.0, 10.0, float),
            ("energy_drain_per_frame", "Verlust/Frame", 0.01, 1.0, 0.01, float),
            ("energy_death_threshold", "Todesschwelle", 0.0, 50.0, 1.0, float),
        ]
    },
    {
        "category": "🤖 Roboter",
        "fields": [
            ("collector_speed", "Sammler-Speed", 1.0, 10.0, 0.5, float),
            ("hunter_speed", "Jäger-Speed", 1.0, 10.0, 0.5, float),
            ("robot_radius", "Roboter-Radius", 5.0, 25.0, 1.0, float),
            ("sensor_ray_count", "Sensor-Strahlen", 3, 30, 1, int),
            ("collector_sensor_ray_length", "Sichtweite Sammler", 50.0, 400.0, 10.0, float),
            ("collector_sensor_fov", "Sichtfeld Sammler", 30.0, 180.0, 5.0, float),
            ("hunter_sensor_ray_length", "Sichtweite Jäger", 50.0, 400.0, 10.0, float),
            ("hunter_sensor_fov", "Sichtfeld Jäger", 30.0, 180.0, 5.0, float),
            ("radio_range", "Funk-Reichweite", 0.0, 1000.0, 50.0, float),
        ]
    },
    {
        "category": "🧬 NEAT",
        "fields": [
            ("collector_pop_size", "Sammler-Population", 10, 200, 5, int),
            ("hunter_pop_size", "Jäger-Population", 5, 100, 5, int),
            ("simulation_frames", "Frames/Eval.", 500, 5000, 100, int),
        ]
    },
    {
        "category": "🏆 Fitness",
        "fields": [
            ("fitness_battery_collected", "Batterie gesammelt", 1.0, 200.0, 5.0, float),
            ("fitness_idle_penalty", "Idle-Strafe/Frame", -1.0, 0.0, 0.01, float),
            ("fitness_survival_bonus", "Überlebens-Bonus", 0.0, 1.0, 0.01, float),
            ("fitness_hunter_kill", "Jäger-Kill-Punkte", 10.0, 500.0, 10.0, float),
            ("fitness_eaten_penalty", "Gefressen-Strafe", -5000.0, 0.0, 50.0, float),
            ("fitness_battery_proximity", "Batterie-Nähe Bonus", 0.0, 1.0, 0.01, float),
            ("fitness_hunter_danger", "Jäger-Nähe Strafe", 0.0, 2.0, 0.05, float),
            ("fitness_hunter_approach_penalty", "Jaeger-Annaeherung Strafe", 0.0, 20.0, 0.5, float),
            ("fitness_danger_zone", "Gefahrenzone (px)", 50.0, 400.0, 10.0, float),
        ]
    },
]


class ConfigMenu:
    """Pygame-basiertes UI-Menü zur Konfigurationsanpassung.

    Wird vor Simulationsstart angezeigt. Ermöglicht das Ändern aller
    Parameter über Slider, sowie Save/Load als JSON.
    """

    MENU_WIDTH = 1400
    MENU_HEIGHT = 1200
    ROW_HEIGHT = 72
    SLIDER_WIDTH = 400
    SLIDER_HEIGHT = 12
    KNOB_RADIUS = 16
    PADDING = 40
    SCROLL_SPEED = 60

    def __init__(self, config: SimConfig) -> None:
        """Initialisiert das Config-Menü mit der gegebenen Konfiguration."""
        self.config = config
        self.running = True
        self.result: tuple[str, SimConfig] | None = None  # ('train'|'test', config) oder None
        self.scroll_y = 0
        self.max_scroll = 0
        self.active_slider = None  # (field_name, ) wenn ein Slider gezogen wird
        self.font = None
        self.font_small = None
        self.font_title = None
        self.font_category = None

    def _init_fonts(self) -> None:
        """Initialisiert die Schriftarten."""
        self.font = pygame.font.SysFont("Segoe UI", 32)
        self.font_small = pygame.font.SysFont("Segoe UI", 26)
        self.font_title = pygame.font.SysFont("Segoe UI", 56, bold=True)
        self.font_category = pygame.font.SysFont("Segoe UI", 36, bold=True)

    def _calculate_content_height(self) -> int:
        """Berechnet die Gesamthöhe des scrollbaren Inhalts."""
        height = 160  # Titel + Abstand
        for cat in CONFIG_FIELDS:
            height += 90  # Kategorie-Header
            height += len(cat["fields"]) * self.ROW_HEIGHT
            height += 30  # Abstand nach Kategorie
        height += 160  # Buttons am Ende
        return height

    def _draw_slider(self, screen: pygame.Surface, x: int, y: int,
                     value: float, min_val: float, max_val: float,
                     field_name: str) -> pygame.Rect:
        """Zeichnet einen einzelnen Slider und gibt dessen Rect zurück."""
        # Slider-Hintergrund
        slider_rect = pygame.Rect(x, y - self.SLIDER_HEIGHT // 2,
                                  self.SLIDER_WIDTH, self.SLIDER_HEIGHT)
        pygame.draw.rect(screen, COLOR_SLIDER_BG, slider_rect, border_radius=3)

        # Normalisierter Wert
        if max_val - min_val == 0:
            norm = 0.0
        else:
            norm = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

        # Gefüllter Teil
        fill_width = int(norm * self.SLIDER_WIDTH)
        if fill_width > 0:
            fill_rect = pygame.Rect(x, y - self.SLIDER_HEIGHT // 2,
                                    fill_width, self.SLIDER_HEIGHT)
            pygame.draw.rect(screen, COLOR_SLIDER_FILL, fill_rect, border_radius=3)

        # Knob
        knob_x = x + fill_width
        pygame.draw.circle(screen, COLOR_SLIDER_KNOB, (knob_x, y), self.KNOB_RADIUS)
        pygame.draw.circle(screen, COLOR_ACCENT, (knob_x, y), self.KNOB_RADIUS, 2)

        # Klickbarer Bereich (erweitert um Knob-Radius)
        click_rect = pygame.Rect(x - self.KNOB_RADIUS, y - self.KNOB_RADIUS - 4,
                                 self.SLIDER_WIDTH + 2 * self.KNOB_RADIUS,
                                 2 * self.KNOB_RADIUS + 8)
        return click_rect

    def _draw_button(self, screen: pygame.Surface, x: int, y: int,
                     width: int, height: int, text: str,
                     mouse_pos: tuple[int, int]) -> pygame.Rect:
        """Zeichnet einen Button und gibt dessen Rect zurück."""
        rect = pygame.Rect(x, y, width, height)
        is_hover = rect.collidepoint(mouse_pos)
        color = COLOR_BUTTON_HOVER if is_hover else COLOR_BUTTON
        pygame.draw.rect(screen, color, rect, border_radius=12)
        label = self.font.render(text, True, COLOR_BUTTON_TEXT)
        label_rect = label.get_rect(center=rect.center)
        screen.blit(label, label_rect)
        return rect

    def _value_from_slider_pos(self, mouse_x: int, slider_x: int,
                               min_val: float, max_val: float,
                               step: float, val_type: type) -> float | int:
        """Berechnet den Wert aus der Mausposition auf dem Slider."""
        norm = max(0.0, min(1.0, (mouse_x - slider_x) / self.SLIDER_WIDTH))
        raw_value = min_val + norm * (max_val - min_val)

        # Auf Schrittweite runden
        if step > 0:
            raw_value = round(raw_value / step) * step

        # Auf Grenzen klemmen
        raw_value = max(min_val, min(max_val, raw_value))

        if val_type == int:
            return int(round(raw_value))
        return round(raw_value, 4)

    def run(self, screen: pygame.Surface | None = None) -> tuple[str, SimConfig] | None:
        """Startet das Config-Menü und gibt die Konfiguration zurück.

        Returns:
            Tuple ('train'|'test', SimConfig) wenn ein Start-Button gedrückt, None wenn ESC/Fenster geschlossen.
        """
        own_screen = screen is None
        if own_screen:
            pygame.init()
            screen = pygame.display.set_mode((self.MENU_WIDTH, self.MENU_HEIGHT))
            pygame.display.set_caption("Neuro-Ökosystem – Konfiguration")

        self._init_fonts()
        clock = pygame.time.Clock()
        content_height = self._calculate_content_height()
        self.max_scroll = max(0, content_height - self.MENU_HEIGHT + 40)

        slider_rects: dict[str, tuple[pygame.Rect, int, float, float, float, type]] = {}

        while self.running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.result = None

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.result = None
                    elif event.key == pygame.K_RETURN:
                        self.running = False
                        self.result = ('test', self.config)

                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_y -= event.y * self.SCROLL_SPEED
                    self.scroll_y = max(0, min(self.max_scroll, self.scroll_y))

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Prüfe Slider-Klicks
                    for fname, (rect, sx, mn, mx, st, vt) in slider_rects.items():
                        if rect.collidepoint(mouse_pos):
                            self.active_slider = fname
                            new_val = self._value_from_slider_pos(
                                mouse_pos[0], sx, mn, mx, st, vt)
                            setattr(self.config, fname, new_val)
                            break

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.active_slider = None

                elif event.type == pygame.MOUSEMOTION:
                    if self.active_slider and self.active_slider in slider_rects:
                        _, sx, mn, mx, st, vt = slider_rects[self.active_slider]
                        new_val = self._value_from_slider_pos(
                            mouse_pos[0], sx, mn, mx, st, vt)
                        setattr(self.config, self.active_slider, new_val)

            # ── Zeichnen ──────────────────────────────────────────────────
            screen.fill(COLOR_BG)
            slider_rects.clear()

            y_offset = self.PADDING - self.scroll_y

            # Titel
            title_surf = self.font_title.render("⚙ Neuro-Ökosystem Konfiguration",
                                                True, COLOR_ACCENT)
            screen.blit(title_surf, (self.PADDING, y_offset))
            y_offset += 100

            # Fenstergröße-Info
            info_text = (f"Fenster: {self.config.window_width}×{self.config.window_height}px "
                         f"({self.config.grid_size}×{self.config.grid_size} Felder)")
            info_surf = self.font_small.render(info_text, True, COLOR_TEXT_DIM)
            screen.blit(info_surf, (self.PADDING, y_offset))
            y_offset += 60

            # Kategorien und Felder
            for cat in CONFIG_FIELDS:
                # Kategorie-Header
                cat_surf = self.font_category.render(cat["category"], True, COLOR_CATEGORY)
                screen.blit(cat_surf, (self.PADDING, y_offset))
                y_offset += 70

                for field_name, label, min_val, max_val, step, val_type in cat["fields"]:
                    current_val = getattr(self.config, field_name)

                    # Label
                    label_surf = self.font.render(label, True, COLOR_TEXT)
                    screen.blit(label_surf, (self.PADDING + 10, y_offset))

                    # Wert-Anzeige
                    if val_type == int:
                        val_text = str(int(current_val))
                    else:
                        val_text = f"{current_val:.2f}"
                    val_surf = self.font.render(val_text, True, COLOR_ACCENT)
                    val_x = self.MENU_WIDTH - self.PADDING - 120
                    screen.blit(val_surf, (val_x, y_offset))

                    # Slider
                    slider_x = val_x - self.SLIDER_WIDTH - 40
                    slider_y = y_offset + self.ROW_HEIGHT // 2
                    click_rect = self._draw_slider(screen, slider_x, slider_y,
                                                   current_val, min_val, max_val,
                                                   field_name)
                    slider_rects[field_name] = (click_rect, slider_x, min_val,
                                               max_val, step, val_type)

                    y_offset += self.ROW_HEIGHT

                y_offset += 30  # Abstand nach Kategorie

            # ── Buttons ─────────────────────────────────────────────────
            button_y = y_offset + 20
            btn_w = 260
            btn_h = 80
            btn_gap = 20
            total_btn_width = 5 * btn_w + 4 * btn_gap
            btn_start_x = (self.MENU_WIDTH - total_btn_width) // 2

            btn_start_train = self._draw_button(screen, btn_start_x, button_y,
                                          btn_w, btn_h, "🚀 Start Headless", mouse_pos)
            btn_start_test = self._draw_button(screen, btn_start_x + btn_w + btn_gap, button_y,
                                          btn_w, btn_h, "👁 Start Visual Test", mouse_pos)
            btn_save = self._draw_button(screen, btn_start_x + 2 * (btn_w + btn_gap),
                                         button_y, btn_w, btn_h, "💾 Speichern",
                                         mouse_pos)
            btn_load = self._draw_button(screen, btn_start_x + 3 * (btn_w + btn_gap),
                                         button_y, btn_w, btn_h, "📂 Laden",
                                         mouse_pos)
            btn_defaults = self._draw_button(screen,
                                             btn_start_x + 4 * (btn_w + btn_gap),
                                             button_y, btn_w, btn_h, "↺ Defaults",
                                             mouse_pos)

            # Button-Klicks prüfen (im Event-Loop oben wurde nur Slider gehandelt)
            if pygame.mouse.get_pressed()[0]:
                if btn_start_train.collidepoint(mouse_pos) and self.active_slider is None:
                    self.running = False
                    self.result = ('train', self.config)
                elif btn_start_test.collidepoint(mouse_pos) and self.active_slider is None:
                    self.running = False
                    self.result = ('test', self.config)
                elif btn_save.collidepoint(mouse_pos) and self.active_slider is None:
                    self.config.save()
                elif btn_load.collidepoint(mouse_pos) and self.active_slider is None:
                    self.config = SimConfig.load()
                elif btn_defaults.collidepoint(mouse_pos) and self.active_slider is None:
                    self.config = SimConfig()

            # ── Scrollbar ────────────────────────────────────────────────
            if self.max_scroll > 0:
                sb_x = self.MENU_WIDTH - 16
                sb_height = self.MENU_HEIGHT
                thumb_height = max(60, int(sb_height * sb_height / content_height))
                thumb_y = int(self.scroll_y / self.max_scroll * (sb_height - thumb_height))
                pygame.draw.rect(screen, COLOR_SCROLLBAR,
                                 (sb_x, 0, 12, sb_height), border_radius=6)
                pygame.draw.rect(screen, COLOR_SCROLLBAR_THUMB,
                                 (sb_x, thumb_y, 12, thumb_height), border_radius=6)

            # ── Hinweis-Leiste unten ──────────────────────────────────────
            hint_surf = self.font_small.render(
                "ENTER = Start  |  ESC = Beenden  |  Mausrad = Scrollen",
                True, COLOR_TEXT_DIM)
            hint_rect = hint_surf.get_rect(
                center=(self.MENU_WIDTH // 2, self.MENU_HEIGHT - 30))
            # Halbtransparenter Hintergrund für Hinweis
            hint_bg = pygame.Surface((self.MENU_WIDTH, 60), pygame.SRCALPHA)
            hint_bg.fill((18, 18, 24, 220))
            screen.blit(hint_bg, (0, self.MENU_HEIGHT - 60))
            screen.blit(hint_surf, hint_rect)

            pygame.display.flip()
            clock.tick(60)

            clock.tick(60)

        return self.result
