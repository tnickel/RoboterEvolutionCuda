"""
hall_of_fame.py – Hall of Fame / Ranking-System für die besten Roboter-Gehirne.

Enthält:
- HallOfFameEntry: Einzelner Eintrag mit Name, Fitness, Genom-Daten
- HallOfFame: Verwaltet die Top-20 Liste, speichert/lädt als Pickle
- HallOfFameMenu: Pygame-UI zur Anzeige und Auswahl von Einträgen
"""

import os
import pickle
import time
import random
import pygame
from dataclasses import dataclass, field


# ─── Farb-Konstanten ────────────────────────────────────────────────────────────
COLOR_BG = (18, 18, 24)
COLOR_PANEL = (28, 28, 38)
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (120, 120, 140)
COLOR_ACCENT = (0, 200, 150)
COLOR_GOLD = (255, 215, 50)
COLOR_SILVER = (192, 192, 210)
COLOR_BRONZE = (205, 140, 50)
COLOR_ROW_EVEN = (25, 25, 35)
COLOR_ROW_ODD = (30, 30, 42)
COLOR_ROW_SELECTED = (0, 80, 60)
COLOR_ROW_HOVER = (40, 40, 55)
COLOR_BUTTON = (0, 180, 130)
COLOR_BUTTON_HOVER = (0, 220, 160)
COLOR_BUTTON_TEXT = (18, 18, 24)
COLOR_SCROLLBAR = (50, 50, 65)
COLOR_SCROLLBAR_THUMB = (100, 100, 120)
COLOR_CHECKBOX = (0, 200, 150)
COLOR_CHECKBOX_BG = (50, 50, 65)

# ─── Robot-Namen ─────────────────────────────────────────────────────────────────
# Generiert aus Präfix + Nummer für einprägsame Robot-Namen
NAME_PREFIXES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Nova", "Apex", "Bolt", "Cipher", "Drift", "Echo", "Flux", "Ghost",
    "Helix", "Ion", "Jade", "Kron", "Lynx", "Mach", "Neon", "Onyx",
    "Pulse", "Quasar", "Razor", "Spark", "Titan", "Ultra", "Vex", "Warp",
    "Xenon", "Yeti", "Zenith", "Blaze", "Cryo", "Dynamo", "Ember", "Fang",
]

SAVE_FILE = "hall_of_fame.pkl"
MAX_ENTRIES = 20


@dataclass
class HallOfFameEntry:
    """Ein Eintrag in der Hall of Fame.

    Attributes:
        name: Einzigartiger Robot-Name
        fitness: Erreichte Fitness
        generation: Generation in der das Genom entstand
        batteries_collected: Anzahl gesammelter Batterien
        genome_data: Serialisiertes NEAT-Genom (pickle bytes)
        timestamp: Zeitpunkt der Aufnahme
        config_hash: Hash der SimConfig (für Kompatibilitätsprüfung)
    """
    name: str
    fitness: float
    generation: int
    batteries_collected: int
    genome_data: bytes  # Pickle-serialisiertes Genom
    timestamp: float = 0.0
    config_hash: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class HallOfFame:
    """Verwaltet die Top-20 Hall of Fame der besten Roboter-Gehirne.

    Speichert die besten Genome als Pickle-Datei und ermöglicht
    das Laden und Injizieren in neue Populationen.
    """

    def __init__(self, save_path: str = SAVE_FILE) -> None:
        self.save_path = save_path
        self.entries: list[HallOfFameEntry] = []
        self._used_names: set[str] = set()
        self.load()

    def _generate_name(self) -> str:
        """Generiert einen einzigartigen Robot-Namen."""
        for _ in range(500):
            prefix = random.choice(NAME_PREFIXES)
            number = random.randint(1, 999)
            name = f"{prefix}-{number:03d}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name
        # Fallback
        name = f"Bot-{random.randint(1000, 9999)}"
        self._used_names.add(name)
        return name

    def try_add(self, genome, fitness: float, generation: int,
                batteries: int, config_hash: str = "") -> HallOfFameEntry | None:
        """Versucht ein Genom in die Hall of Fame aufzunehmen.

        Wird nur aufgenommen wenn die Fitness besser als der schlechteste
        Eintrag ist (oder weniger als MAX_ENTRIES vorhanden sind).

        Args:
            genome: NEAT-Genom Objekt
            fitness: Erreichte Fitness
            generation: Generation des Genoms
            batteries: Gesammelte Batterien
            config_hash: Hash der Konfiguration

        Returns:
            HallOfFameEntry wenn aufgenommen, sonst None.
        """
        # Prüfen ob gut genug
        if len(self.entries) >= MAX_ENTRIES:
            worst_fitness = min(e.fitness for e in self.entries)
            if fitness <= worst_fitness:
                return None

        # Genom serialisieren
        genome_data = pickle.dumps(genome)

        # Neuen Eintrag erstellen
        entry = HallOfFameEntry(
            name=self._generate_name(),
            fitness=fitness,
            generation=generation,
            batteries_collected=batteries,
            genome_data=genome_data,
            config_hash=config_hash,
        )

        self.entries.append(entry)
        # Sortieren (beste zuerst) und auf MAX_ENTRIES kürzen
        self.entries.sort(key=lambda e: e.fitness, reverse=True)
        if len(self.entries) > MAX_ENTRIES:
            removed = self.entries.pop()
            self._used_names.discard(removed.name)

        self.save()
        print(f"[HALL] Neuer Eintrag: {entry.name} "
              f"(Fitness={fitness:.1f}, Gen={generation})")
        return entry

    def get_genome(self, entry: HallOfFameEntry):
        """Deserialisiert das Genom aus einem Eintrag.

        Returns:
            NEAT-Genom Objekt.
        """
        return pickle.loads(entry.genome_data)

    def save(self) -> None:
        """Speichert die Hall of Fame als Pickle-Datei."""
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.entries, f)
        except Exception as e:
            print(f"[HALL] Fehler beim Speichern: {e}")

    def load(self) -> None:
        """Lädt die Hall of Fame aus der Pickle-Datei."""
        if not os.path.exists(self.save_path):
            self.entries = []
            return
        try:
            with open(self.save_path, 'rb') as f:
                self.entries = pickle.load(f)
            self._used_names = {e.name for e in self.entries}
            print(f"[HALL] {len(self.entries)} Eintraege geladen")
        except Exception as e:
            print(f"[HALL] Fehler beim Laden: {e}")
            self.entries = []

    def clear(self) -> None:
        """Löscht alle Einträge."""
        self.entries.clear()
        self._used_names.clear()
        self.save()

    def remove_entry(self, index: int) -> None:
        """Löscht einen Eintrag an einem bestimmten Index."""
        if 0 <= index < len(self.entries):
            removed = self.entries.pop(index)
            self._used_names.discard(removed.name)
            self.save()
            print(f"[HALL] Eintrag geloescht: {removed.name}")


class HallOfFameMenu:
    """Pygame-UI Menü zur Anzeige der Hall of Fame und Auswahl von Genomen.

    Zeigt die Top-20 Ranking-Liste mit Checkboxen zur Auswahl.
    Ausgewählte Genome werden in die nächste Population injiziert.
    """

    MENU_WIDTH = 1700
    MENU_HEIGHT = 1200
    ROW_HEIGHT = 80
    HEADER_HEIGHT = 200
    FOOTER_HEIGHT = 130

    def __init__(self, hall: HallOfFame) -> None:
        self.hall = hall
        # Standardmaessig die besten 5 Eintraege auswaehlen (oder weniger, wenn die Liste kleiner ist)
        num_auto_select = min(5, len(self.hall.entries))
        self.selected: set[int] = set(range(num_auto_select))
        self.running = True
        self.result = None  # 'start', 'back', None
        self.scroll_y = 0
        self.hover_row = -1

    def run(self, screen: pygame.Surface | None = None) -> list[HallOfFameEntry]:
        """Zeigt das Hall of Fame Menü und gibt ausgewählte Einträge zurück.

        Returns:
            Liste der ausgewählten HallOfFameEntry-Objekte.
        """
        own_screen = screen is None
        if own_screen:
            screen = pygame.display.set_mode((self.MENU_WIDTH, self.MENU_HEIGHT))

        pygame.display.set_caption("Neuro-Oekosystem - Hall of Fame")

        font_title = pygame.font.SysFont("Segoe UI", 56, bold=True)
        font = pygame.font.SysFont("Segoe UI", 34)
        font_header = pygame.font.SysFont("Segoe UI", 32, bold=True)
        font_small = pygame.font.SysFont("Segoe UI", 26)
        font_bold = pygame.font.SysFont("Segoe UI", 34, bold=True)
        clock = pygame.time.Clock()

        max_visible = (self.MENU_HEIGHT - self.HEADER_HEIGHT - self.FOOTER_HEIGHT) // self.ROW_HEIGHT
        max_scroll = max(0, len(self.hall.entries) - max_visible)

        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            self.hover_row = -1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.result = None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.result = 'back'
                    elif event.key == pygame.K_RETURN:
                        self.running = False
                        self.result = 'start'
                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_y = max(0, min(max_scroll,
                                               self.scroll_y - event.y))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Checkbox-Klick prüfen
                    row_area_top = self.HEADER_HEIGHT
                    row_area_bottom = self.MENU_HEIGHT - self.FOOTER_HEIGHT
                    if row_area_top <= mouse_pos[1] < row_area_bottom:
                        row_idx = (mouse_pos[1] - row_area_top) // self.ROW_HEIGHT
                        actual_idx = row_idx + self.scroll_y
                        if 0 <= actual_idx < len(self.hall.entries):
                            if actual_idx in self.selected:
                                self.selected.discard(actual_idx)
                            else:
                                self.selected.add(actual_idx)

            # ── Zeichnen ──────────────────────────────────────────────
            screen.fill(COLOR_BG)

            # Titel
            title = font_title.render("Hall of Fame - Top 20", True, COLOR_GOLD)
            screen.blit(title, (40, 20))

            subtitle = font_small.render(
                f"{len(self.hall.entries)} Eintraege | "
                f"{len(self.selected)} ausgewaehlt | "
                f"Klicken = Auswaehlen | ENTER = Mit Auswahl starten",
                True, COLOR_TEXT_DIM)
            screen.blit(subtitle, (40, 90))

            # Tabellen-Header (mit Hintergrund-Panel)
            header_y = self.HEADER_HEIGHT - 60
            header_panel = pygame.Rect(20, header_y - 5, self.MENU_WIDTH - 40, 60)
            pygame.draw.rect(screen, (35, 35, 50), header_panel, border_radius=8)
            pygame.draw.rect(screen, COLOR_ACCENT, header_panel, 2, border_radius=8)

            # Spalten-Positionen und Labels
            col_defs = [
                (36,  ""),            # Checkbox
                (100, "Rang"),
                (220, "Robot-Name"),
                (580, "Fitness"),
                (840, "Batterien"),
                (1100, "Generation"),
                (1360, "Datum"),
            ]
            for col_x, label in col_defs:
                if label:
                    screen.blit(font_header.render(label, True, COLOR_ACCENT),
                                (col_x, header_y + 10))

            # Tabellenzeilen
            row_area_top = self.HEADER_HEIGHT
            visible_rows = (self.MENU_HEIGHT - self.HEADER_HEIGHT - self.FOOTER_HEIGHT) // self.ROW_HEIGHT

            for vis_idx in range(visible_rows):
                actual_idx = vis_idx + self.scroll_y
                if actual_idx >= len(self.hall.entries):
                    break

                entry = self.hall.entries[actual_idx]
                row_y = row_area_top + vis_idx * self.ROW_HEIGHT

                # Hover-Erkennung
                row_rect = pygame.Rect(0, row_y, self.MENU_WIDTH, self.ROW_HEIGHT)
                is_hover = row_rect.collidepoint(mouse_pos)
                if is_hover:
                    self.hover_row = actual_idx

                # Hintergrundfarbe
                if actual_idx in self.selected:
                    bg_color = COLOR_ROW_SELECTED
                elif is_hover:
                    bg_color = COLOR_ROW_HOVER
                elif vis_idx % 2 == 0:
                    bg_color = COLOR_ROW_EVEN
                else:
                    bg_color = COLOR_ROW_ODD

                pygame.draw.rect(screen, bg_color, row_rect)

                # Checkbox
                cb_rect = pygame.Rect(40, row_y + 20, 40, 40)
                pygame.draw.rect(screen, COLOR_CHECKBOX_BG, cb_rect, border_radius=8)
                if actual_idx in self.selected:
                    pygame.draw.rect(screen, COLOR_CHECKBOX, cb_rect, border_radius=6)
                    # Häkchen
                    pygame.draw.line(screen, COLOR_BG,
                                    (cb_rect.x + 6, cb_rect.y + 16),
                                    (cb_rect.x + 14, cb_rect.y + 24), 4)
                    pygame.draw.line(screen, COLOR_BG,
                                    (cb_rect.x + 14, cb_rect.y + 24),
                                    (cb_rect.x + 26, cb_rect.y + 8), 4)
                else:
                    pygame.draw.rect(screen, COLOR_TEXT_DIM, cb_rect, 2, border_radius=6)

                # Rang-Farbe
                rank = actual_idx + 1
                if rank == 1:
                    rank_color = COLOR_GOLD
                elif rank == 2:
                    rank_color = COLOR_SILVER
                elif rank == 3:
                    rank_color = COLOR_BRONZE
                else:
                    rank_color = COLOR_TEXT

                text_y = row_y + 20

                # Rang
                screen.blit(font_bold.render(f"{rank}.", True, rank_color), (110, text_y))

                # Name
                screen.blit(font.render(entry.name, True, rank_color), (220, text_y))

                # Fitness
                fit_color = COLOR_ACCENT if entry.fitness > 0 else COLOR_TEXT
                screen.blit(font.render(f"{entry.fitness:.1f}", True, fit_color), (580, text_y))

                # Batterien
                screen.blit(font.render(str(entry.batteries_collected), True, COLOR_TEXT), (840, text_y))

                # Generation
                screen.blit(font.render(str(entry.generation), True, COLOR_TEXT), (1100, text_y))

                # Datum
                date_str = time.strftime("%d.%m %H:%M",
                                         time.localtime(entry.timestamp))
                screen.blit(font_small.render(date_str, True, COLOR_TEXT_DIM), (1360, text_y + 4))

            # ── Footer / Buttons ──────────────────────────────────────
            footer_y = self.MENU_HEIGHT - self.FOOTER_HEIGHT
            pygame.draw.line(screen, COLOR_ACCENT,
                             (30, footer_y), (self.MENU_WIDTH - 30, footer_y), 2)

            btn_w, btn_h = 320, 76
            btn_gap = 40
            total_w = 3 * btn_w + 2 * btn_gap
            start_x = (self.MENU_WIDTH - total_w) // 2
            btn_y = footer_y + 25

            # Start-Button
            btn_start_rect = pygame.Rect(start_x, btn_y, btn_w, btn_h)
            btn_start_hover = btn_start_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen,
                             COLOR_BUTTON_HOVER if btn_start_hover else COLOR_BUTTON,
                             btn_start_rect, border_radius=12)
            sel_text = f"Start ({len(self.selected)} geladen)" if self.selected else "Start (ohne)"
            lbl = font.render(sel_text, True, COLOR_BUTTON_TEXT)
            screen.blit(lbl, lbl.get_rect(center=btn_start_rect.center))

            # Zurück-Button
            btn_back_rect = pygame.Rect(start_x + btn_w + btn_gap, btn_y, btn_w, btn_h)
            btn_back_hover = btn_back_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen,
                             COLOR_BUTTON_HOVER if btn_back_hover else (80, 80, 100),
                             btn_back_rect, border_radius=12)
            lbl2 = font.render("Zurueck", True, COLOR_TEXT)
            screen.blit(lbl2, lbl2.get_rect(center=btn_back_rect.center))

            # Alles Löschen-Button
            btn_clear_rect = pygame.Rect(start_x + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h)
            btn_clear_hover = btn_clear_rect.collidepoint(mouse_pos)
            color_clear = (220, 80, 80) if btn_clear_hover else (180, 50, 50)
            pygame.draw.rect(screen, color_clear, btn_clear_rect, border_radius=12)
            lbl3 = font.render("Alles loeschen", True, COLOR_TEXT)
            screen.blit(lbl3, lbl3.get_rect(center=btn_clear_rect.center))

            # Button-Klicks
            if pygame.mouse.get_pressed()[0]:
                if btn_start_rect.collidepoint(mouse_pos):
                    self.running = False
                    self.result = 'start'
                elif btn_back_rect.collidepoint(mouse_pos):
                    self.running = False
                    self.result = 'back'
                elif btn_clear_rect.collidepoint(mouse_pos):
                    self.hall.clear()
                    self.selected.clear()
                    self.scroll_y = 0
                    pygame.time.delay(200)  # Kurzer Delay gegen Endlos-Auslösung

            # Leere Liste Hinweis
            if not self.hall.entries:
                empty_text = font.render(
                    "Noch keine Eintraege - trainiere zuerst!",
                    True, COLOR_TEXT_DIM)
                screen.blit(empty_text, (self.MENU_WIDTH // 2 - 300, 500))

            pygame.display.flip()
            clock.tick(60)

        # Ausgewählte Einträge zurückgeben
        selected_entries = [self.hall.entries[i] for i in sorted(self.selected)
                            if i < len(self.hall.entries)]
        return selected_entries if self.result == 'start' else []
