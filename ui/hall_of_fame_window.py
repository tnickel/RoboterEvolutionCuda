"""
hall_of_fame_window.py – Eigenstaendiges Hall-of-Fame-Fenster.

Zeigt in einem separaten Tkinter-Fenster die Top-20 der besten
Roboter-Gehirne als Live-Rangliste. Aktualisiert sich automatisch
ueber eine Thread-sichere Queue.
"""

import threading
import queue
import time
import tkinter as tk
from tkinter import ttk


class HallOfFameWindow:
    """Separates Tkinter-Fenster fuer die Hall of Fame Rangliste.

    Laeuft in einem eigenen Thread und aktualisiert sich
    automatisch wenn neue Eintraege hinzukommen.
    """

    def __init__(self, hall, offset_x: int = 0) -> None:
        self.hall = hall
        self.update_queue: queue.Queue = queue.Queue()
        self.running = False
        self.thread = None
        self.root = None
        self.offset_x = offset_x
        self._last_entry_count = 0

    def start(self) -> None:
        """Startet das Hall-of-Fame-Fenster in einem separaten Thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Schliesst das Fenster."""
        self.running = False

    def notify_update(self) -> None:
        """Benachrichtigt das Fenster, dass sich die Hall of Fame geaendert hat."""
        self.update_queue.put(True)

    def _run_window(self) -> None:
        """Hauptschleife des Tkinter-Fensters."""
        # Eigene Tk-Instanz fuer diesen Thread
        self.root = tk.Tk()
        self.root.withdraw()  # Verstecke die leere Tk-Root
        self.window = tk.Toplevel(self.root)
        self.window.title("Hall of Fame - Top 20")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        # Fenster rechts vom Simulationsfenster, obere Haelfte
        win_w = 520
        win_h = int(sh * 0.65) - 40
        pos_x = sw - win_w - 10
        pos_y = 0

        self.window.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.window.configure(bg="#121218")
        self.window.attributes("-topmost", True)

        # ── Titel-Bereich ──
        title_frame = tk.Frame(self.window, bg="#121218")
        title_frame.pack(fill=tk.X, padx=15, pady=(12, 0))

        tk.Label(title_frame, text="🏆 Hall of Fame",
                 font=("Segoe UI", 28, "bold"),
                 fg="#FFD732", bg="#121218").pack(side=tk.LEFT)

        self.lbl_count = tk.Label(title_frame, text="(0/20)",
                                   font=("Segoe UI", 16),
                                   fg="#888898", bg="#121218")
        self.lbl_count.pack(side=tk.RIGHT, padx=10)

        # Trennlinie
        tk.Frame(self.window, bg="#FFD732", height=2).pack(fill=tk.X, padx=15, pady=(8, 0))

        # ── Tabellen-Header ──
        header_frame = tk.Frame(self.window, bg="#1a1a24")
        header_frame.pack(fill=tk.X, padx=15, pady=(8, 0))

        headers = [
            ("#", 40),
            ("Name", 140),
            ("Fitness", 90),
            ("Batterien", 80),
            ("Gen", 60),
            ("Zeit", 90),
        ]
        for text, width in headers:
            tk.Label(header_frame, text=text, font=("Segoe UI", 12, "bold"),
                     fg="#00C896", bg="#1a1a24", width=width // 10,
                     anchor=tk.W).pack(side=tk.LEFT, padx=4)

        # ── Scrollbare Tabelle ──
        container = tk.Frame(self.window, bg="#121218")
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(4, 10))

        self.canvas_widget = tk.Canvas(container, bg="#121218",
                                        highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL,
                                  command=self.canvas_widget.yview)
        self.table_frame = tk.Frame(self.canvas_widget, bg="#121218")

        self.table_frame.bind("<Configure>",
                               lambda e: self.canvas_widget.configure(
                                   scrollregion=self.canvas_widget.bbox("all")))

        self.canvas_widget.create_window((0, 0), window=self.table_frame,
                                          anchor=tk.NW)
        self.canvas_widget.configure(yscrollcommand=scrollbar.set)

        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mausrad-Scrolling
        self.canvas_widget.bind_all("<MouseWheel>",
                                     lambda e: self.canvas_widget.yview_scroll(
                                         -1 * (e.delta // 120), "units"))

        # ── Status-Leiste ──
        self.lbl_status = tk.Label(self.window,
                                    text="Warte auf erste Eintraege...",
                                    font=("Segoe UI", 11),
                                    fg="#888898", bg="#121218")
        self.lbl_status.pack(fill=tk.X, padx=15, pady=(0, 8))

        # Initiale Darstellung
        self._refresh_table()

        # Polling starten
        self._poll_updates()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        """Wird aufgerufen wenn das Fenster geschlossen wird."""
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def _poll_updates(self) -> None:
        """Prueft regelmaessig ob Updates vorliegen."""
        if not self.running:
            return

        # Prüfe Queue und Entry-Count
        needs_refresh = False
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()
                needs_refresh = True
            except queue.Empty:
                break

        # Auch bei neuen Einträgen ohne explizites Signal aktualisieren
        current_count = len(self.hall.entries)
        if current_count != self._last_entry_count:
            needs_refresh = True
            self._last_entry_count = current_count

        if needs_refresh:
            self._refresh_table()

        if self.root:
            self.root.after(1000, self._poll_updates)

    def _refresh_table(self) -> None:
        """Aktualisiert die gesamte Tabelle mit den aktuellen Hall-of-Fame-Daten."""
        # Alte Zeilen entfernen
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        entries = self.hall.entries
        self.lbl_count.config(text=f"({len(entries)}/20)")

        if not entries:
            tk.Label(self.table_frame,
                     text="Noch keine Eintraege - trainiere zuerst!",
                     font=("Segoe UI", 14),
                     fg="#888898", bg="#121218").pack(pady=40)
            self.lbl_status.config(text="Warte auf erste Eintraege...")
            return

        # Farben für Ränge
        rank_colors = {
            1: "#FFD732",   # Gold
            2: "#C0C0D2",   # Silber
            3: "#CD8C32",   # Bronze
        }

        for i, entry in enumerate(entries):
            rank = i + 1
            rank_color = rank_colors.get(rank, "#B4B4BE")

            # Zeilen-Hintergrund (alternierend)
            row_bg = "#1a1a24" if i % 2 == 0 else "#222232"

            row_frame = tk.Frame(self.table_frame, bg=row_bg)
            row_frame.pack(fill=tk.X, padx=2, pady=1)

            # Rang
            rang_text = f"  {rank}."
            if rank == 1:
                rang_text = " 🥇"
            elif rank == 2:
                rang_text = " 🥈"
            elif rank == 3:
                rang_text = " 🥉"

            tk.Label(row_frame, text=rang_text,
                     font=("Segoe UI", 13, "bold"),
                     fg=rank_color, bg=row_bg,
                     width=4, anchor=tk.W).pack(side=tk.LEFT, padx=(6, 2))

            # Name
            tk.Label(row_frame, text=entry.name,
                     font=("Segoe UI", 13, "bold"),
                     fg=rank_color, bg=row_bg,
                     width=14, anchor=tk.W).pack(side=tk.LEFT, padx=2)

            # Fitness
            fit_color = "#00C896" if entry.fitness > 0 else "#E65050"
            tk.Label(row_frame, text=f"{entry.fitness:.0f}",
                     font=("Segoe UI", 13),
                     fg=fit_color, bg=row_bg,
                     width=8, anchor=tk.W).pack(side=tk.LEFT, padx=2)

            # Batterien
            tk.Label(row_frame, text=f"{entry.batteries_collected} 🔋",
                     font=("Segoe UI", 13),
                     fg="#E6DC50", bg=row_bg,
                     width=8, anchor=tk.W).pack(side=tk.LEFT, padx=2)

            # Generation
            tk.Label(row_frame, text=f"G{entry.generation}",
                     font=("Segoe UI", 12),
                     fg="#888898", bg=row_bg,
                     width=6, anchor=tk.W).pack(side=tk.LEFT, padx=2)

            # Zeit
            date_str = time.strftime("%H:%M:%S",
                                     time.localtime(entry.timestamp))
            tk.Label(row_frame, text=date_str,
                     font=("Segoe UI", 11),
                     fg="#666678", bg=row_bg,
                     width=8, anchor=tk.W).pack(side=tk.LEFT, padx=2)

        # Status aktualisieren
        best = entries[0]
        self.lbl_status.config(
            text=f"Bester: {best.name} mit {best.fitness:.0f} Fitness | "
                 f"{len(entries)} Eintraege gesamt")
