"""
commentary.py – Live-Kommentar-Fenster fuer das Neuro-Oekosystem.

Zeigt in einem separaten Tkinter-Fenster in einfacher Sprache,
was gerade in der Simulation passiert. Laeuft in einem eigenen Thread.
"""

import threading
import queue
import time
import tkinter as tk
from tkinter import scrolledtext
import random


# --- Kommentar-Vorlagen -------------------------------------------------------

GENERATION_START = [
    "Generation {gen} beginnt! {collectors} Sammler und {hunters} Jaeger betreten die Arena.",
    "Runde {gen}! Alle Roboter starten mit frischer Energie. Wer wird diesmal der Beste?",
    "Neue Generation #{gen} - die Evolution geht weiter!",
    "Generation {gen}: {collectors} gruene Sammler gegen {hunters} rote Jaeger. Los geht's!",
]

GENERATION_END = [
    "Generation {gen} ist vorbei! Bester Sammler: {best_c:.0f} Punkte, bester Jaeger: {best_h:.0f} Punkte.",
    "Runde {gen} beendet. {alive} Sammler haben ueberlebt, {kills} wurden gefressen.",
    "Ende Generation {gen}. Die Natur selektiert die Besten fuer die naechste Runde.",
]

KILL_COMMENTS = [
    "Ein Jaeger hat einen Sammler erwischt! {kills} Kills insgesamt.",
    "Sammler gefressen! Die Jaeger werden immer besser im Jagen.",
    "Noch ein Sammler weniger... {alive} kaempfen noch ums Ueberleben.",
    "Die Jaeger schlagen zu! Nur noch {alive} Sammler uebrig.",
    "Ein roter Jaeger hat zugeschnappt - {kills} erfolgreiche Jagden bisher.",
]

BATTERY_COMMENTS = [
    "Ein Sammler hat eine Batterie gefunden! Energie aufgeladen.",
    "Batterie eingesammelt! Das gibt Bonus-Punkte und frische Energie.",
    "{total} Batterien wurden in dieser Runde bisher gesammelt.",
]

FITNESS_IMPROVING = [
    "Die Sammler werden schlauer! Beste Fitness ist auf {fitness:.0f} gestiegen.",
    "Fortschritt! Die Sammler lernen, Batterien gezielter zu finden.",
    "Die Evolution wirkt - die Roboter bewegen sich immer zielgerichteter.",
]

FITNESS_DECLINING = [
    "Die Sammler haben es schwer - die Jaeger werden zu gut.",
    "Rueckschritt bei den Sammlern. Der Selektionsdruck durch die Jaeger ist hoch.",
]

HUNTER_IMPROVING = [
    "Die Jaeger werden zu echten Raubtieren! Beste Fitness: {fitness:.0f}.",
    "Jagdinstinkt verbessert! Die roten Jaeger lernen, Beute zu finden.",
    "Die Raeuber-Evolution laeuft auf Hochtouren.",
]

MANY_KILLS = [
    "Massaker! {kills} Sammler wurden in dieser Runde gefressen. Die Jaeger dominieren!",
    "Die Jaeger sind unerbittlich - {kills} erfolgreiche Jagden!",
]

FEW_KILLS = [
    "Die Sammler haben gut ueberlebt! Nur {kills} wurden erwischt.",
    "Die Sammler lernen zu fliehen! Wenige Verluste in dieser Runde.",
]

HALL_OF_FAME = [
    "NEUER REKORD! '{name}' schafft es in die Hall of Fame mit {fitness:.0f} Punkten!",
    "Hall of Fame Update: '{name}' ist jetzt unter den besten 20 Robotern aller Zeiten!",
]

ECOSYSTEM_BALANCE = [
    "Das Oekosystem balanciert sich: Sammler werden schlauer, Jaeger werden schneller.",
    "Wettruestung! Beide Seiten verbessern sich gegenseitig durch die Co-Evolution.",
    "Faszinierend: Die Sammler entwickeln Ausweichmanoever, die Jaeger kontern mit besserer Jagdtaktik.",
]

MILESTONE_COMMENTS = {
    5: "5 Generationen geschafft! Die ersten Verhaltens-Muster werden sichtbar.",
    10: "10 Generationen! Die Roboter sollten jetzt nicht mehr planlos herumfahren.",
    25: "25 Generationen - die Evolution nimmt Fahrt auf!",
    50: "Halbzeit bei 50 Generationen! Schau dir an, wie clever sie geworden sind.",
    100: "100 Generationen! Die Roboter sind jetzt richtig erfahren.",
}


class CommentaryEvent:
    """Ein Kommentar-Ereignis mit Typ und Daten."""

    def __init__(self, event_type: str, data: dict = None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = time.time()


class CommentaryWindow:
    """Separates Tkinter-Fenster fuer Live-Kommentare.

    Laeuft in einem eigenen Thread und empfaengt Events
    ueber eine Thread-sichere Queue.
    """

    def __init__(self, offset_x: int = 0) -> None:
        self.event_queue: queue.Queue = queue.Queue()
        self.running = False
        self.thread = None
        self.root = None
        self._last_comment_time = 0
        self._min_interval = 1.5  # Mindestabstand zwischen Kommentaren (Sekunden)
        self._last_collector_fitness = -9999
        self._last_hunter_fitness = -9999
        self.offset_x = offset_x

    def start(self) -> None:
        """Startet das Kommentar-Fenster in einem separaten Thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Schliesst das Kommentar-Fenster."""
        self.running = False

    def post_event(self, event_type: str, **data) -> None:
        """Sendet ein Event an das Kommentar-Fenster (thread-sicher).

        Args:
            event_type: Art des Events (z.B. 'gen_start', 'kill', etc.)
            **data: Zusaetzliche Daten fuer den Kommentar.
        """
        self.event_queue.put(CommentaryEvent(event_type, data))

    def _run_window(self) -> None:
        """Hauptschleife des Tkinter-Fensters (laeuft im eigenen Thread)."""
        self.root = tk.Tk()
        self.root.title("Neuro-Oekosystem - Live Kommentar")
        
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        
        # Dynamisch an verbleibenden Platz anpassen (untere 35% der Höhe)
        win_w = max(600, sw - self.offset_x)
        win_h = int(sh * 0.35) - 40  # Minus Taskleisten-Puffer
        pos_y = int(sh * 0.65)
        
        self.root.geometry(f"{win_w}x{win_h}+{self.offset_x}+{pos_y}")
        self.root.configure(bg="#121218")
        self.root.attributes("-topmost", True)

        # Titel-Label
        title = tk.Label(
            self.root, text="Live Kommentar",
            font=("Segoe UI", -42, "bold"),
            fg="#00C896", bg="#121218")
        title.pack(pady=(10, 5))

        subtitle = tk.Label(
            self.root, text="Was passiert gerade in der Simulation?",
            font=("Segoe UI", -24),
            fg="#aaaaaa", bg="#121218")
        subtitle.pack(pady=(0, 10))

        # Textfeld
        self.text_widget = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Segoe UI", -30),
            bg="#1a1a24",
            fg="#FFFFFF",
            insertbackground="#00C896",
            selectbackground="#00C896",
            relief=tk.FLAT,
            padx=16,
            pady=12,
            spacing3=4,  # Deutlich reduzierte Abstände zwischen den Einträgen
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.text_widget.config(state=tk.DISABLED)

        # Tags fuer Farben
        self.text_widget.tag_config("normal", foreground="#FFFFFF", font=("Segoe UI", -30))
        self.text_widget.tag_config("highlight", foreground="#00C896", font=("Segoe UI", -30))
        self.text_widget.tag_config("warning", foreground="#E6C832", font=("Segoe UI", -30))
        self.text_widget.tag_config("danger", foreground="#E63232", font=("Segoe UI", -30))
        self.text_widget.tag_config("info", foreground="#FFFFFF", font=("Segoe UI", -30))
        self.text_widget.tag_config("milestone", foreground="#FFD700",
                                     font=("Segoe UI", -32, "bold"))
        self.text_widget.tag_config("separator", foreground="#888898",
                                     font=("Segoe UI", -22))
        self.text_widget.tag_config("time", foreground="#aaaaaa",
                                     font=("Segoe UI", -24))

        # Willkommens-Nachricht
        self._add_comment(
            "Willkommen zum Neuro-Oekosystem!\n"
            "Hier siehst du in einfacher Sprache, was die KI-Roboter gerade machen.\n"
            "Gruene Sammler suchen Batterien, rote Jaeger jagen die Sammler.\n"
            "Beide lernen durch Evolution - Generation fuer Generation!",
            "info")

        # Event-Polling
        self._poll_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        """Wird aufgerufen wenn das Fenster geschlossen wird."""
        self.running = False
        self.root.destroy()

    def _poll_events(self) -> None:
        """Prueft regelmaessig die Event-Queue und generiert Kommentare."""
        if not self.running:
            return

        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self._process_event(event)
            except queue.Empty:
                break

        # Naechster Poll in 200ms
        if self.root:
            self.root.after(200, self._poll_events)

    def _process_event(self, event: CommentaryEvent) -> None:
        """Verarbeitet ein Event und generiert einen passenden Kommentar."""
        now = time.time()
        data = event.data

        if event.event_type == "gen_start":
            gen = data.get("gen", 0)
            # Trennlinie zwischen Runden
            if gen > 0:
                self._add_separator(gen)
            comment = random.choice(GENERATION_START).format(**data)
            self._add_comment(comment, "info")

            # Milestone-Check
            if gen in MILESTONE_COMMENTS:
                self._add_comment(MILESTONE_COMMENTS[gen], "milestone")

        elif event.event_type == "gen_end":
            comment = random.choice(GENERATION_END).format(**data)
            self._add_comment(comment, "normal")

            # Kills-Analyse
            kills = data.get("kills", 0)
            if kills >= 20:
                self._add_comment(
                    random.choice(MANY_KILLS).format(**data), "danger")
            elif kills <= 5:
                self._add_comment(
                    random.choice(FEW_KILLS).format(**data), "highlight")

            # Fitness-Trends
            best_c = data.get("best_c", 0)
            best_h = data.get("best_h", 0)

            if best_c > self._last_collector_fitness + 5:
                self._add_comment(
                    random.choice(FITNESS_IMPROVING).format(fitness=best_c),
                    "highlight")
            elif best_c < self._last_collector_fitness - 10:
                self._add_comment(
                    random.choice(FITNESS_DECLINING), "warning")

            if best_h > self._last_hunter_fitness + 10:
                self._add_comment(
                    random.choice(HUNTER_IMPROVING).format(fitness=best_h),
                    "danger")

            self._last_collector_fitness = best_c
            self._last_hunter_fitness = best_h

            # Oekosystem-Balance (alle 10 Generationen)
            gen = data.get("gen", 0)
            if gen > 0 and gen % 10 == 0:
                self._add_comment(
                    random.choice(ECOSYSTEM_BALANCE), "info")

        elif event.event_type == "hall_of_fame":
            comment = random.choice(HALL_OF_FAME).format(**data)
            self._add_comment(comment, "milestone")

        elif event.event_type == "training_mode":
            mode = data.get("mode", "")
            if mode == "training":
                self._add_comment(
                    "Wechsel in den Turbo-Modus! Kein Rendering - maximale Geschwindigkeit.",
                    "info")
            else:
                self._add_comment(
                    "Zurueck zur Visualisierung. Schau den Robotern beim Lernen zu!",
                    "info")

        elif event.event_type == "custom":
            self._add_comment(data.get("text", ""), data.get("tag", "normal"))

    def _add_comment(self, text: str, tag: str = "normal") -> None:
        """Fuegt einen Kommentar zum Textfeld hinzu."""
        if not self.text_widget:
            return

        self.text_widget.config(state=tk.NORMAL)

        # Zeitstempel und Text ohne doppelte Leerzeilen
        timestamp = time.strftime("%H:%M:%S")
        self.text_widget.insert(tk.END, f"[{timestamp}] ", "time")
        self.text_widget.insert(tk.END, f"{text}\n", tag)

        # Auto-Scroll nach unten
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def _add_separator(self, gen: int) -> None:
        """Fuegt eine visuelle Trennlinie zwischen Generationen ein."""
        if not self.text_widget:
            return

        self.text_widget.config(state=tk.NORMAL)
        line = f"{'=' * 20}  Runde {gen}  {'=' * 20}\n"
        self.text_widget.insert(tk.END, line, "separator")
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)
