"""
stats_graph.py – Echtzeit-Graphen-Fenster fuer das Neuro-Oekosystem.

Zeigt in einem separaten Fenster Live-Graphen fuer:
- Beste und durchschnittliche Fitness (Sammler + Jaeger)
- Anzahl Kills pro Generation
- Ueberlebensrate der Sammler
- Gesammelte Batterien

Laeuft in einem eigenen Thread mit matplotlib + TkAgg Backend.
"""

import threading
import queue
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

GRAPH_MAX_WIDTH = 760
GRAPH_MIN_WIDTH = 560
HOF_RESERVED_WIDTH = 540
GRAPH_HEIGHT_RATIO = 0.55


class StatsData:
    """Datencontainer fuer eine Generation."""

    def __init__(self, gen: int, best_c: float, avg_c: float,
                 best_h: float, avg_h: float,
                 kills: int, alive: int, total_collectors: int,
                 batteries: int, gen_time: float = 0.0,
                 escape_ratio: float = 0.0):
        self.gen = gen
        self.best_c = best_c
        self.avg_c = avg_c
        self.best_h = best_h
        self.avg_h = avg_h
        self.kills = kills
        self.alive = alive
        self.total_collectors = total_collectors
        self.batteries = batteries
        self.gen_time = gen_time
        self.escape_ratio = escape_ratio


class StatsGraphWindow:
    """Separates Fenster mit Live-Graphen der Evolutionsstatistiken.

    Zeigt 4 Graphen:
    1. Sammler-Fitness (Beste + Durchschnitt)
    2. Jaeger-Fitness (Beste + Durchschnitt)
    3. Kills pro Generation + Ueberlebende Sammler
    4. Gesammelte Batterien pro Generation
    """

    def __init__(self, offset_x: int = 0) -> None:
        self.data_queue: queue.Queue = queue.Queue()
        self.command_queue: queue.Queue = queue.Queue()
        self.running = False
        self.thread = None
        self.offset_x = offset_x

        # Daten-Listen
        self.generations: list[int] = []
        self.best_collector: list[float] = []
        self.avg_collector: list[float] = []
        self.best_hunter: list[float] = []
        self.avg_hunter: list[float] = []
        self.kills_per_gen: list[int] = []
        self.alive_per_gen: list[int] = []
        self.batteries_per_gen: list[int] = []
        self.escape_ratio_per_gen: list[float] = []

    def start(self) -> None:
        """Startet das Graphen-Fenster in einem separaten Thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Schliesst das Fenster."""
        self.running = False

    def add_generation(self, gen: int, best_c: float, avg_c: float,
                       best_h: float, avg_h: float,
                       kills: int, alive: int, total_collectors: int,
                       batteries: int, gen_time: float = 0.0,
                       escape_ratio: float = 0.0) -> None:
        """Fuegt Daten einer Generation hinzu (thread-sicher)."""
        self.data_queue.put(StatsData(
            gen, best_c, avg_c, best_h, avg_h,
            kills, alive, total_collectors, batteries, gen_time, escape_ratio))

    def get_commands(self) -> list[str]:
        """Gibt alle neuen Kommandos (Button-Klicks) zurueck."""
        cmds = []
        while not self.command_queue.empty():
            try:
                cmds.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def update_speed_label(self, speed: int) -> None:
        """Aktualisiert die Geschwindigkeits-Anzeige im Graphen-Fenster."""
        if self.running and hasattr(self, 'lbl_speed'):
            try:
                if speed >= 500:
                    self.lbl_speed.config(text="⚡ TURBO", fg="#E6C832")
                else:
                    self.lbl_speed.config(text=f"Speed: {speed}x", fg="#aaaabc")
            except Exception:
                pass

    def _toggle_turbo(self) -> None:
        """Schaltet den Turbo-Modus ein/aus."""
        self.turbo_active = not self.turbo_active
        if self.turbo_active:
            self.btn_turbo.config(text="🚀 Turbo: AN", bg="#E6C832", fg="black")
            self.btn_turbo.config(text="Turbo: AN")
            self._send_command("turbo_on")
        else:
            self.btn_turbo.config(text="🚀 Turbo: AUS", bg="#2a2a3a", fg="white")
            self.btn_turbo.config(text="Turbo: AUS")
            self._send_command("turbo_off")

    def _send_command(self, cmd: str) -> None:
        self.command_queue.put(cmd)

    def _run_window(self) -> None:
        """Hauptschleife des matplotlib-Fensters."""
        self.root = tk.Tk()
        self.root.title("Neuro-Oekosystem - Lernkurven")
        
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        
        # Dynamisch an verbleibenden Platz anpassen (65% der Höhe minus Taskleistenpuffer)
        available_w = max(GRAPH_MIN_WIDTH, sw - self.offset_x - HOF_RESERVED_WIDTH)
        win_w = min(GRAPH_MAX_WIDTH, available_w)
        win_h = int(sh * GRAPH_HEIGHT_RATIO) - 36
        
        self.root.geometry(f"{win_w}x{win_h}+{self.offset_x}+0")
        self.root.minsize(GRAPH_MIN_WIDTH, 360)
        self.root.configure(bg="#121218")

        # Matplotlib Figure mit dunklem Theme
        plt.style.use('dark_background')
        # Extrem kleine Basisgröße. Tkinter expandiert es dann korrekt.
        self.fig, ax_main = plt.subplots(1, 1, figsize=(4.8, 2.9), facecolor='#121218')
        self.axes = ax_main  # Nur noch ein Haupt-Achsen-Objekt
        
        # Manuelles Layout
        self.fig.subplots_adjust(left=0.13, right=0.86, top=0.86, bottom=0.19)

        # Farben
        c_green = '#00C878'
        c_red = '#E65050'
        c_yellow = '#E6C832'
        c_blue = '#64C8FF'
        c_grid = '#2a2a3a'

        # --- Haupt-Achse (Links): Fitness ---
        self.ax1 = ax_main
        self.ax1.set_title("Evolutionäre Entwicklung: Fitness & Metriken", color='#ffffff',
                           fontsize=11, fontweight='bold', pad=6)
        self.ax1.set_title("Fitness & Metriken", color='#ffffff',
                           fontsize=11, fontweight='bold', pad=6)
        self.ax1.set_xlabel("Generation", color='#aaaabc', fontsize=8)
        self.ax1.set_ylabel("Fitness", color='#aaaabc', fontsize=8)
        
        # Fitness-Linien (Beste)
        self.line_best_c, = self.ax1.plot([], [], color=c_green, linewidth=2.5, label='Sammler (Beste)')
        self.line_best_h, = self.ax1.plot([], [], color=c_red, linewidth=2.5, label='Jäger (Beste)')
        
        self.ax1.set_facecolor('#1a1a24')
        self.ax1.grid(True, color=c_grid, alpha=0.5, linewidth=0.5)
        self.ax1.tick_params(colors='#aaaabc', labelsize=8)

        # --- Sekundäre Achse (Rechts): Anzahl (Kills, Batterien) ---
        self.ax2 = self.ax1.twinx()
        self.ax2.set_ylabel("Anzahl", color='#aaaabc', fontsize=8)
        
        self.line_kills, = self.ax2.plot([], [], color=c_yellow, linewidth=2.0, linestyle='--', label='Kills')
        self.line_bats, = self.ax2.plot([], [], color=c_blue, linewidth=2.0, linestyle=':', label='Batterien')
        self.line_alive, = self.ax2.plot([], [], color='#00885A', linewidth=2.0, linestyle='-.', label='Überlebende')
        self.line_escape, = self.ax2.plot([], [], color='#FF00FF', linewidth=2.0, linestyle=':', label='Flucht (%)')
        
        self.ax2.tick_params(colors='#aaaabc', labelsize=8)

        # Legenden für beide Achsen kombinieren
        lines_1, labels_1 = self.ax1.get_legend_handles_labels()
        lines_2, labels_2 = self.ax2.get_legend_handles_labels()
        self.ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left',
                        fontsize=7, facecolor='#1a1a24', edgecolor='#333348',
                        ncol=2, borderpad=0.4, labelspacing=0.3,
                        handlelength=1.8, handletextpad=0.5)

        # --- Controls Leiste unten (Zuerst packen, damit sie sichtbar bleibt!) ---
        ctrl_frame = tk.Frame(self.root, bg="#121218", height=38)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(2, 6))

        # Canvas in Tkinter einbetten (Danach packen, damit er nur den Rest nimmt)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.lbl_speed = tk.Label(ctrl_frame, text="Speed: 1x", bg="#121218", fg="#aaaabc", font=("Segoe UI", 10, "bold"))
        self.lbl_speed.pack(side=tk.LEFT, padx=(6, 8))

        self.turbo_active = False
        self.btn_turbo = tk.Button(ctrl_frame, text="🚀 Turbo: AUS", command=self._toggle_turbo,
                                   bg="#2a2a3a", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=8, pady=3)
        self.btn_turbo.pack(side=tk.LEFT, padx=4)
        self.btn_turbo.config(text="Turbo: AUS")

        self.lbl_gen = tk.Label(ctrl_frame, text="Gen: 0", bg="#121218", fg="#aaaabc", font=("Segoe UI", 10, "bold"))
        self.lbl_gen.pack(side=tk.LEFT, padx=8)

        self.lbl_time = tk.Label(ctrl_frame, text="⏱ --", bg="#121218", fg="#888898", font=("Segoe UI", 12))
        self.lbl_time.pack(side=tk.LEFT, padx=4)
        self.lbl_time.config(text="--", font=("Segoe UI", 10))

        btn_brain = tk.Button(ctrl_frame, text="🧠 Brain Viewer", command=lambda: self._send_command("open_brain_viewer"),
                              bg="#2a2a3a", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=8, pady=3)
        btn_brain.config(text="Brain")
        btn_brain.pack(side=tk.LEFT, padx=4)

        btn_sensors = tk.Button(ctrl_frame, text="Strahlen Ein/Aus", command=lambda: self._send_command("toggle_sensors"),
                                bg="#00C878", fg="black", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=8, pady=3)
        btn_sensors.config(text="Strahlen")
        btn_sensors.pack(side=tk.LEFT, padx=4)



        # Polling starten
        self._poll_data()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        """Wird aufgerufen wenn das Fenster geschlossen wird."""
        self.running = False
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()

    def _poll_data(self) -> None:
        """Prueft regelmaessig die Daten-Queue und aktualisiert Graphen."""
        if not self.running:
            if self.root:
                self._on_close()
            return

        updated = False
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.generations.append(data.gen)
                self.best_collector.append(data.best_c)
                self.avg_collector.append(data.avg_c)
                self.best_hunter.append(data.best_h)
                self.avg_hunter.append(data.avg_h)
                self.kills_per_gen.append(data.kills)
                self.alive_per_gen.append(data.alive)
                self.batteries_per_gen.append(data.batteries)
                self.escape_ratio_per_gen.append(data.escape_ratio * 100.0) # Als Prozentwert fuer Achse 2
                updated = True
                # Generations-Label aktualisieren
                try:
                    self.lbl_gen.config(text=f"Gen: {data.gen}")
                    if data.gen_time >= 60:
                        mins = int(data.gen_time // 60)
                        secs = data.gen_time % 60
                        self.lbl_time.config(text=f"⏱ {mins}m {secs:.0f}s")
                    else:
                        self.lbl_time.config(text=f"⏱ {data.gen_time:.1f}s")
                except Exception:
                    pass
            except queue.Empty:
                break

        if updated:
            self._update_plots()

        if self.root:
            self.root.after(500, self._poll_data)

    def _update_plots(self) -> None:
        """Aktualisiert alle 4 Graphen mit den neuesten Daten."""
        gens = self.generations

        # Daten für Fitness (Linke Achse) aktualisieren
        self.line_best_c.set_data(gens, self.best_collector)
        self.line_best_h.set_data(gens, self.best_hunter)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Daten für Anzahl (Rechte Achse) aktualisieren
        self.line_kills.set_data(gens, self.kills_per_gen)
        self.line_bats.set_data(gens, self.batteries_per_gen)
        self.line_alive.set_data(gens, self.alive_per_gen)
        self.line_escape.set_data(gens, self.escape_ratio_per_gen)
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.canvas.draw_idle()
