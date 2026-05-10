import tkinter as tk
from tkinter import messagebox
import pickle
import threading
from ai.hall_of_fame import HallOfFame


# Sensor-Layout: Inputs kommen paarweise (Distanz, Typ) pro Strahl
# Bei 5 Strahlen: In1=Dist_Links, In2=Typ_Links, In3=Dist_LinksM, In4=Typ_LinksM, ...
# Strahlen sind: Links-außen, Links-mitte, Mitte, Rechts-mitte, Rechts-außen

def get_input_label(node_id: int, num_inputs: int) -> str:
    if node_id >= 0:
        return str(node_id)
    
    idx = -node_id - 1
    vals_per_ray = 2
    if num_inputs >= 16 and (num_inputs - 1) % 5 == 0:
        vals_per_ray = 5
    elif num_inputs >= 12:
        vals_per_ray = 4
        
    if vals_per_ray == 5 and idx == num_inputs - 1:
        return "Radio In"
        
    num_rays = num_inputs // vals_per_ray
    
    ray_idx = idx // vals_per_ray
    val_idx = idx % vals_per_ray
    
    ray_names_5 = ["L", "LM", "M", "RM", "R"]
    ray_names_3 = ["L", "M", "R"]
    
    if num_rays == 5:
        ray_name = ray_names_5[ray_idx] if ray_idx < 5 else f"R{ray_idx}"
    elif num_rays == 3:
        ray_name = ray_names_3[ray_idx] if ray_idx < 3 else f"R{ray_idx}"
    else:
        ray_name = f"R{ray_idx}"
        
    if vals_per_ray == 5:
        val_names = ["Dist", "Bat", "Hunt", "Wall", "Coll"]
        return f"{val_names[val_idx]} {ray_name}"
    elif vals_per_ray == 4:
        val_names = ["Dist", "Bat", "Hunt", "Wall"]
        return f"{val_names[val_idx]} {ray_name}"
    else:
        val_names = ["Dist", "Typ"]
        return f"{val_names[val_idx]} {ray_name}"


def _analyze_strategy(genome, num_inputs: int, num_outputs: int) -> list[str]:
    """Analysiert die Gewichte eines NEAT-Genoms und leitet eine Strategie-Beschreibung ab.

    Schaut sich an, wie die Sensor-Inputs (Distanz & Typ) auf die Motor-Outputs
    verdrahtet sind, und formuliert daraus verständliche Sätze.
    """
    findings = []

    # Sammle alle aktiven Verbindungen, die direkt oder indirekt auf Outputs wirken
    # Für die einfache Analyse: nur direkte Input→Output Verbindungen
    direct_weights = {}  # (input_id, output_id) -> weight
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        in_node, out_node = cg.key
        direct_weights[(in_node, out_node)] = cg.weight

    vals_per_ray = 2
    has_radio = False
    if num_inputs >= 16 and (num_inputs - 1) % 5 == 0:
        vals_per_ray = 5
        has_radio = True
    elif num_inputs >= 12:
        vals_per_ray = 4
    num_rays = num_inputs // vals_per_ray

    # --- Analyse 1: Komplexität ---
    hidden_count = len([n for n in genome.nodes.keys() if n >= num_outputs])
    active_conns = sum(1 for c in genome.connections.values() if c.enabled)
    total_conns = len(genome.connections)

    if hidden_count == 0:
        findings.append("🧠 Einfaches Reflexnetz (keine versteckten Neuronen)")
    else:
        findings.append(f"🧠 Komplexes Netz mit {hidden_count} versteckten Neuronen")
    findings.append(f"🔗 {active_conns} aktive Verbindungen (von {total_conns} total)")

    # --- Analyse 2: Distanz-Reaktion (fährt auf Objekte zu oder weg?) ---
    dist_to_motor = []
    for ray_idx in range(num_rays):
        dist_input = -(ray_idx * vals_per_ray + 1)
        for out_id in range(num_outputs):
            w = direct_weights.get((dist_input, out_id), 0)
            dist_to_motor.append(w)

    avg_dist_weight = sum(dist_to_motor) / max(1, len(dist_to_motor))
    if avg_dist_weight > 0.3:
        findings.append("🎯 Zielorientiert: Fährt auf nahe Objekte zu (pos. Distanz→Motor)")
    elif avg_dist_weight < -0.3:
        findings.append("🛡️ Vorsichtig: Weicht nahen Objekten aus (neg. Distanz→Motor)")
    else:
        findings.append("😐 Neutral gegenüber Distanz (kein klares Annäherungsverhalten)")

    # --- Analyse 3: Typ-Reaktion (reagiert auf Batterien vs. Jäger?) ---
    type_to_motor = []
    for ray_idx in range(num_rays):
        type_input = -(ray_idx * vals_per_ray + 2)
        for out_id in range(num_outputs):
            w = direct_weights.get((type_input, out_id), 0)
            type_to_motor.append(w)

    avg_type_weight = sum(type_to_motor) / max(1, len(type_to_motor))
    if avg_type_weight > 0.3:
        findings.append("⚡ Typ-sensitiv: Reagiert stärker auf Batterien/Jäger (pos. Typ→Motor)")
    elif avg_type_weight < -0.3:
        findings.append("🚫 Typ-aversiv: Vermeidet Objekte mit hohem Typ-Wert (Jäger!)")

    # --- Analyse 4: Links/Rechts-Asymmetrie (Kurvenverhalten) ---
    left_to_right = 0  # linke Sensoren → rechter Motor (1)
    right_to_left = 0  # rechte Sensoren → linker Motor (0)

    left_inputs = [-(0 * vals_per_ray + i + 1) for i in range(vals_per_ray)]
    right_inputs = [-((num_rays - 1) * vals_per_ray + i + 1) for i in range(vals_per_ray)]

    for inp in left_inputs:
        left_to_right += direct_weights.get((inp, 1), 0)
    for inp in right_inputs:
        right_to_left += direct_weights.get((inp, 0), 0)

    if left_to_right > 0.5 and right_to_left > 0.5:
        findings.append("🔄 Wendemanöver: Dreht sich zu Objekten hin (links→rechts, rechts→links)")
    elif left_to_right < -0.5 and right_to_left < -0.5:
        findings.append("↩️ Ausweichmanöver: Dreht sich von Objekten weg")

    # --- Analyse 5: Mitte-Sensor-Dominanz ---
    center_ray_idx = num_rays // 2
    center_dist = -(center_ray_idx * vals_per_ray + 1)
    center_w_r = direct_weights.get((center_dist, 0), 0)
    center_w_l = direct_weights.get((center_dist, 1), 0)
    center_total = abs(center_w_r) + abs(center_w_l)

    if center_total > 2.0:
        if center_w_r > 0 and center_w_l > 0:
            findings.append("⬆️ Frontal-Angriff: Mittlerer Sensor treibt beide Motoren an → geradeaus auf Ziel!")
        elif center_w_r < 0 and center_w_l < 0:
            findings.append("⬇️ Frontal-Flucht: Bremst ab wenn etwas direkt voraus ist")

    # --- Analyse 6: Stärkste einzelne Verbindung ---
    strongest_w = 0
    strongest_conn = None
    for (in_n, out_n), w in direct_weights.items():
        if abs(w) > abs(strongest_w) and in_n < 0:
            strongest_w = w
            strongest_conn = (in_n, out_n)

    if strongest_conn:
        in_label = get_input_label(strongest_conn[0], num_inputs)
        if strongest_conn[1] == 0:
            out_label = "Motor R"
        elif strongest_conn[1] == 1:
            out_label = "Motor L"
        else:
            out_label = "Radio Out"
        direction = "verstärkt" if strongest_w > 0 else "hemmt"
        findings.append(f"💪 Stärkste Verbindung: {in_label} {direction} {out_label} (Gewicht: {strongest_w:.2f})")

    if not findings:
        findings.append("❓ Keine klare Strategie erkennbar (möglicherweise noch zu früh in der Evolution)")

    return findings


class BrainViewerWindow:
    """Zeigt das neuronale Netz eines NEAT-Genoms aus der Hall of Fame."""

    def __init__(self, hall: HallOfFame, num_inputs: int = 10, num_outputs: int = 2) -> None:
        self.hall = hall
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.running = False
        self.thread = None
        self.root = None
        self.canvas = None
        self.current_idx = 0

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_window, daemon=True)
            self.thread.start()

    def _run_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("NEAT Brain Viewer")
        self.root.geometry("1100x750")
        self.root.configure(bg="#121218")

        # Top Control Bar
        ctrl_frame = tk.Frame(self.root, bg="#1a1a24", height=60)
        ctrl_frame.pack(fill=tk.X, side=tk.TOP)

        btn_prev = tk.Button(ctrl_frame, text="< Vorheriger", command=self._prev_genome,
                             bg="#2a2a3a", fg="white", font=("Segoe UI", 12), relief=tk.FLAT, padx=10, pady=5)
        btn_prev.pack(side=tk.LEFT, padx=10, pady=10)

        btn_delete = tk.Button(ctrl_frame, text="Löschen", command=self._delete_genome,
                             bg="#E65050", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, padx=10, pady=5)
        btn_delete.pack(side=tk.LEFT, padx=10, pady=10)

        self.lbl_info = tk.Label(ctrl_frame, text="", bg="#1a1a24", fg="#00C878", font=("Segoe UI", 14, "bold"))
        self.lbl_info.pack(side=tk.LEFT, expand=True)

        btn_next = tk.Button(ctrl_frame, text="Nächster >", command=self._next_genome,
                             bg="#2a2a3a", fg="white", font=("Segoe UI", 12), relief=tk.FLAT, padx=10, pady=5)
        btn_next.pack(side=tk.RIGHT, padx=10, pady=10)

        # Bottom Legend
        legend_frame = tk.Frame(self.root, bg="#1a1a24", height=40)
        legend_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(legend_frame, text="Sensoren (Inputs)", fg="#64C8FF", bg="#1a1a24", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=20, pady=5)
        tk.Label(legend_frame, text="Motoren (Outputs)", fg="#E6C832", bg="#1a1a24", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=20, pady=5)
        tk.Label(legend_frame, text="Versteckte Neuronen", fg="#aaaaaa", bg="#1a1a24", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=20, pady=5)
        tk.Label(legend_frame, text="Grün=Positiv(+)", fg="#00C878", bg="#1a1a24", font=("Segoe UI", 10, "bold")).pack(side=tk.RIGHT, padx=15, pady=5)
        tk.Label(legend_frame, text="Rot=Negativ(-)", fg="#E65050", bg="#1a1a24", font=("Segoe UI", 10, "bold")).pack(side=tk.RIGHT, padx=15, pady=5)

        # Strategie-Panel (rechts, nimmt Platz neben dem Canvas)
        self.strategy_frame = tk.Frame(self.root, bg="#1a1a24", width=300)
        self.strategy_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=(0, 5), pady=5)
        self.strategy_frame.pack_propagate(False)

        strategy_title = tk.Label(self.strategy_frame, text="📋 Strategie-Analyse",
                                  bg="#1a1a24", fg="#E6C832", font=("Segoe UI", 14, "bold"))
        strategy_title.pack(pady=(10, 5), padx=10, anchor=tk.W)

        self.strategy_text = tk.Text(self.strategy_frame, bg="#1a1a24", fg="#cccccc",
                                     font=("Segoe UI", 11), wrap=tk.WORD, relief=tk.FLAT,
                                     padx=10, pady=5, spacing3=6)
        self.strategy_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 10))
        self.strategy_text.config(state=tk.DISABLED)
        self.strategy_text.tag_config("finding", foreground="#cccccc", font=("Segoe UI", 11))
        self.strategy_text.tag_config("emoji", foreground="#E6C832", font=("Segoe UI", 11))

        # Canvas for Drawing (links)
        self.canvas = tk.Canvas(self.root, bg="#121218", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self._draw_current_genome()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        self.running = False
        self.root.quit()
        self.root.destroy()

    def _prev_genome(self) -> None:
        if not self.hall.entries: return
        self.current_idx = (self.current_idx - 1) % len(self.hall.entries)
        self._draw_current_genome()

    def _next_genome(self) -> None:
        if not self.hall.entries: return
        self.current_idx = (self.current_idx + 1) % len(self.hall.entries)
        self._draw_current_genome()

    def _delete_genome(self) -> None:
        if not self.hall.entries: return
        entry = self.hall.entries[self.current_idx]
        if messagebox.askyesno("Löschen bestätigen", f"Möchtest du '{entry.name}' wirklich aus der Hall of Fame löschen?"):
            self.hall.remove_entry(self.current_idx)
            if self.current_idx >= len(self.hall.entries):
                self.current_idx = max(0, len(self.hall.entries) - 1)
            self._draw_current_genome()

    def _update_strategy_panel(self, genome) -> None:
        """Aktualisiert das Strategie-Panel mit der Analyse des Genoms."""
        findings = _analyze_strategy(genome, self.num_inputs, self.num_outputs)
        self.strategy_text.config(state=tk.NORMAL)
        self.strategy_text.delete("1.0", tk.END)
        for f in findings:
            self.strategy_text.insert(tk.END, f + "\n\n", "finding")
        self.strategy_text.config(state=tk.DISABLED)

    def _draw_current_genome(self) -> None:
        self.canvas.delete("all")
        if not self.hall.entries:
            self.lbl_info.config(text="Hall of Fame ist leer")
            self.canvas.create_text(350, 300, text="Keine Roboter in der Hall of Fame", fill="white", font=("Segoe UI", 20))
            self.strategy_text.config(state=tk.NORMAL)
            self.strategy_text.delete("1.0", tk.END)
            self.strategy_text.config(state=tk.DISABLED)
            return

        entry = self.hall.entries[self.current_idx]
        rank = self.current_idx + 1
        self.lbl_info.config(text=f"Rang {rank}: {entry.name} | Fitness: {entry.fitness:.1f} | Batterien: {entry.batteries_collected}")

        # Unpickle genome
        try:
            genome = pickle.loads(entry.genome_data)
        except Exception as e:
            self.canvas.create_text(350, 300, text=f"Fehler beim Laden des Genoms:\n{e}", fill="red", font=("Segoe UI", 14))
            return

        # Strategie-Analyse aktualisieren
        self._update_strategy_panel(genome)

        # Canvas bounds (dynamisch)
        self.canvas.update_idletasks()
        w = max(400, self.canvas.winfo_width())
        h = max(300, self.canvas.winfo_height())

        node_pos = {}

        # Determine hidden nodes
        hidden_nodes = [n for n in genome.nodes.keys() if n >= self.num_outputs]

        # 1. Inputs (Left) – mit beschreibenden Labels
        in_x = 120
        if self.num_inputs > 0:
            in_y_step = (h - 100) / max(1, self.num_inputs)
            for i in range(self.num_inputs):
                node_id = -(i + 1)
                node_pos[node_id] = (in_x, 50 + i * in_y_step + in_y_step/2)

        # 2. Outputs (Right)
        out_x = w - 100
        if self.num_outputs > 0:
            out_y_step = (h - 100) / max(1, self.num_outputs)
            for i in range(self.num_outputs):
                node_id = i
                node_pos[node_id] = (out_x, 50 + i * out_y_step + out_y_step/2)

        # 3. Hidden (Middle)
        if hidden_nodes:
            mid_x = w // 2
            hid_y_step = (h - 100) / max(1, len(hidden_nodes))
            for i, node_id in enumerate(hidden_nodes):
                offset_x = mid_x + (i % 2) * 80 - 40
                node_pos[node_id] = (offset_x, 50 + i * hid_y_step + hid_y_step/2)

        # Draw Connections (Edges)
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            in_node, out_node = cg.key

            if in_node not in node_pos:
                if in_node < 0:
                    node_pos[in_node] = (in_x, h - 50)
                else:
                    continue
            if out_node not in node_pos:
                continue

            x1, y1 = node_pos[in_node]
            x2, y2 = node_pos[out_node]

            weight = cg.weight
            color = "#00C878" if weight > 0 else "#E65050"
            width = min(8, max(1, abs(weight) * 1.5))

            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, arrow=tk.LAST, arrowshape=(10, 12, 4))

        # Draw Nodes
        for node_id, (x, y) in node_pos.items():
            color = "#aaaaaa"
            label = str(node_id)
            r = 20

            if node_id < 0:
                color = "#64C8FF"
                label = get_input_label(node_id, self.num_inputs)
                r = 22  # Etwas größer für den Text
            elif node_id < self.num_outputs:
                if node_id == 0:
                    label = "Motor R"
                    color = "#E6C832"
                elif node_id == 1:
                    label = "Motor L"
                    color = "#E6C832"
                else:
                    label = "Radio Out"
                    color = "#D264FF"
                r = 24

            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#2a2a3a", outline=color, width=2)
            self.canvas.create_text(x, y, text=label, fill="white", font=("Segoe UI", 8, "bold"))
