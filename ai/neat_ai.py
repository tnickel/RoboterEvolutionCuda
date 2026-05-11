"""
neat_ai.py – NEAT-Neuroevolution fuer das Neuro-Oekosystem.

Stufe 4: Co-Evolution mit zwei separaten Populationen
(Sammler und Jaeger) in derselben Welt.

Enthaelt:
- CoEvolutionManager: Verwaltet zwei NEAT-Populationen parallel
- Integration mit Hall of Fame und Spatial Grid
"""

import os
import csv
import math
import random
import pickle
import time
import neat
import pygame
from config.config_manager import SimConfig
from core.fast_net import FastNetwork
from core.cuda_net import CudaBatchedNetwork
from core.world import World
from core.entities import Collector, Hunter, Battery, ENTITY_BATTERY, ENTITY_COLLECTOR, ENTITY_HUNTER
from core.sensors import draw_sensors, cast_rays_batch
from core.spatial_grid import SpatialGrid
from core.fast_net import FastNetwork
from ai.hall_of_fame import HallOfFame, HallOfFameEntry
from ui.commentary import CommentaryWindow
from ui.stats_graph import StatsGraphWindow
from ui.brain_viewer import BrainViewerWindow
from ui.hall_of_fame_window import HallOfFameWindow


# --- Farb-Konstanten fuer HUD ------------------------------------------------
COLOR_HUD_BG = (18, 18, 24, 180)
COLOR_HUD_TEXT = (255, 255, 255)  # Reines Weiß für hohen Kontrast
COLOR_HUD_ACCENT = (0, 200, 150)
COLOR_HUD_WARN = (230, 200, 50)
COLOR_HUD_GEN = (255, 255, 255)   # Reines Weiß für hohen Kontrast
COLOR_HUD_HUNTER = (230, 80, 80)
COLOR_HUD_COLLECTOR = (0, 200, 120)

FPS_TARGET = 60


GENERATION_LOG_FIELDS = [
    "run_id",
    "timestamp",
    "generation",
    "duration_sec",
    "training_mode",
    "speed_multiplier",
    "collectors_total",
    "collectors_real",
    "collectors_injected",
    "collectors_alive",
    "hunters_total",
    "hunters_alive",
    "batteries_collected",
    "batteries_collected_real",
    "kills",
    "best_collector_fitness",
    "avg_collector_fitness",
    "best_hunter_fitness",
    "avg_hunter_fitness",
    "avg_fit_battery",
    "avg_fit_proximity",
    "avg_fit_survival",
    "avg_fit_hunter_penalty",
    "avg_fit_escape",
    "avg_fit_approach",
    "avg_fit_death",
    "avg_fit_idle",
    "collector_sees_hunter_frames",
    "collector_danger_frames",
    "collector_escape_events",
    "collector_approach_events",
    "collector_neutral_events",
    "escape_ratio",
    "approach_ratio",
    "avg_nearest_hunter_dist",
]


def _load_neat_config(config_path: str, pop_size: int,
                      num_inputs: int, num_outputs: int = 3) -> neat.Config:
    """Laedt und konfiguriert eine NEAT-Config-Datei.

    Erstellt eine temporaere Kopie mit angepassten Werten fuer
    pop_size und num_inputs, damit NEAT von Anfang an die
    richtige Topologie verwendet.

    Args:
        config_path: Pfad zur NEAT-Config-Datei.
        pop_size: Populationsgroesse.
        num_inputs: Anzahl Inputs.
        num_outputs: Anzahl Outputs.

    Returns:
        Konfigurierte neat.Config Instanz.
    """
    import tempfile

    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             config_path)

    # Config-Datei lesen und dynamische Werte ersetzen
    with open(full_path, 'r') as f:
        content = f.read()

    # num_inputs, num_outputs und pop_size in der Config ersetzen
    import re
    content = re.sub(r'num_inputs\s*=\s*\d+', f'num_inputs              = {num_inputs}', content)
    content = re.sub(r'num_outputs\s*=\s*\d+', f'num_outputs              = {num_outputs}', content)
    content = re.sub(r'pop_size\s*=\s*\d+', f'pop_size              = {pop_size}', content)

    # Temporaere Config-Datei schreiben
    tmp_dir = os.path.dirname(full_path)
    tmp_path = os.path.join(tmp_dir, f"_tmp_{os.path.basename(config_path)}")
    with open(tmp_path, 'w') as f:
        f.write(content)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        tmp_path
    )

    # Temporaere Datei aufräumen
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return config


def _append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    """Appendet eine Zeile in eine CSV-Datei und schreibt bei Bedarf den Header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not needs_header:
        with open(path, "r", newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line and first_line.split(",") != fieldnames:
            backup_path = path.replace(".csv", f"_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            os.replace(path, backup_path)
            needs_header = True
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


class CoEvolutionManager:
    """Verwaltet die Co-Evolution zweier NEAT-Populationen.

    Sammler und Jaeger werden gleichzeitig in derselben Welt simuliert.
    Beide Populationen haben getrennte Fitness-Funktionen und evolvieren
    parallel. Ein Spatial Grid optimiert die Performance.

    Attributes:
        sim_config: Simulationskonfiguration
        hall: Hall of Fame
        collector_pop: NEAT-Population der Sammler
        hunter_pop: NEAT-Population der Jaeger
        generation: Aktuelle Generation
        best_collector_fitness: Beste Sammler-Fitness
        best_hunter_fitness: Beste Jaeger-Fitness
    """

    def __init__(self, sim_config: SimConfig, hall: HallOfFame,
                 commentary: CommentaryWindow | None = None,
                 stats_graph: StatsGraphWindow | None = None,
                 hof_window: HallOfFameWindow | None = None,
                 injected_entries: list[HallOfFameEntry] | None = None) -> None:
        self.sim_config = sim_config
        self.hall = hall
        self.commentary = commentary
        self.stats_graph = stats_graph
        self.hof_window = hof_window
        self.generation = 0
        self.best_collector_fitness = 0.0
        self.best_hunter_fitness = 0.0
        self.avg_collector_fitness = 0.0
        self.avg_hunter_fitness = 0.0
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(os.getcwd(), "log")
        self.generation_log_path = os.path.join(self.log_dir, "evolution_generations.csv")
        os.makedirs(self.log_dir, exist_ok=True)

        # Welt einmalig erstellen (wird alle 10 Generationen erneuert)
        self.world = World(sim_config)
        self.world_regen_interval = 10

        num_inputs = sim_config.sensor_ray_count * 5 + 2  # 5 Werte pro Strahl + 2 direkte (dx/dy) Nearest-Feind/Beute-Inputs

        # --- Sammler-Population -----------------------------------------------
        self.collector_config = _load_neat_config(
            "config-collector.txt", sim_config.collector_pop_size, num_inputs)
        self.collector_pop = neat.Population(self.collector_config)

        # --- Jaeger-Population ------------------------------------------------
        self.hunter_config = _load_neat_config(
            "config-hunter.txt", sim_config.hunter_pop_size, num_inputs)
        self.hunter_pop = neat.Population(self.hunter_config)

        # Hall-of-Fame Genome dauerhaft als Gaeste speichern
        self.injected_entries = injected_entries or []

        # Reporter

    def run_generation(self, screen: pygame.Surface | None,
                       clock: pygame.time.Clock,
                       show_sensors: bool, show_radio: bool, training_mode: bool,
                       speed_multiplier: int) -> tuple[bool, bool, bool, bool, int]:
        """Fuehrt eine Generation der Co-Evolution durch.

        Args:
            screen: Pygame-Screen (None = kein Rendering)
            clock: Pygame-Clock
            show_sensors: Ob Sensoren gezeichnet werden
            show_radio: Ob Radio-Funkwellen gezeichnet werden
            training_mode: Ob im schnellen Training-Modus
            speed_multiplier: Simulationsschritte pro Frame

        Returns:
            Tuple (continue, show_sensors, show_radio, training_mode, speed_multiplier)
            continue=False wenn Benutzer beendet hat.
        """
        # --- Welt: Wiederverwenden oder alle N Generationen erneuern ----------
        if self.generation > 0 and self.generation % self.world_regen_interval == 0:
            self.world = World(self.sim_config)
            print(f"[NEAT] Neue Welt generiert (alle {self.world_regen_interval} Generationen)")
        world = self.world
        # Batterien zuruecksetzen fuer neue Runde
        for b in world.batteries:
            b.active = True
            b.respawn_timer = 0
        grid = SpatialGrid(cell_size=150)

        # --- Sammler erstellen -------------------------------------------------
        collector_genomes = list(self.collector_pop.population.items())
        collectors: list[Collector] = []
        collector_genome_list = []
        collector_nets = []

        for genome_id, genome in collector_genomes:
            genome.fitness = 0.0
            neat_net = neat.nn.FeedForwardNetwork.create(genome, self.collector_config)
            net = FastNetwork(neat_net)  # JIT-kompiliertes Netz (10-20x schneller)
            x, y = world.spawn_robot_position()
            c = Collector(x, y, self.sim_config)
            c.genome = genome
            c.net = net
            c.is_injected = getattr(genome, 'is_injected', False)
            collectors.append(c)
            collector_genome_list.append(genome)
            collector_nets.append(net)

        # --- Hall of Fame Gaeste hinzufuegen ---
        live_guests = self.hall.entries[:5]
        for entry in live_guests:
            try:
                loaded_genome = self.hall.get_genome(entry)
                loaded_genome.is_guest = True
                net = FastNetwork(neat.nn.FeedForwardNetwork.create(loaded_genome, self.collector_config))
                c = Collector(*self.world.spawn_robot_position(), self.sim_config)
                
                c.color = (0, 255, 255)  # Cyan
                c.is_guest = True
                c.is_injected = True
                c.hof_name = entry.name
                c.genome = loaded_genome
                
                collectors.append(c)
                collector_nets.append(net)
                collector_genome_list.append(loaded_genome)
            except Exception as e:
                print(f"[NEAT] Fehler beim Laden von Gast {entry.name}: {e}")


        print(f"[NEAT] --- Generation {self.generation} startet ---")
        guest_names = [getattr(c, 'hof_name', 'Unknown') for c in collectors if getattr(c, 'is_guest', False)]
        if guest_names:
            print(f"[NEAT] {len(guest_names)} Hall of Fame Gaeste im Spielfeld: {', '.join(guest_names)}")
        
        expected_total = self.sim_config.collector_pop_size + len(live_guests)
        if len(collectors) == expected_total:
            print(f"[NEAT] SICHERHEITSCHECK OK: Roboteranzahl ist korrekt erhoeht auf {len(collectors)}.")
        else:
            print(f"[NEAT] SICHERHEITSCHECK FEHLGESCHLAGEN! Erwartet: {expected_total}, Aktuell: {len(collectors)}.")

        # --- Jaeger erstellen --------------------------------------------------
        hunter_genomes = list(self.hunter_pop.population.items())
        hunters: list[Hunter] = []
        hunter_genome_list = []
        hunter_nets = []

        for genome_id, genome in hunter_genomes:
            genome.fitness = 0.0
            neat_net = neat.nn.FeedForwardNetwork.create(genome, self.hunter_config)
            net = FastNetwork(neat_net)  # JIT-kompiliertes Netz
            x, y = world.spawn_robot_position()
            h = Hunter(x, y, self.sim_config)
            h.genome = genome
            h.net = net
            hunters.append(h)
            hunter_genome_list.append(genome)
            hunter_nets.append(net)

        all_robots = collectors + hunters
        
        # CUDA Batch Networks kompilieren
        collector_batch_net = None
        hunter_batch_net = None
        try:
            import cupy as cp
            # Dummy call to ensure cupy is working
            cp.zeros(1)
            collector_batch_net = CudaBatchedNetwork(collector_nets)
            hunter_batch_net = CudaBatchedNetwork(hunter_nets)
        except Exception as e:
            print(f"[NEAT] CUDA GPU Beschleunigung nicht verfuegbar, nutze CPU Fallback: {e}")

        # Commentary: Generation Start
        if self.commentary:
            self.commentary.post_event("gen_start",
                gen=self.generation,
                collectors=len(collectors),
                hunters=len(hunters))

        # --- Simulation --------------------------------------------------------
        total_frames = self.sim_config.simulation_frames
        current_step = 0
        gen_start_time = time.time()
        collector_sees_hunter_frames = 0
        collector_danger_frames = 0
        collector_escape_events = 0
        collector_approach_events = 0
        collector_neutral_events = 0
        nearest_hunter_dist_sum = 0.0
        nearest_hunter_dist_count = 0

        # Initiale Speed-Anzeige
        if self.stats_graph:
            self.stats_graph.update_speed_label(speed_multiplier)

        while current_step < total_frames:
            # Externe Kommandos vom Graph-Fenster
            if self.stats_graph:
                for cmd in self.stats_graph.get_commands():
                    if cmd == "turbo_on":
                        training_mode = True
                        speed_multiplier = 500
                        self.stats_graph.update_speed_label(speed_multiplier)
                        print("[NEAT] Turbo AN (Training-Modus)")
                    elif cmd == "turbo_off":
                        training_mode = False
                        speed_multiplier = 1
                        self.stats_graph.update_speed_label(speed_multiplier)
                        print("[NEAT] Turbo AUS (Visualisierung)")
                    elif cmd == "open_brain_viewer":
                        num_in = self.sim_config.sensor_ray_count * 5 + 1
                        viewer = BrainViewerWindow(self.hall, num_inputs=num_in, num_outputs=2)
                        viewer.start()
                    elif cmd == "toggle_sensors":
                        show_sensors = not show_sensors
                    elif cmd == "toggle_radio":
                        show_radio = not show_radio

            # Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return (False, show_sensors, show_radio, training_mode, speed_multiplier)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return (False, show_sensors, show_radio, training_mode, speed_multiplier)
                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    elif event.key == pygame.K_f:
                        show_radio = not show_radio
                    elif event.key == pygame.K_t:
                        training_mode = not training_mode
                        print(f"[NEAT] Modus: {'Training' if training_mode else 'Visualisierung'}")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                        speed_multiplier = min(100, speed_multiplier + 1)
                        if self.stats_graph:
                            self.stats_graph.update_speed_label(speed_multiplier)
                        print(f"[NEAT] Speed: {speed_multiplier}x")
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        speed_multiplier = max(1, speed_multiplier - 1)
                        if self.stats_graph:
                            self.stats_graph.update_speed_label(speed_multiplier)
                        print(f"[NEAT] Speed: {speed_multiplier}x")

            # Simulation-Steps
            if training_mode and collector_batch_net and hunter_batch_net:
                # FULL CUDA GPU BYPASS
                from core.cuda_sim import CudaSimulation
                sim = CudaSimulation(collectors, hunters, world.batteries, world.walls, self.sim_config, collector_batch_net, hunter_batch_net, obstacles=world.obstacles)
                c_fits, h_fits, alive_arr, bats_active, kills, bats, indiv_kills, indiv_bats, bat_x, bat_y, bat_timer = sim.run_generation(total_frames - current_step)
                
                for i, c in enumerate(collectors):
                    collector_genome_list[i].fitness += float(c_fits[i])
                    c.alive = bool(alive_arr[i])
                    c.batteries_collected += int(indiv_bats[i])
                for i, h in enumerate(hunters):
                    hunter_genome_list[i].fitness += float(h_fits[i])
                    h.alive = bool(alive_arr[len(collectors) + i])
                    h.kills += int(indiv_kills[len(collectors) + i])
                
                # Update world state so rendering (if any) looks okay
                for i, b in enumerate(world.batteries):
                    b.active = bool(bats_active[i])
                    b.x = float(bat_x[i])
                    b.y = float(bat_y[i])
                    b.respawn_timer = int(bat_timer[i])
                    
                current_step = total_frames
                break  # Skip the CPU loop entirely!

            steps = speed_multiplier if not training_mode else total_frames - current_step
            steps = min(steps, total_frames - current_step)

            # Performance-Caches (ausserhalb der Schleife erstellen)
            collector_sensor_range = self.sim_config.collector_sensor_ray_length
            hunter_sensor_range = self.sim_config.hunter_sensor_ray_length
            walls = world.walls
            obstacles = world.obstacles
            obstacles_tuples = [(float(obs.left), float(obs.top), float(obs.right), float(obs.bottom)) for obs in obstacles]
            w_width = world.width
            w_height = world.height
            idle_penalty = self.sim_config.fitness_idle_penalty
            survival_bonus = self.sim_config.fitness_survival_bonus
            battery_bonus = self.sim_config.fitness_battery_collected
            kill_bonus = self.sim_config.fitness_hunter_kill
            eaten_penalty = self.sim_config.fitness_eaten_penalty
            energy_refill = self.sim_config.energy_start
            bat_proximity_bonus = self.sim_config.fitness_battery_proximity
            hunter_danger_penalty = max(self.sim_config.fitness_hunter_danger, 0.15)
            hunter_approach_penalty = max(
                getattr(self.sim_config, 'fitness_hunter_approach_penalty', 15.0),
                15.0)
            danger_zone = self.sim_config.fitness_danger_zone
            danger_zone_sq = danger_zone * danger_zone  # Vermeidet sqrt!

            # Pre-built Index-Lookup: Collector -> Genome-Index (O(1) statt O(N))
            collector_idx_map = {id(c): i for i, c in enumerate(collectors)}

            for step_i in range(steps):
                current_step += 1

                # Spatial Grid aktualisieren
                grid.clear()
                for r in all_robots:
                    if r.alive:
                        grid.insert(r)
                for b in world.batteries:
                    if b.active:
                        grid.insert(b)

                # --- Radio-Signale verarbeiten (Deaktiviert zur Rauschreduzierung) ---

                # Sensoren nur jeden 2. Frame im Turbo aktualisieren (fuer bessere Reaktionszeiten)
                # Positionsfehler ist minimal: Roboter bewegen sich nur ~3px/Frame
                update_sensors_flag = (not training_mode) or (step_i % 2 == 0)

                # --- Sensoren: Parallel (Turbo) oder Einzeln (Normal) ---------
                if update_sensors_flag:
                    if training_mode:
                        # TURBO: Parallele Batch-Berechnung auf allen CPU-Kernen
                        cast_rays_batch(collectors, walls, world.batteries,
                                        all_robots, self.sim_config)
                        cast_rays_batch(hunters, walls, world.batteries,
                                        all_robots, self.sim_config)
                    else:
                        # NORMAL: Spatial-Grid-basierte Einzelberechnung
                        for c in collectors:
                            if not c.alive:
                                continue
                            nearby = grid.get_nearby(c, collector_sensor_range)
                            nearby_bats = [e for e in nearby if e.entity_type == ENTITY_BATTERY]
                            nearby_robs = [e for e in nearby
                                           if e.entity_type != ENTITY_BATTERY and e is not c]
                            c.update_sensors(walls, nearby_bats, [c] + nearby_robs)
                        for h in hunters:
                            if not h.alive:
                                continue
                            nearby = grid.get_nearby(h, hunter_sensor_range)
                            nearby_bats = [e for e in nearby if e.entity_type == ENTITY_BATTERY]
                            nearby_robs = [e for e in nearby
                                           if e.entity_type != ENTITY_BATTERY and e is not h]
                            h.update_sensors(walls, nearby_bats, [h] + nearby_robs)

                # --- Sammler Update -------------------------------------------
                import numpy as np
                collector_inputs_batch = None
                if collector_batch_net:
                    collector_inputs_batch = np.zeros((len(collectors), collector_batch_net.n_inputs), dtype=np.float64)
                    
                has_c_data = False
                
                for i, c in enumerate(collectors):
                    if not c.alive:
                        continue

                    # NEU: Direkte Gefahr-Inputs (Nearest Hunter) vorbereiten
                    search_radius = self.sim_config.collector_sensor_ray_length
                    nearby_hunters_for_input = grid.get_nearby(c, search_radius)
                    nearest_dist = search_radius
                    nearest_hunter = None
                    for h_ in nearby_hunters_for_input:
                        if h_.entity_type == ENTITY_HUNTER and h_.alive:
                            dx = h_.x - c.x
                            dy = h_.y - c.y
                            dsq = dx * dx + dy * dy
                            if dsq < nearest_dist * nearest_dist:
                                nearest_dist = dsq ** 0.5
                                nearest_hunter = h_
                                
                    dx_norm = 0.0
                    dy_norm = 0.0
                    if nearest_hunter:
                        dist_norm = 1.0 - (nearest_dist / search_radius)
                        angle_to_hunter = math.atan2(nearest_hunter.y - c.y, nearest_hunter.x - c.x)
                        rel_angle = angle_to_hunter - c.angle
                        dx_norm = math.cos(rel_angle) * dist_norm
                        dy_norm = math.sin(rel_angle) * dist_norm
                        
                    # Neuronales Netz
                    if not c.sensor_data:  # Noch keine Sensordaten -> Skip
                        continue
                        
                    has_c_data = True
                    inputs = c.get_sensor_inputs()
                    inputs.append(dx_norm)
                    inputs.append(dy_norm)
                    
                    if collector_batch_net:
                        collector_inputs_batch[i] = inputs
                    else:
                        output = collector_nets[i].activate(inputs)
                        c.move(output[0], output[1])

                if collector_batch_net and has_c_data:
                    c_batch_out = collector_batch_net.activate_batch(collector_inputs_batch)
                    for i, c in enumerate(collectors):
                        if c.alive and c.sensor_data:
                            c.move(c_batch_out[i, 0], c_batch_out[i, 1])

                for i, c in enumerate(collectors):
                    if not c.alive or not c.sensor_data:
                        continue
                        
                    c.clamp_to_bounds(w_width, w_height)
                    c.check_obstacle_collision_fast(obstacles_tuples)

                    # Energie + Fitness
                    c.drain_energy()
                    
                    # Anti-Kreisdreher-Logik: Vorwaertsgeschwindigkeit berechnen
                    # v = (motor_left + motor_right) / 2
                    forward_speed = (c.motor_left + c.motor_right) / 2.0
                    if forward_speed < 0.1:  # Steht still, fährt rückwärts oder dreht sich nur im Kreis
                        collector_genome_list[i].fitness += idle_penalty
                        c.fit_idle += idle_penalty
                        
                    collector_genome_list[i].fitness += survival_bonus
                    c.fit_surv += survival_bonus

                    # ── Proximity-Bonus & Batterie Sammeln (FUSED fuer Performance) ──
                    nearby_bats = grid.get_nearby(c, collector_sensor_range)
                    nearest_bat_dist_sq = collector_sensor_range * collector_sensor_range
                    collect_dist_sq = (c.radius + 15) * (c.radius + 15)  # 15 = max battery radius
                    
                    for nb in nearby_bats:
                        if nb.entity_type == ENTITY_BATTERY and nb.active:
                            dx = c.x - nb.x
                            dy = c.y - nb.y
                            dsq = dx * dx + dy * dy
                            
                            # Kollisionspruefung
                            if dsq < collect_dist_sq:
                                nb.collect()
                                c.add_energy(self.sim_config.battery_energy)
                                c.batteries_collected += 1
                                collector_genome_list[i].fitness += battery_bonus
                                c.fit_battery += battery_bonus
                                # Keine proximity fuer eingesammelte
                                continue
                                
                            if dsq < nearest_bat_dist_sq:
                                nearest_bat_dist_sq = dsq
                                
                    if nearest_bat_dist_sq < collector_sensor_range * collector_sensor_range:
                        d = nearest_bat_dist_sq ** 0.5
                        prox_bonus = bat_proximity_bonus * (1.0 - d / collector_sensor_range)
                        collector_genome_list[i].fitness += prox_bonus
                        c.fit_prox += prox_bonus

                    # ── Proximity-Malus: Weg vom Jaeger (Spatial-Grid!) ──
                    nearby_hunters = grid.get_nearby(c, danger_zone)
                    
                    # Track previous distances to hunters to calculate escape bonus
                    if not hasattr(c, 'prev_hunter_dists'):
                        c.prev_hunter_dists = {}
                    
                    current_hunter_dists = {}
                    nearest_hunter_dist = None
                    
                    for nh in nearby_hunters:
                        if nh.entity_type == ENTITY_HUNTER and nh.alive:
                            dx = c.x - nh.x
                            dy = c.y - nh.y
                            dsq = dx * dx + dy * dy
                            if dsq < danger_zone_sq:
                                d = dsq ** 0.5
                                collector_danger_frames += 1
                                if nearest_hunter_dist is None or d < nearest_hunter_dist:
                                    nearest_hunter_dist = d
                                current_hunter_dists[id(nh)] = d
                                danger_penalty = hunter_danger_penalty * (1.0 - d / danger_zone)
                                if danger_penalty > 0.0:
                                    collector_genome_list[i].fitness -= danger_penalty
                                    c.fit_hunter_pen -= danger_penalty
                                
                                # Escape Bonus: If we moved further away from this hunter since last frame
                                if id(nh) in c.prev_hunter_dists:
                                    prev_d = c.prev_hunter_dists[id(nh)]
                                    dist_delta = d - prev_d
                                    if dist_delta > 0:  # Moving away!
                                        collector_escape_events += 1
                                        # Bonus proportional to how fast we are escaping (max collector_speed)
                                        # DRSTISCH ERHÖHTER FLUCHTBONUS: 15.0 statt 5.0, damit Fliehen extrem lukrativ wird!
                                        escape_bonus = (dist_delta / self.sim_config.collector_speed) * 15.0
                                        collector_genome_list[i].fitness += escape_bonus
                                        c.fit_escape += escape_bonus
                                    elif dist_delta < 0:
                                        collector_approach_events += 1
                                        approach_penalty = ((-dist_delta / self.sim_config.collector_speed) *
                                                            hunter_approach_penalty)
                                        collector_genome_list[i].fitness -= approach_penalty
                                        c.fit_approach -= approach_penalty
                                    else:
                                        collector_neutral_events += 1
                                        
                    if nearest_hunter_dist is not None:
                        nearest_hunter_dist_sum += nearest_hunter_dist
                        nearest_hunter_dist_count += 1

                    # Update previous distances for next frame
                    c.prev_hunter_dists = current_hunter_dists

                # --- Jaeger Update --------------------------------------------
                hunter_inputs_batch = None
                if hunter_batch_net:
                    hunter_inputs_batch = np.zeros((len(hunters), hunter_batch_net.n_inputs), dtype=np.float64)
                    
                has_h_data = False
                
                for i, h in enumerate(hunters):
                    if not h.alive:
                        continue

                    # NEU: Direkte Beute-Inputs (Nearest Collector) vorbereiten
                    search_radius = self.sim_config.hunter_sensor_ray_length
                    nearby_collectors_for_input = grid.get_nearby(h, search_radius)
                    nearest_dist = search_radius
                    nearest_collector = None
                    for prey in nearby_collectors_for_input:
                        if prey.entity_type == ENTITY_COLLECTOR and prey.alive:
                            dx = prey.x - h.x
                            dy = prey.y - h.y
                            dsq = dx * dx + dy * dy
                            if dsq < nearest_dist * nearest_dist:
                                nearest_dist = dsq ** 0.5
                                nearest_collector = prey
                                
                    dx_norm = 0.0
                    dy_norm = 0.0
                    if nearest_collector:
                        dist_norm = 1.0 - (nearest_dist / search_radius)
                        angle_to_collector = math.atan2(nearest_collector.y - h.y, nearest_collector.x - h.x)
                        rel_angle = angle_to_collector - h.angle
                        dx_norm = math.cos(rel_angle) * dist_norm
                        dy_norm = math.sin(rel_angle) * dist_norm
                        
                    # Neuronales Netz
                    if not h.sensor_data:  # Noch keine Sensordaten -> Skip
                        continue
                        
                    has_h_data = True
                    inputs = h.get_sensor_inputs()
                    inputs.append(dx_norm)
                    inputs.append(dy_norm)
                    
                    if hunter_batch_net:
                        hunter_inputs_batch[i] = inputs
                    else:
                        output = hunter_nets[i].activate(inputs)
                        h.move(output[0], output[1])

                if hunter_batch_net and has_h_data:
                    h_batch_out = hunter_batch_net.activate_batch(hunter_inputs_batch)
                    for i, h in enumerate(hunters):
                        if h.alive and h.sensor_data:
                            h.move(h_batch_out[i, 0], h_batch_out[i, 1])

                for i, h in enumerate(hunters):
                    if not h.alive or not h.sensor_data:
                        continue
                        
                    h.clamp_to_bounds(w_width, w_height)
                    h.check_obstacle_collision_fast(obstacles_tuples)

                    # Energie + Fitness
                    h.drain_energy()
                    
                    forward_speed = (h.motor_left + h.motor_right) / 2.0
                    if forward_speed < 0.1:
                        hunter_genome_list[i].fitness += idle_penalty
                    hunter_genome_list[i].fitness += survival_bonus

                    # ── Proximity-Bonus: Richtung Sammler ──
                    # Jaeger bekommen Bonus wenn nahe an Sammlern
                    nearby_prey = grid.get_nearby(h, hunter_sensor_range)
                    nearest_prey_dist_sq = hunter_sensor_range * hunter_sensor_range
                    for np_ in nearby_prey:
                        if np_.entity_type == ENTITY_COLLECTOR and np_.alive:
                            dx = h.x - np_.x
                            dy = h.y - np_.y
                            dsq = dx * dx + dy * dy
                            if dsq < nearest_prey_dist_sq:
                                nearest_prey_dist_sq = dsq
                    if nearest_prey_dist_sq < hunter_sensor_range * hunter_sensor_range:
                        d = nearest_prey_dist_sq ** 0.5
                        hunter_genome_list[i].fitness += 0.1 * (1.0 - d / hunter_sensor_range)

                    # Sammler fressen (Kollisionspruefung)
                    nearby_collectors = [e for e in
                                         grid.get_nearby(h, h.radius + 20)
                                         if e.entity_type == ENTITY_COLLECTOR and e.alive]
                    for prey in nearby_collectors:
                        dx = h.x - prey.x
                        dy = h.y - prey.y
                        dist_sq = dx * dx + dy * dy
                        catch_dist = h.radius + prey.radius
                        if dist_sq < catch_dist * catch_dist:
                            # Sammler gefressen!
                            prey.alive = False
                            h.kills += 1
                            h.add_energy(energy_refill)
                            hunter_genome_list[i].fitness += kill_bonus
                            # Sammler bekommt Strafe
                            prey_idx = collector_idx_map[id(prey)]
                            collector_genome_list[prey_idx].fitness += eaten_penalty
                            prey.fit_death += eaten_penalty

                # Welt updaten
                world.update()

            # --- Rendering ----------------------------------------------------
            if not training_mode and screen is not None:
                world.draw(screen)
                for r in all_robots:
                    draw_sensors(screen, r, show_sensors)
                    if show_radio and getattr(r, 'alive', False):
                        # Sigmoid ruht bei 0.5. Nur Werte darüber sind aktives "Schreien".
                        intensity = (getattr(r, 'radio_out', 0.0) - 0.5) * 2.0
                        if intensity > 0.1:
                            radius = int(r.config.radio_range)
                            alpha_fill = int(intensity * 25)      # Sehr dezente Füllung
                            alpha_outline = int(intensity * 180)  # Kräftigere Ränder
                            
                            # Padding hinzufügen, damit dicke Ränder nicht abgeschnitten werden
                            pad = 4
                            surf = pygame.Surface((radius * 2 + pad * 2, radius * 2 + pad * 2), pygame.SRCALPHA)
                            
                            center = (radius + pad, radius + pad)
                            
                            # Sanfte Hintergrundfüllung
                            pygame.draw.circle(surf, (100, 200, 255, alpha_fill), center, radius)
                            
                            # Funk-Wellen (konzentrische Kreise)
                            pygame.draw.circle(surf, (100, 200, 255, alpha_outline), center, radius, 2)
                            pygame.draw.circle(surf, (100, 200, 255, int(alpha_outline * 0.6)), center, int(radius * 0.66), 1)
                            pygame.draw.circle(surf, (100, 200, 255, int(alpha_outline * 0.3)), center, int(radius * 0.33), 1)
                            
                            screen.blit(surf, (int(r.x) - radius - pad, int(r.y) - radius - pad))
                    r.draw(screen)
                self._draw_hud(screen, clock, collectors, hunters, world,
                               current_step, total_frames)
                pygame.display.flip()
                clock.tick(FPS_TARGET)

        # --- Fitness finalisieren ---------------------------------------------
        # Nur echte trainierende Genome fuer die Statistik werten
        c_fits = [g.fitness for g in collector_genome_list if not getattr(g, 'is_injected', False)]
        h_fits = [g.fitness for g in hunter_genome_list]

        self.avg_collector_fitness = sum(c_fits) / len(c_fits) if c_fits else 0
        self.best_collector_fitness = max(c_fits) if c_fits else 0
        self.avg_hunter_fitness = sum(h_fits) / len(h_fits) if h_fits else 0
        self.best_hunter_fitness = max(h_fits) if h_fits else 0

        # Breakdown der Fitness fuer Debugging (Nur echte trainierende Sammler)
        real_collectors = [c for c in collectors if not c.is_injected]
        avg_fit_bat = sum(c.fit_battery for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_prox = sum(c.fit_prox for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_surv = sum(c.fit_surv for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_pen = sum(c.fit_hunter_pen for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_escape = sum(c.fit_escape for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_approach = sum(c.fit_approach for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_death = sum(c.fit_death for c in real_collectors) / len(real_collectors) if real_collectors else 0
        avg_fit_idle = sum(c.fit_idle for c in real_collectors) / len(real_collectors) if real_collectors else 0
        
        print(f"[FITNESS BREAKDOWN AVG] Bat: {avg_fit_bat:.1f} | Prox: {avg_fit_prox:.1f} | Surv: {avg_fit_surv:.1f} | Pen: {avg_fit_pen:.1f} | Escape: {avg_fit_escape:.1f} | Approach: {avg_fit_approach:.1f} | Death: {avg_fit_death:.1f} | Idle: {avg_fit_idle:.1f}")

        total_kills = sum(h.kills for h in hunters)
        total_bats = sum(c.batteries_collected for c in collectors)
        alive_collectors = sum(1 for c in collectors if c.alive)

        print(f"[NEAT] Gen {self.generation}: "
              f"Sammler best={self.best_collector_fitness:.1f} avg={self.avg_collector_fitness:.1f} | "
              f"Jaeger best={self.best_hunter_fitness:.1f} avg={self.avg_hunter_fitness:.1f} | "
              f"Kills={total_kills} Batt={total_bats} Alive={alive_collectors}")

        gen_duration = time.time() - gen_start_time
        movement_events = (collector_escape_events + collector_approach_events +
                           collector_neutral_events)
        escape_ratio = (collector_escape_events / movement_events
                        if movement_events else 0.0)
        approach_ratio = (collector_approach_events / movement_events
                          if movement_events else 0.0)
        avg_nearest_hunter_dist = (nearest_hunter_dist_sum / nearest_hunter_dist_count
                                   if nearest_hunter_dist_count else 0.0)
        injected_collectors = [c for c in collectors if c.is_injected]
        real_bats = sum(c.batteries_collected for c in real_collectors)

        _append_csv_row(self.generation_log_path, GENERATION_LOG_FIELDS, {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generation": self.generation,
            "duration_sec": f"{gen_duration:.4f}",
            "training_mode": int(training_mode),
            "speed_multiplier": speed_multiplier,
            "collectors_total": len(collectors),
            "collectors_real": len(real_collectors),
            "collectors_injected": len(injected_collectors),
            "collectors_alive": alive_collectors,
            "hunters_total": len(hunters),
            "hunters_alive": sum(1 for h in hunters if h.alive),
            "batteries_collected": total_bats,
            "batteries_collected_real": real_bats,
            "kills": total_kills,
            "best_collector_fitness": f"{self.best_collector_fitness:.6f}",
            "avg_collector_fitness": f"{self.avg_collector_fitness:.6f}",
            "best_hunter_fitness": f"{self.best_hunter_fitness:.6f}",
            "avg_hunter_fitness": f"{self.avg_hunter_fitness:.6f}",
            "avg_fit_battery": f"{avg_fit_bat:.6f}",
            "avg_fit_proximity": f"{avg_fit_prox:.6f}",
            "avg_fit_survival": f"{avg_fit_surv:.6f}",
            "avg_fit_hunter_penalty": f"{avg_fit_pen:.6f}",
            "avg_fit_escape": f"{avg_fit_escape:.6f}",
            "avg_fit_approach": f"{avg_fit_approach:.6f}",
            "avg_fit_death": f"{avg_fit_death:.6f}",
            "avg_fit_idle": f"{avg_fit_idle:.6f}",
            "collector_sees_hunter_frames": collector_sees_hunter_frames,
            "collector_danger_frames": collector_danger_frames,
            "collector_escape_events": collector_escape_events,
            "collector_approach_events": collector_approach_events,
            "collector_neutral_events": collector_neutral_events,
            "escape_ratio": f"{escape_ratio:.6f}",
            "approach_ratio": f"{approach_ratio:.6f}",
            "avg_nearest_hunter_dist": f"{avg_nearest_hunter_dist:.6f}",
        })

        # --- Hall of Fame (beste Sammler) -------------------------------------
        if c_fits:
            # Den besten Sammler finden, der KEIN Gast ist
            best_c_idx = -1
            best_c_val = -999999
            for i, c in enumerate(collectors):
                if not getattr(c.genome, 'is_guest', False):
                    if c_fits[i] > best_c_val:
                        best_c_val = c_fits[i]
                        best_c_idx = i

            if best_c_idx != -1:
                hof_entry = self.hall.try_add(
                    genome=collector_genome_list[best_c_idx],
                    fitness=best_c_val,
                    generation=self.generation,
                    batteries=collectors[best_c_idx].batteries_collected,
                )
                if hof_entry and self.commentary:
                    self.commentary.post_event("hall_of_fame",
                        name=hof_entry.name, fitness=hof_entry.fitness)

        # Hall of Fame Fenster aktualisieren
        if self.hof_window:
            self.hof_window.notify_update()

        # Commentary: Generation End
        if self.commentary:
            self.commentary.post_event("gen_end",
                gen=self.generation,
                best_c=self.best_collector_fitness,
                best_h=self.best_hunter_fitness,
                kills=total_kills,
                alive=alive_collectors,
                batteries=total_bats)

        # Stats-Graph aktualisieren
        if self.stats_graph:
            gen_duration = time.time() - gen_start_time
            self.stats_graph.add_generation(
                gen=self.generation,
                best_c=self.best_collector_fitness,
                avg_c=self.avg_collector_fitness,
                best_h=self.best_hunter_fitness,
                avg_h=self.avg_hunter_fitness,
                kills=total_kills,
                alive=alive_collectors,
                total_collectors=len(collectors),
                batteries=total_bats,
                gen_time=gen_duration,
                escape_ratio=escape_ratio)

        # --- Naechste Generation ----------------------------------------------
        # Sammler
        self.collector_pop.species.speciate(
            self.collector_config, self.collector_pop.population,
            self.collector_pop.generation)
        try:
            self.collector_pop.population = self.collector_pop.reproduction.reproduce(
                self.collector_config, self.collector_pop.species,
                self.collector_config.pop_size, self.collector_pop.generation)
        except AssertionError:
            # Bekannter NEAT-Bug: Node-ID-Konflikt bei get_new_node_key
            # Fix: Node-Indexer zuruecksetzen
            print("[NEAT] WARNUNG: Node-ID-Konflikt bei Sammlern - setze Indexer zurueck")
            genome_indexer = self.collector_pop.config.genome_config
            if hasattr(genome_indexer, 'node_indexer'):
                max_id = max(max(g.nodes.keys()) for g in self.collector_pop.population.values()) + 1
                genome_indexer.node_indexer = max_id
            self.collector_pop.population = self.collector_pop.reproduction.reproduce(
                self.collector_config, self.collector_pop.species,
                self.collector_config.pop_size, self.collector_pop.generation)
        self.collector_pop.generation += 1

        # Jaeger
        self.hunter_pop.species.speciate(
            self.hunter_config, self.hunter_pop.population,
            self.hunter_pop.generation)
        try:
            self.hunter_pop.population = self.hunter_pop.reproduction.reproduce(
                self.hunter_config, self.hunter_pop.species,
                self.hunter_config.pop_size, self.hunter_pop.generation)
        except AssertionError:
            print("[NEAT] WARNUNG: Node-ID-Konflikt bei Jaegern - setze Indexer zurueck")
            genome_indexer = self.hunter_pop.config.genome_config
            if hasattr(genome_indexer, 'node_indexer'):
                max_id = max(max(g.nodes.keys()) for g in self.hunter_pop.population.values()) + 1
                genome_indexer.node_indexer = max_id
            self.hunter_pop.population = self.hunter_pop.reproduction.reproduce(
                self.hunter_config, self.hunter_pop.species,
                self.hunter_config.pop_size, self.hunter_pop.generation)
        self.hunter_pop.generation += 1

        self.generation += 1
        return (True, show_sensors, show_radio, training_mode, speed_multiplier)

    def _draw_hud(self, screen: pygame.Surface, clock: pygame.time.Clock,
                  collectors: list[Collector], hunters: list[Hunter],
                  world: World, frame: int, total_frames: int) -> None:
        """Zeichnet das erweiterte HUD mit Co-Evolution-Statistiken und Hall of Fame."""
        hud_width = 600
        hud_height = 400
        hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_surf.fill(COLOR_HUD_BG)
        screen.blit(hud_surf, (8, 8))

        x = 16
        y = 12
        h = 36
        font = pygame.font.SysFont("Segoe UI", 30, bold=True)
        font_s = pygame.font.SysFont("Segoe UI", 24)

        # FPS + Gen
        fps = clock.get_fps()
        fps_color = COLOR_HUD_ACCENT if fps >= 30 else COLOR_HUD_WARN
        screen.blit(font.render(f"FPS: {fps:.0f}", True, fps_color), (x, y))
        screen.blit(font_s.render(
            f"Gen: {self.generation}  Frame: {frame}/{total_frames}",
            True, COLOR_HUD_GEN), (x + 90, y + 2))

        # Sammler-Stats
        alive_c = sum(1 for c in collectors if c.alive)
        screen.blit(font.render("Sammler", True, COLOR_HUD_COLLECTOR), (x, y + h * 1.5))
        screen.blit(font_s.render(
            f"Aktiv: {alive_c}/{len(collectors)}  "
            f"Best: {self.best_collector_fitness:.0f}  "
            f"Avg: {self.avg_collector_fitness:.0f}",
            True, COLOR_HUD_TEXT), (x, y + h * 2.5))
        total_bats = sum(c.batteries_collected for c in collectors)
        screen.blit(font_s.render(
            f"Batterien gesammelt: {total_bats}",
            True, COLOR_HUD_TEXT), (x, y + h * 3.3))

        # Jaeger-Stats
        alive_h = sum(1 for hu in hunters if hu.alive)
        screen.blit(font.render("Jaeger", True, COLOR_HUD_HUNTER), (x, y + h * 4.5))
        screen.blit(font_s.render(
            f"Aktiv: {alive_h}/{len(hunters)}  "
            f"Best: {self.best_hunter_fitness:.0f}  "
            f"Avg: {self.avg_hunter_fitness:.0f}",
            True, COLOR_HUD_TEXT), (x, y + h * 5.5))
        total_kills = sum(hu.kills for hu in hunters)
        screen.blit(font_s.render(
            f"Kills: {total_kills}",
            True, COLOR_HUD_TEXT), (x, y + h * 6.3))

        # Batterien
        active_bat = sum(1 for b in world.batteries if b.active)
        screen.blit(font_s.render(
            f"Batterien aktiv: {active_bat}/{self.sim_config.battery_count}  |  "
            f"Hall of Fame: {len(self.hall.entries)}",
            True, COLOR_HUD_TEXT), (x, y + h * 7.5))

        # Steuerungs-Hinweis
        screen.blit(font_s.render(
            "T=Training  S=Sensoren  +/-=Speed  ESC=Ende",
            True, COLOR_HUD_TEXT), (x, y + h * 8.8))



def run_neat_training(config: SimConfig, screen: pygame.Surface,
                      clock: pygame.time.Clock, hall: HallOfFame,
                      injected_entries: list[HallOfFameEntry] | None = None,
                      mode: str = 'train') -> None:
    """Startet das Co-Evolution NEAT-Training mit Live-Kommentar.

    Args:
        config: Simulationskonfiguration.
        screen: Pygame-Screen.
        clock: Pygame-Clock.
        hall: Hall of Fame Instanz.
        injected_entries: Ausgewaehlte HoF-Eintraege zum Injizieren.
        mode: 'train' (Headless, Turbo) oder 'test' (Visual, Normal speed).
    """
    # Commentary + Graph starten
    commentary = CommentaryWindow(offset_x=config.window_width)
    commentary.start()

    stats_graph = StatsGraphWindow(offset_x=config.window_width)
    stats_graph.start()

    hof_window = HallOfFameWindow(hall, offset_x=config.window_width)
    hof_window.start()

    manager = CoEvolutionManager(config, hall, commentary=commentary,
                                 stats_graph=stats_graph,
                                 hof_window=hof_window,
                                 injected_entries=injected_entries)

    show_sensors = False
    show_radio = False
    training_mode = (mode == 'train')
    speed_multiplier = 500 if training_mode else 1

    print(f"[NEAT] Starte Co-Evolution (Unendlich)")
    print(f"[NEAT] Tasten: T=Training, S=Sensoren, F=Funk, +/-=Speed, ESC=Beenden")

    gen = 0
    while True:
        result = manager.run_generation(
            screen, clock, show_sensors, show_radio, training_mode, speed_multiplier)
        cont, show_sensors, show_radio, training_mode, speed_multiplier = result
        if not cont:
            print(f"[NEAT] Training abgebrochen in Generation {gen}")
            commentary.post_event("custom",
                text=f"Training nach {gen} Generationen beendet. Tschuess!",
                tag="info")
            break
        gen += 1

    print(f"[NEAT] Training beendet.")
    print(f"  Sammler beste Fitness: {manager.best_collector_fitness:.1f}")
    print(f"  Jaeger beste Fitness:  {manager.best_hunter_fitness:.1f}")
    print(f"  Hall of Fame: {len(hall.entries)} Eintraege")

    # Halte das Pygame-Fenster und die Graphen offen, bis der Nutzer sie schließt
    if screen is not None:
        font = pygame.font.SysFont("Segoe UI", 36, bold=True)
        text = font.render("Training beendet. Schließe das Graphen-Fenster zum Beenden.", True, (255, 200, 0))
        text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        
        # Hintergrund leicht abdunkeln
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        screen.blit(text, text_rect)
        pygame.display.flip()

    print("[NEAT] Warte auf das Schließen des Graphen-Fensters...")
    
    # Warte bis das Graphen-Fenster manuell geschlossen wird
    while stats_graph.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stats_graph.stop()
        clock.tick(30)

    commentary.stop()
    stats_graph.stop()
    hof_window.stop()
