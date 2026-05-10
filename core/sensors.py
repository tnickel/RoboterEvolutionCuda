"""
sensors.py – Raycasting-Sensorsystem für das Neuro-Ökosystem.

Implementiert einen 2D-Raycasting-Algorithmus, der für jeden Roboter
mehrere Sichtstrahlen berechnet. Jeder Strahl liefert die normalisierte
Distanz zum nächsten Objekt und dessen Typ.

Performance: Die Kernberechnungen (_ray_segment_intersection,
_ray_circle_intersection, _cast_rays_numba) sind mit Numba JIT-kompiliert
und laufen nahezu mit C-Geschwindigkeit.

Objekttypen:
    0 = nichts getroffen
    1 = Wand (inkl. Hindernisse)
    2 = Batterie
    3 = Sammler (Collector)
    4 = Jäger (Hunter)
"""

import math
import numpy as np
import pygame
from numba import njit, prange
from config.config_manager import SimConfig

# ─── Objekttyp-Konstanten ────────────────────────────────────────────────────────
OBJ_NONE = 0
OBJ_WALL = 1
OBJ_BATTERY = 2
OBJ_COLLECTOR = 3
OBJ_HUNTER = 4

# ─── Farb-Konstanten für Visualisierung ──────────────────────────────────────────
COLOR_RAY_MISS = (40, 180, 80, 150)     # Grün – kein Treffer
COLOR_RAY_WALL = (180, 60, 60, 200)     # Rot – Wand getroffen
COLOR_RAY_BATTERY = (220, 200, 50, 200) # Gelb – Batterie getroffen
COLOR_RAY_COLLECTOR = (50, 200, 150, 200)  # Cyan – Sammler getroffen
COLOR_RAY_HUNTER = (220, 50, 50, 200)   # Rot – Jäger getroffen
COLOR_HIT_POINT = (255, 255, 255)       # Weiß – Trefferpunkt

# Farb-Zuordnung nach Objekttyp
RAY_COLORS = {
    OBJ_NONE: COLOR_RAY_MISS,
    OBJ_WALL: COLOR_RAY_WALL,
    OBJ_BATTERY: COLOR_RAY_BATTERY,
    OBJ_COLLECTOR: COLOR_RAY_COLLECTOR,
    OBJ_HUNTER: COLOR_RAY_HUNTER,
}


# ══════════════════════════════════════════════════════════════════════════════════
# Numba JIT-kompilierte Kernfunktionen (laufen als nativer Maschinencode)
# ══════════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _ray_segment_intersection_nb(ox, oy, dx, dy, x1, y1, x2, y2):
    """Schnittpunkt Strahl <-> Liniensegment (Numba-optimiert).

    Returns:
        Distanz t zum Schnittpunkt, oder -1.0 wenn kein Schnitt.
    """
    sx = x2 - x1
    sy = y2 - y1

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-10:
        return -1.0

    t_num = (x1 - ox) * sy - (y1 - oy) * sx
    u_num = (x1 - ox) * dy - (y1 - oy) * dx

    t = t_num / denom
    u = u_num / denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t

    return -1.0


@njit(cache=True)
def _ray_circle_intersection_nb(ox, oy, dx, dy, cx, cy, cr):
    """Schnittpunkt Strahl <-> Kreis (Numba-optimiert).

    Returns:
        Kleinste positive Distanz t, oder -1.0 wenn kein Schnitt.
    """
    fx = ox - cx
    fy = oy - cy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - cr * cr

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:
        return -1.0

    sqrt_disc = math.sqrt(discriminant)
    inv_2a = 1.0 / (2.0 * a)

    t1 = (-b - sqrt_disc) * inv_2a
    t2 = (-b + sqrt_disc) * inv_2a

    if t1 >= 0.0:
        return t1
    if t2 >= 0.0:
        return t2

    return -1.0


@njit(cache=True)
def _cast_rays_numba(robot_x, robot_y, robot_angle,
                     ray_count, max_length, fov_rad,
                     walls_arr,
                     bat_x, bat_y, bat_r, n_bats,
                     rob_x, rob_y, rob_r, rob_type, n_robs):
    """Berechnet alle Sensor-Strahlen fuer einen Roboter (Numba-optimiert).

    Alle Eingaben sind primitive Typen oder NumPy-Arrays, damit Numba
    die gesamte Berechnung als nativen Maschinencode ausfuehren kann.

    Args:
        robot_x, robot_y: Position des Roboters
        robot_angle: Blickrichtung in Rad
        ray_count: Anzahl Strahlen
        max_length: Maximale Strahllaenge
        fov_rad: Sichtfeld in Rad
        walls_arr: Nx4 Array der Waende (x1,y1,x2,y2)
        bat_x, bat_y, bat_r: Arrays der Batterie-Positionen und Radien
        n_bats: Anzahl aktiver Batterien
        rob_x, rob_y, rob_r, rob_type: Arrays der Roboter-Daten
        n_robs: Anzahl nahegelegener Roboter

    Returns:
        result: ray_count x 4 Array (dist_norm, obj_type, hit_x, hit_y)
    """
    result = np.empty((ray_count, 4), dtype=np.float64)
    n_walls = walls_arr.shape[0]

    for i in range(ray_count):
        # Strahlwinkel berechnen
        if ray_count > 1:
            angle_offset = -fov_rad / 2.0 + (i / (ray_count - 1)) * fov_rad
        else:
            angle_offset = 0.0

        ray_angle = robot_angle + angle_offset
        ray_dx = math.cos(ray_angle)
        ray_dy = math.sin(ray_angle)

        closest_dist = max_length
        closest_type = 0.0  # OBJ_NONE

        # Waende pruefen
        for w in range(n_walls):
            t = _ray_segment_intersection_nb(
                robot_x, robot_y, ray_dx, ray_dy,
                walls_arr[w, 0], walls_arr[w, 1],
                walls_arr[w, 2], walls_arr[w, 3])
            if t >= 0.0 and t < closest_dist:
                closest_dist = t
                closest_type = 1.0  # OBJ_WALL

        # Batterien pruefen
        for b in range(n_bats):
            t = _ray_circle_intersection_nb(
                robot_x, robot_y, ray_dx, ray_dy,
                bat_x[b], bat_y[b], bat_r[b])
            if t >= 0.0 and t < closest_dist:
                closest_dist = t
                closest_type = 2.0  # OBJ_BATTERY

        # Andere Roboter pruefen
        for r in range(n_robs):
            t = _ray_circle_intersection_nb(
                robot_x, robot_y, ray_dx, ray_dy,
                rob_x[r], rob_y[r], rob_r[r])
            if t >= 0.0 and t < closest_dist:
                closest_dist = t
                closest_type = rob_type[r]  # OBJ_COLLECTOR oder OBJ_HUNTER

        # Ergebnis speichern
        dist_norm = closest_dist / max_length
        hit_x = robot_x + ray_dx * closest_dist
        hit_y = robot_y + ray_dy * closest_dist

        result[i, 0] = dist_norm
        result[i, 1] = closest_type
        result[i, 2] = hit_x
        result[i, 3] = hit_y

    return result


@njit(parallel=True, cache=True)
def _cast_rays_batch_numba(n_robots,
                           robot_x_arr, robot_y_arr, robot_angle_arr,
                           robot_rob_idx,
                           ray_count, max_length, fov_rad,
                           walls_arr,
                           bat_x, bat_y, bat_r, n_bats,
                           rob_x, rob_y, rob_r, rob_type, n_robs):
    """Berechnet Sensor-Strahlen fuer ALLE Roboter parallel (multi-core).

    Nutzt Numba prange fuer echte Thread-Parallelisierung ueber alle
    CPU-Kerne. Jeder Roboter wird unabhaengig auf einem eigenen Core
    berechnet.

    Args:
        n_robots: Anzahl zu berechnender Roboter
        robot_x_arr, robot_y_arr, robot_angle_arr: Arrays der Roboter-Positionen
        robot_rob_idx: Index jedes Roboters im rob_x/rob_y-Array (zum Ausschluss)
        (restliche Parameter identisch zu _cast_rays_numba)

    Returns:
        result: n_robots x ray_count x 4 Array (dist_norm, obj_type, hit_x, hit_y)
    """
    result = np.empty((n_robots, ray_count, 4), dtype=np.float64)
    n_walls = walls_arr.shape[0]

    for robot_idx in prange(n_robots):
        rx = robot_x_arr[robot_idx]
        ry = robot_y_arr[robot_idx]
        ra = robot_angle_arr[robot_idx]
        self_rob_idx = robot_rob_idx[robot_idx]

        for i in range(ray_count):
            if ray_count > 1:
                angle_offset = -fov_rad / 2.0 + (i / (ray_count - 1)) * fov_rad
            else:
                angle_offset = 0.0

            ray_angle = ra + angle_offset
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            closest_dist = max_length
            closest_type = 0.0

            # Waende pruefen
            for w in range(n_walls):
                t = _ray_segment_intersection_nb(
                    rx, ry, ray_dx, ray_dy,
                    walls_arr[w, 0], walls_arr[w, 1],
                    walls_arr[w, 2], walls_arr[w, 3])
                if t >= 0.0 and t < closest_dist:
                    closest_dist = t
                    closest_type = 1.0

            # Batterien pruefen
            for b in range(n_bats):
                t = _ray_circle_intersection_nb(
                    rx, ry, ray_dx, ray_dy,
                    bat_x[b], bat_y[b], bat_r[b])
                if t >= 0.0 and t < closest_dist:
                    closest_dist = t
                    closest_type = 2.0

            # Andere Roboter pruefen
            for r in range(n_robs):
                # Eigenen Roboter ueberspringen (exakter Index-Vergleich)
                if r == self_rob_idx:
                    continue
                t = _ray_circle_intersection_nb(
                    rx, ry, ray_dx, ray_dy,
                    rob_x[r], rob_y[r], rob_r[r])
                if t >= 0.0 and t < closest_dist:
                    closest_dist = t
                    closest_type = rob_type[r]

            dist_norm = closest_dist / max_length
            hit_x = rx + ray_dx * closest_dist
            hit_y = ry + ray_dy * closest_dist

            result[robot_idx, i, 0] = dist_norm
            result[robot_idx, i, 1] = closest_type
            result[robot_idx, i, 2] = hit_x
            result[robot_idx, i, 3] = hit_y

    return result


# ══════════════════════════════════════════════════════════════════════════════════
# Python-Wrapper (konvertiert Objekte zu NumPy-Arrays fuer Numba)
# ══════════════════════════════════════════════════════════════════════════════════

# Globaler Cache fuer Wand-Arrays (aendern sich nie waehrend einer Runde)
_walls_cache: np.ndarray | None = None
_walls_cache_id: int = 0


def _get_walls_array(walls):
    """Konvertiert Wand-Liste zu NumPy-Array (mit Caching)."""
    global _walls_cache, _walls_cache_id
    walls_id = id(walls)
    if _walls_cache is not None and _walls_cache_id == walls_id:
        return _walls_cache

    _walls_cache = np.array(walls, dtype=np.float64)
    _walls_cache_id = walls_id
    return _walls_cache


def cast_rays(robot, walls: list[tuple[int, int, int, int]],
              batteries: list, robots: list,
              config: SimConfig) -> list[tuple[float, int, float, float]]:
    """Berechnet die Sensor-Strahlen für einen Roboter.

    Wrapper-Funktion, die Python-Objekte in NumPy-Arrays umwandelt
    und die Numba-optimierte Kernfunktion aufruft.

    Args:
        robot: Der Roboter, dessen Sensoren berechnet werden.
        walls: Liste der Wandsegmente [(x1,y1,x2,y2), ...].
        batteries: Liste aller Batterien.
        robots: Liste aller Roboter (exkl. den aktuellen).
        config: Simulationskonfiguration.

    Returns:
        Liste von Tupeln (distance_norm, object_type, hit_x, hit_y) pro Strahl.
        distance_norm: 0.0 = direkt davor, 1.0 = nichts erkannt.
        object_type: OBJ_NONE/OBJ_WALL/OBJ_BATTERY/OBJ_COLLECTOR/OBJ_HUNTER.
        hit_x, hit_y: Trefferpunkt-Koordinaten (für Visualisierung).
    """
    from core.entities import Hunter, ENTITY_HUNTER
    
    ray_count = config.sensor_ray_count
    if getattr(robot, 'entity_type', -1) == ENTITY_HUNTER:
        max_length = config.hunter_sensor_ray_length
        fov_rad = math.radians(config.hunter_sensor_fov)
    else:
        max_length = config.collector_sensor_ray_length
        fov_rad = math.radians(config.collector_sensor_fov)

    # Wand-Array (gecached)
    walls_arr = _get_walls_array(walls)

    # Batterien im Umkreis zu NumPy-Arrays konvertieren
    nearby_bats_x = []
    nearby_bats_y = []
    nearby_bats_r = []
    max_len_sq = (max_length + 15) ** 2  # 15 = max battery radius
    for b in batteries:
        if not b.active:
            continue
        dx = b.x - robot.x
        dy = b.y - robot.y
        if dx * dx + dy * dy <= max_len_sq:
            nearby_bats_x.append(b.x)
            nearby_bats_y.append(b.y)
            nearby_bats_r.append(b.RADIUS)

    n_bats = len(nearby_bats_x)
    if n_bats > 0:
        bat_x = np.array(nearby_bats_x, dtype=np.float64)
        bat_y = np.array(nearby_bats_y, dtype=np.float64)
        bat_r = np.array(nearby_bats_r, dtype=np.float64)
    else:
        bat_x = np.empty(0, dtype=np.float64)
        bat_y = np.empty(0, dtype=np.float64)
        bat_r = np.empty(0, dtype=np.float64)

    # Roboter im Umkreis zu NumPy-Arrays konvertieren
    from core.entities import Collector, Hunter, ENTITY_HUNTER
    nearby_rob_x = []
    nearby_rob_y = []
    nearby_rob_r = []
    nearby_rob_type = []
    max_rob_sq = (max_length + 20) ** 2  # 20 = max robot radius

    for r in robots:
        if r is robot or not r.alive:
            continue
        dx = r.x - robot.x
        dy = r.y - robot.y
        if dx * dx + dy * dy <= max_rob_sq:
            nearby_rob_x.append(r.x)
            nearby_rob_y.append(r.y)
            nearby_rob_r.append(r.radius)
            if getattr(r, 'entity_type', -1) == ENTITY_HUNTER:
                nearby_rob_type.append(4.0)  # OBJ_HUNTER
            else:
                nearby_rob_type.append(3.0)  # OBJ_COLLECTOR

    n_robs = len(nearby_rob_x)
    if n_robs > 0:
        rob_x = np.array(nearby_rob_x, dtype=np.float64)
        rob_y = np.array(nearby_rob_y, dtype=np.float64)
        rob_r = np.array(nearby_rob_r, dtype=np.float64)
        rob_type = np.array(nearby_rob_type, dtype=np.float64)
    else:
        rob_x = np.empty(0, dtype=np.float64)
        rob_y = np.empty(0, dtype=np.float64)
        rob_r = np.empty(0, dtype=np.float64)
        rob_type = np.empty(0, dtype=np.float64)

    # Numba-Kernfunktion aufrufen
    result = _cast_rays_numba(
        robot.x, robot.y, robot.angle,
        ray_count, max_length, fov_rad,
        walls_arr,
        bat_x, bat_y, bat_r, n_bats,
        rob_x, rob_y, rob_r, rob_type, n_robs)

    # Ergebnis in Python-Liste konvertieren
    return [(result[i, 0], int(result[i, 1]), result[i, 2], result[i, 3])
            for i in range(ray_count)]


def cast_rays_batch(robot_list: list, walls: list, batteries: list,
                    all_robots: list, config: SimConfig) -> None:
    """Berechnet Sensor-Strahlen fuer ALLE Roboter parallel (multi-core).

    Sammelt die Daten aller lebenden Roboter in NumPy-Arrays und
    ruft die parallele Numba-Batch-Funktion auf. Schreibt die
    Ergebnisse direkt in robot.sensor_data zurueck.

    Args:
        robot_list: Liste der Roboter, deren Sensoren berechnet werden sollen.
        walls: Liste der Wandsegmente [(x1,y1,x2,y2), ...].
        batteries: Liste aller Batterien.
        all_robots: Liste aller Roboter (fuer Erkennung anderer Roboter).
        config: Simulationskonfiguration.
    """
    from core.entities import Collector, Hunter, ENTITY_HUNTER

    # Lebende Roboter filtern
    alive_robots = [r for r in robot_list if r.alive]
    n_alive = len(alive_robots)
    if n_alive == 0:
        return

    ray_count = config.sensor_ray_count
    if getattr(alive_robots[0], 'entity_type', -1) == ENTITY_HUNTER:
        max_length = config.hunter_sensor_ray_length
        fov_rad = math.radians(config.hunter_sensor_fov)
    else:
        max_length = config.collector_sensor_ray_length
        fov_rad = math.radians(config.collector_sensor_fov)

    # Roboter-Positionen sammeln
    robot_x_arr = np.array([r.x for r in alive_robots], dtype=np.float64)
    robot_y_arr = np.array([r.y for r in alive_robots], dtype=np.float64)
    robot_angle_arr = np.array([r.angle for r in alive_robots], dtype=np.float64)

    # Wand-Array (gecached)
    walls_arr = _get_walls_array(walls)

    # Batterien zu Arrays
    bat_data_x = []
    bat_data_y = []
    bat_data_r = []
    for b in batteries:
        if b.active:
            bat_data_x.append(b.x)
            bat_data_y.append(b.y)
            bat_data_r.append(b.RADIUS)

    n_bats = len(bat_data_x)
    bat_x = np.array(bat_data_x, dtype=np.float64) if n_bats > 0 else np.empty(0, dtype=np.float64)
    bat_y = np.array(bat_data_y, dtype=np.float64) if n_bats > 0 else np.empty(0, dtype=np.float64)
    bat_r = np.array(bat_data_r, dtype=np.float64) if n_bats > 0 else np.empty(0, dtype=np.float64)

    # Alle lebenden Roboter fuer Sichtpruefung zu Arrays
    # Wir erstellen auch ein Mapping: Roboter-Objekt -> Index im Array
    rob_data_x = []
    rob_data_y = []
    rob_data_r = []
    rob_data_type = []
    rob_obj_to_idx = {}  # Mapping Roboter-id -> Array-Index
    alive_rob_idx = 0
    for r in all_robots:
        if not r.alive:
            continue
        rob_obj_to_idx[id(r)] = alive_rob_idx
        alive_rob_idx += 1
        rob_data_x.append(r.x)
        rob_data_y.append(r.y)
        rob_data_r.append(r.radius)
        if getattr(r, 'entity_type', -1) == ENTITY_HUNTER:
            rob_data_type.append(4.0)  # OBJ_HUNTER
        else:
            rob_data_type.append(3.0)  # OBJ_COLLECTOR

    n_robs = len(rob_data_x)
    rob_x = np.array(rob_data_x, dtype=np.float64) if n_robs > 0 else np.empty(0, dtype=np.float64)
    rob_y = np.array(rob_data_y, dtype=np.float64) if n_robs > 0 else np.empty(0, dtype=np.float64)
    rob_r = np.array(rob_data_r, dtype=np.float64) if n_robs > 0 else np.empty(0, dtype=np.float64)
    rob_type = np.array(rob_data_type, dtype=np.float64) if n_robs > 0 else np.empty(0, dtype=np.float64)

    # Index jedes Roboters im rob_x/rob_y-Array fuer Self-Exclusion
    robot_rob_idx = np.array(
        [rob_obj_to_idx.get(id(r), -1) for r in alive_robots],
        dtype=np.int64)

    # Parallele Batch-Berechnung (alle Roboter gleichzeitig auf allen Kernen)
    result = _cast_rays_batch_numba(
        n_alive,
        robot_x_arr, robot_y_arr, robot_angle_arr,
        robot_rob_idx,
        ray_count, max_length, fov_rad,
        walls_arr,
        bat_x, bat_y, bat_r, n_bats,
        rob_x, rob_y, rob_r, rob_type, n_robs)

    # Ergebnisse in die Roboter-Objekte zurueckschreiben
    for idx, robot in enumerate(alive_robots):
        robot.sensor_data = [
            (result[idx, i, 0], int(result[idx, i, 1]),
             result[idx, i, 2], result[idx, i, 3])
            for i in range(ray_count)
        ]


def draw_sensors(screen: pygame.Surface, robot, show_sensors: bool) -> None:
    """Zeichnet die Sensor-Strahlen eines Roboters.

    Args:
        screen: Pygame-Zeichenfläche.
        robot: Der Roboter mit sensor_data.
        show_sensors: Ob die Sensoren gezeichnet werden sollen.
    """
    if not show_sensors or not robot.alive or not robot.sensor_data:
        return

    ix, iy = int(robot.x), int(robot.y)

    for dist_norm, obj_type, hit_x, hit_y in robot.sensor_data:
        hx, hy = int(hit_x), int(hit_y)

        # Farbe nach Objekttyp
        color = RAY_COLORS.get(obj_type, COLOR_RAY_MISS)

        # Strahl zeichnen (als dünne Linie)
        pygame.draw.line(screen, color[:3], (ix, iy), (hx, hy), 1)

        # Trefferpunkt markieren (wenn etwas getroffen wurde)
        if obj_type != OBJ_NONE:
            pygame.draw.circle(screen, COLOR_HIT_POINT, (hx, hy), 3)
