"""
spatial_grid.py – Grid-basierte Kollisionserkennung fuer das Neuro-Oekosystem.

Teilt die Welt in Zellen auf und ermoeglicht effiziente
Proximity-Abfragen. Raycasts und Kollisionschecks pruefen
nur Objekte in relevanten Zellen statt aller Objekte.

Performance-Optimierungen:
- defaultdict vermeidet wiederholte key-in-dict Pruefungen
- clear() leert Listen in-place statt dict neu zu erstellen (weniger GC)
- get_nearby() inlined die Distanzpruefung
"""

from collections import defaultdict


class SpatialGrid:
    """Grid-basierte Raum-Partitionierung fuer effiziente Proximity-Abfragen.

    Die Welt wird in quadratische Zellen unterteilt. Jede Entity wird
    in die Zelle eingefuegt, die ihren Mittelpunkt enthaelt.
    Abfragen liefern nur Entities aus relevanten Zellen.

    Attributes:
        cell_size: Groesse einer Zelle in Pixeln
        grid: Dictionary {(cell_x, cell_y): [entity_list]}
    """

    __slots__ = ('cell_size', 'grid', '_active_cells', '_inv_cell_size')

    def __init__(self, cell_size: int = 100) -> None:
        self.cell_size = cell_size
        self._inv_cell_size = 1.0 / cell_size  # Multiplikation statt Division
        self.grid: dict[tuple[int, int], list] = defaultdict(list)
        self._active_cells: list[tuple[int, int]] = []  # Tracking aktiver Zellen

    def clear(self) -> None:
        """Leert das gesamte Grid. Muss jeden Frame aufgerufen werden.
        
        Leert Listen in-place statt das Dict neu zu erstellen.
        Das reduziert GC-Druck und dict-Reallokationen.
        """
        for key in self._active_cells:
            self.grid[key].clear()
        self._active_cells.clear()

    def insert(self, entity) -> None:
        """Fuegt eine Entity in die passende Zelle ein.

        Args:
            entity: Objekt mit x, y Attributen.
        """
        key = (int(entity.x * self._inv_cell_size),
               int(entity.y * self._inv_cell_size))
        cell = self.grid[key]
        if not cell:  # Zelle war leer -> als aktiv tracken
            self._active_cells.append(key)
        cell.append(entity)

    def insert_all(self, entities: list) -> None:
        """Fuegt mehrere Entities ein.

        Args:
            entities: Liste von Objekten mit x, y Attributen.
        """
        inv = self._inv_cell_size
        grid = self.grid
        active = self._active_cells
        for entity in entities:
            key = (int(entity.x * inv), int(entity.y * inv))
            cell = grid[key]
            if not cell:
                active.append(key)
            cell.append(entity)

    def query_radius(self, x: float, y: float, radius: float) -> list:
        """Gibt alle Entities im Umkreis einer Position zurueck.

        Prueft alle Zellen, die vom Suchkreis ueberlappt werden.
        Fuehrt anschliessend eine exakte Distanzpruefung durch.

        Args:
            x, y: Suchposition
            radius: Suchradius

        Returns:
            Liste von Entities innerhalb des Radius.
        """
        results = []
        r_sq = radius * radius
        inv = self._inv_cell_size
        grid = self.grid

        # Zellen-Bereich berechnen, der vom Suchkreis ueberlappt wird
        min_cx = int((x - radius) * inv)
        max_cx = int((x + radius) * inv)
        min_cy = int((y - radius) * inv)
        max_cy = int((y + radius) * inv)

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell = grid.get((cx, cy))
                if cell:
                    for entity in cell:
                        dx = entity.x - x
                        dy = entity.y - y
                        if dx * dx + dy * dy <= r_sq:
                            results.append(entity)

        return results

    def get_nearby(self, entity, radius: float) -> list:
        """Gibt alle Nachbar-Entities einer Entity zurueck (exkl. sich selbst).

        Inlined die Distanzpruefung und Selbst-Ausschluss fuer Performance.

        Args:
            entity: Die Entity, deren Nachbarn gesucht werden.
            radius: Suchradius.

        Returns:
            Liste von Nachbar-Entities (ohne die Entity selbst).
        """
        results = []
        x = entity.x
        y = entity.y
        r_sq = radius * radius
        inv = self._inv_cell_size
        grid = self.grid

        min_cx = int((x - radius) * inv)
        max_cx = int((x + radius) * inv)
        min_cy = int((y - radius) * inv)
        max_cy = int((y + radius) * inv)

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell = grid.get((cx, cy))
                if cell:
                    for e in cell:
                        if e is not entity:
                            dx = e.x - x
                            dy = e.y - y
                            if dx * dx + dy * dy <= r_sq:
                                results.append(e)

        return results

    def query_rect(self, x1: float, y1: float, x2: float, y2: float) -> list:
        """Gibt alle Entities in einem Rechteck zurueck.

        Args:
            x1, y1: Obere linke Ecke
            x2, y2: Untere rechte Ecke

        Returns:
            Liste von Entities im Rechteck.
        """
        results = []
        inv = self._inv_cell_size
        min_cx = int(x1 * inv)
        max_cx = int(x2 * inv)
        min_cy = int(y1 * inv)
        max_cy = int(y2 * inv)

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell = self.grid.get((cx, cy))
                if cell:
                    for entity in cell:
                        if x1 <= entity.x <= x2 and y1 <= entity.y <= y2:
                            results.append(entity)

        return results
