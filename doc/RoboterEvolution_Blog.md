# 🤖 RoboterEvolution – Wenn KI-Roboter das Überleben lernen

**Ein Neuro-Ökosystem mit Co-Evolution, gebaut mit Python, Pygame und NEAT**

---

## Die Idee: Evolution im Zeitraffer

Was passiert, wenn man 100 virtuelle Roboter in eine Welt setzt, ihnen ein Gehirn gibt – und dann die Natur entscheiden lässt, wer überlebt?

Genau das macht **RoboterEvolution**: Eine Echtzeit-Simulation, in der zwei Arten von KI-Robotern durch **Neuroevolution** lernen, in einer 2D-Welt zu überleben. Keine vorprogrammierten Regeln, kein menschliches Eingreifen – nur Evolution.

### Die Spieler

| Rolle | Farbe | Ziel | Population |
|-------|-------|------|------------|
| **Sammler** | 🟢 Grün | Batterien finden, Jägern ausweichen | 100 |
| **Jäger** | 🔴 Rot | Sammler jagen und fressen | 10 |
| **Hall-of-Fame Gäste** | 🔵 Blau | Die 5 besten Gehirne aller Zeiten | 5 |

Die Sammler müssen Batterien einsammeln um zu überleben. Die Jäger müssen Sammler fangen um zu überleben. Beide Gruppen entwickeln ihre Strategien **gleichzeitig** – ein evolutionäres Wettrüsten.

---

## Wie funktioniert das? – Die Technik im Detail

### 1. Das Gehirn: NEAT-Algorithmus

Jeder Roboter besitzt ein eigenes **neuronales Netz** als Gehirn. Dieses Netz wird nicht von Hand designt, sondern durch den **NEAT-Algorithmus** (NeuroEvolution of Augmenting Topologies) automatisch entwickelt.

NEAT ist besonders, weil es nicht nur die **Gewichte** eines Netzes optimiert, sondern auch dessen **Struktur**. Es kann neue Neuronen und Verbindungen hinzufügen – das Netz wird also im Laufe der Evolution immer komplexer, aber nur dort, wo Komplexität tatsächlich hilft.

**Ablauf einer Generation:**
1. Alle Roboter werden in die Welt gesetzt
2. 2.500 Frames lang simuliert (ca. 40 Sekunden Echtzeit)
3. Jeder Roboter bekommt eine **Fitness** basierend auf seinem Verhalten
4. Die besten Roboter "paaren" sich – ihre Gehirne werden kombiniert und mutiert
5. Die nächste Generation startet

### 2. Die Sinne: Raycasting-Sensoren

Jeder Roboter hat **5 Sichtstrahlen**, die wie Laserpointer von ihm ausgehen. Diese Strahlen erkennen:

- **Wände und Hindernisse** – damit der Roboter nicht vor Wände fährt
- **Batterien** – damit der Roboter Nahrung finden kann
- **Jäger** – damit der Roboter vor Gefahr fliehen kann

Jeder Strahl liefert **4 Werte** an das neuronale Netz:

| Input | Bedeutung | Beispiel |
|-------|-----------|---------|
| `distanz` | Wie weit ist das Objekt? (0 = direkt davor, 1 = nichts in Sicht) | 0.3 |
| `ist_batterie` | Ist es eine Batterie? (1 = ja, 0 = nein) | 1.0 |
| `ist_jaeger` | Ist es ein Jäger? (1 = GEFAHR!, 0 = nein) | 0.0 |
| `ist_wand` | Ist es eine Wand? (1 = ja, 0 = nein) | 0.0 |

Diese klare Trennung der Signale ist entscheidend: Das neuronale Netz kann mit einer **einzigen Gewichtsverbindung** lernen, dass `ist_jaeger = 1` bedeutet "dreh um und flieh!". In einer früheren Version waren die Objekttypen als einzelne Zahl kodiert (Wand = 0.25, Batterie = 0.50, Jäger = 1.0) – das war für das Netz fast unmöglich zu unterscheiden.

**Performance:** Die Raycasting-Berechnungen sind mit **Numba JIT** kompiliert und laufen mit nahezu C-Geschwindigkeit. Im Turbo-Modus werden alle Roboter parallel auf mehreren CPU-Kernen berechnet.

### 3. Die Physik: Differential-Drive Kinematik

Die Roboter bewegen sich wie echte Panzerfahrzeuge mit **zwei unabhängigen Motoren** (links und rechts):

```
Beide Motoren vorwärts  → Roboter fährt geradeaus
Nur linker Motor        → Roboter dreht nach rechts
Motoren gegenläufig     → Roboter dreht auf der Stelle
```

Das neuronale Netz hat genau **2 Outputs**: `linker_motor` und `rechter_motor` (jeweils -1.0 bis 1.0). Damit kann der Roboter jede beliebige Bewegung ausführen – er muss nur lernen, welche Motorwerte in welcher Situation sinnvoll sind.

### 4. Die Welt

Die Simulationswelt ist ein **1600×1600 Pixel** großes Feld mit:

- **Wänden** am Rand
- **30 zufälligen Hindernissen** (Rechtecke)
- **100 Batterien** die bei Einsammlung sofort an neuer Position respawnen

Die Welt bleibt **10 Generationen lang** identisch, damit die Evolution einen fairen Vergleich hat. Danach wird eine neue Welt generiert, damit die Roboter nicht nur eine einzige Karte auswendig lernen.

---

## Das Fitness-System: Wie lernen die Roboter?

Die Fitness ist die "Überlebenswährung" der Evolution. Roboter mit hoher Fitness pflanzen sich fort, Roboter mit niedriger Fitness sterben aus.

### Diskrete Events

| Event | Fitness-Änderung | Bedeutung |
|-------|-----------------|-----------|
| Batterie gesammelt | **+100** | Hauptziel: Aktiv Nahrung suchen |
| Von Jäger gefressen | **-200** | Brutale Strafe: Tod ist teuer |
| Jäger fängt Sammler | **+200** | Belohnung für erfolgreiche Jagd |
| Herumstehen (pro Frame) | **-0.02** | Strafe für Faulheit |
| Überleben (pro Frame) | **+0.01** | Minimaler Tiebreaker |

### Fitness-Gradient: Das "Wärmer/Kälter"-System

Das Herzstück des Lernsystems ist der **Proximity-Gradient**. Statt nur bei diskreten Events (Batterie berührt, gefressen werden) Feedback zu geben, bekommen die Roboter **jeden Frame** ein Richtungssignal:

**Für Sammler:**
- 🟢 **Nahe an Batterie** → kleiner Bonus pro Frame (max. +0.1) → *"Du wirst wärmer!"*
- 🔴 **Nahe an Jäger** (< 150 Pixel) → Strafe pro Frame (max. -0.15) → *"GEFAHR! Weg da!"*

**Für Jäger:**
- 🟢 **Nahe an Sammler** → kleiner Bonus pro Frame → *"Beute in der Nähe!"*

Ohne diesen Gradienten bekommt ein Roboter **kein Feedback** bis er zufällig eine Batterie berührt. Das ist wie ein blinder Mensch, dem man nur sagt "du hast Gold gefunden" – aber nie "du wirst wärmer/kälter". Mit dem Gradienten lernen die Roboter schon in den ersten Generationen, gezielt auf Batterien zuzufahren.

**Alle Gradient-Parameter sind live im Config-Menü einstellbar:**
- Batterie-Nähe Bonus (Standard: 0.10)
- Jäger-Nähe Strafe (Standard: 0.15)
- Gefahrenzone Radius (Standard: 150 Pixel)

---

## Co-Evolution: Das Wettrüsten

Das Faszinierendste an diesem System ist die **Co-Evolution**: Sammler und Jäger entwickeln sich gleichzeitig und beeinflussen sich gegenseitig.

```
Generation 1:  Sammler fahren zufällig → Jäger fangen leicht
Generation 10: Sammler lernen wegzufahren → Jäger müssen schneller werden
Generation 30: Jäger lernen Hinterhalte → Sammler lernen Ausweichmanöver
Generation 50: Sammler nutzen Hindernisse als Deckung → Jäger umzingeln
...und so weiter, endlos.
```

Dieses Wettrüsten produziert Strategien, die **kein Mensch programmiert hat**. Die Roboter entwickeln emergentes Verhalten – Verhaltensweisen, die aus den einfachen Regeln von alleine entstehen.

### Die Balance-Herausforderung

Ein kritischer Aspekt ist das **Gleichgewicht** zwischen Sammlern und Jägern:

- **Zu viele/starke Jäger** → Alle Sammler sterben sofort → Keine Evolution möglich
- **Zu wenige/schwache Jäger** → Kein Selektionsdruck → Sammler lernen nicht auszuweichen

Deshalb: **100 Sammler vs. nur 10 Jäger**. Sammler sind schneller (3.5 vs. 3.0), sehen weiter (300 vs. 200 Pixel) und haben ein breiteres Sichtfeld (120° vs. 90°). Die Jäger müssen strategisch jagen – rohe Geschwindigkeit reicht nicht.

---

## Die Hall of Fame

Die **Hall of Fame** speichert die 20 besten Roboter-Gehirne aller Zeiten. Diese dienen als evolutionäres Gedächtnis:

- **Live-Injektion (Gäste)**: In *jeder einzelnen Generation* schaut das System live in die aktuelle Hall of Fame. Die besten 5 Roboter werden sofort als unsterbliche "Gast-Roboter" (🔵 cyan-blaue Farbe) ins laufende Spiel eingespeist, um der aktuellen Population live als Vorbild zu dienen. Dies geschieht vollautomatisch im Hintergrund.
- **Persistenz**: Die Hall of Fame wird als Pickle-Datei gespeichert und überlebt Neustarts.
- **Live-Fenster**: Ein eigenständiges Tkinter-Fenster zeigt die Rangliste in Echtzeit – mit 🥇🥈🥉 Medaillen, Fitness-Scores und gesammelten Batterien.

---

## Die Benutzeroberfläche

Das Projekt besteht aus **vier synchronisierten Fenstern**:

### 1. Simulationsfenster (Pygame)
Das Hauptfenster zeigt die Welt in Echtzeit:
- Grüne Kreise = Sammler
- Rote Kreise = Jäger
- Blaue Kreise = Hall-of-Fame Gäste
- Gelbe Punkte = Batterien
- Graue Rechtecke = Hindernisse

**Tastatursteuerung:**
- `T` = Turbo-Modus (kein Rendering, maximale Geschwindigkeit)
- `S` = Sensor-Strahlen ein/ausblenden
- `+/-` = Simulationsgeschwindigkeit anpassen
- `ESC` = Beenden

### 2. Lernkurven-Fenster (Matplotlib)
Zeigt die evolutionäre Entwicklung als Live-Graph:
- Beste Fitness von Sammlern und Jägern
- Kills pro Generation
- Überlebende Sammler
- Gesammelte Batterien

Mit Turbo-Button und Brain-Viewer-Zugang.

### 3. Live-Kommentar (Tkinter)
Ein Echtzeit-Kommentator, der die Simulation begleitet:
> *"Generation 15 gestartet! 100 Sammler vs. 10 Jäger..."*
> *"🏆 Neuer Hall of Fame Eintrag: Apex-042 mit 1.250 Fitness!"*
> *"⚔️ Die Jäger haben 23 Sammler erlegt!"*

### 4. Hall of Fame (Tkinter)
Eigenständiges Ranking-Fenster mit den Top 20 der besten Gehirne.

### 5. Brain Viewer (Tkinter)
Diagnostik-Tool zur Visualisierung der neuronalen Netze:
- Zeigt die Netzwerk-Topologie eines ausgewählten Genoms
- Analysiert Gewichtsverbindungen und interpretiert die Strategie
- Erkennt Muster wie "Annäherung an Batterien" oder "Flucht vor Jägern"

---

## Konfiguration: Alles einstellbar

Beim Start öffnet sich ein **interaktives Config-Menü** mit Schiebereglern für alle Parameter:

### Welt & Umgebung
| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| Weltgröße | 40×40 | N×N Felder (1600×1600 Pixel) |
| Hindernisse | 30 | Zufällige Rechteck-Hindernisse |
| Batterien | 100 | Nahrungsquellen mit Sofort-Respawn |

### Roboter & Sensoren
| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| Sammler-Speed | 3.5 | Schneller als Jäger (Fluchtvorteil) |
| Jäger-Speed | 3.0 | Muss strategisch jagen |
| Sensor-Strahlen | 5 | Sichtstrahlen pro Roboter |
| Sichtweite Sammler | 300 px | Sehen weiter als Jäger |
| Sichtfeld Sammler | 120° | Breiteres Sichtfeld |
| Sichtweite Jäger | 200 px | Kürzere Sicht |
| Sichtfeld Jäger | 90° | Tunnelblick |

### Energie
| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| Start-Energie | 100 | Energie bei Geburt |
| Verlust/Frame | 0.04 | Reicht für genau 2.500 Frames |
| Batterie-Energie | 30 | Energie pro gesammelter Batterie |

### Evolution
| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| Sammler-Population | 100 | Größe der Sammler-Population |
| Jäger-Population | 10 | Bewusst klein für Balance |
| Frames/Generation | 2.500 | Simulationsdauer pro Runde |

### Fitness
| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| Batterie gesammelt | +100 | Hauptziel |
| Gefressen-Strafe | -200 | Tod ist teuer |
| Jäger-Kill-Bonus | +200 | Belohnung fürs Fangen |
| Batterie-Nähe Bonus | 0.10 | Gradient zum Futter |
| Jäger-Nähe Strafe | 0.15 | Gradient weg von Gefahr |
| Gefahrenzone | 150 px | Radius der Jäger-Gefahrenzone |

Alle Einstellungen werden als **JSON** gespeichert und beim nächsten Start automatisch geladen.

---

## Technologie-Stack

| Komponente | Technologie | Zweck |
|------------|-------------|-------|
| Simulation | **Pygame** | 2D-Rendering und Event-Handling |
| KI | **neat-python** | Neuroevolution (NEAT-Algorithmus) |
| Performance | **Numba** | JIT-Kompilierung für Raycasting |
| Mathematik | **NumPy** | Array-Operationen für Sensoren |
| Graphen | **Matplotlib** | Live-Lernkurven |
| UI-Fenster | **Tkinter** | Kommentar, Hall of Fame, Brain Viewer |

---

## Architektur

```
RoboterEvolution/
├── main.py                    # Einstiegspunkt: Menü → HoF → Training
├── config/
│   └── config_manager.py      # SimConfig Dataclass + Slider-Menü
├── core/
│   ├── entities.py            # Battery, Robot, Collector, Hunter
│   ├── world.py               # Welt-Generation, Hindernisse, Spawning
│   ├── sensors.py             # Numba-JIT Raycasting
│   └── spatial_grid.py        # Räumliche Partitionierung für Performance
├── ai/
│   ├── neat_ai.py             # CoEvolutionManager (Herzstück)
│   ├── hall_of_fame.py        # Persistente Top-20 Rangliste
│   ├── config-collector.txt   # NEAT-Config für Sammler
│   └── config-hunter.txt      # NEAT-Config für Jäger
├── ui/
│   ├── commentary.py          # Live-Kommentar-Fenster
│   ├── stats_graph.py         # Matplotlib Lernkurven
│   ├── brain_viewer.py        # Netzwerk-Visualisierung
│   └── hall_of_fame_window.py # HoF-Ranking-Fenster
└── doc/
    └── RoboterEvolution_Blog.md  # Diese Dokumentation
```

---

## Installation & Start

### Voraussetzungen
- Python 3.12+
- Windows (für Pygame-Fenster)

### Setup
```bash
pip install pygame neat-python matplotlib numba numpy
```

### Starten
```bash
python main.py
```

Oder einfach `start.bat` doppelklicken.

### Ablauf
1. **Config-Menü** öffnet sich → Parameter anpassen → "Start" klicken
2. **Hall of Fame** wird angezeigt → Alte Genome auswählen oder "Start" für frischen Beginn
3. **Simulation** startet mit allen 4 Fenstern
4. `T` drücken für **Turbo-Modus** (100× schneller, kein Rendering)

---

## Lessons Learned

### Was funktioniert
- **Klare Sensor-Signale** sind entscheidend. Separate Inputs für jeden Objekttyp (`ist_batterie`, `ist_jaeger`, `ist_wand`) statt einer einzigen kodierten Zahl ermöglichen schnelles Lernen.
- **Fitness-Gradienten** (Wärmer/Kälter-System) sind der Schlüssel. Ohne sie bekommt die Evolution kein Richtungssignal.
- **Stabile Welten** über mehrere Generationen reduzieren Fitness-Rauschen und machen den Lernfortschritt sichtbar.
- **Asymmetrie** zwischen Sammler und Jäger (Geschwindigkeit, Sichtweite, Population) ist essenziell für ein stabiles Ökosystem.

### Was schwierig ist
- **Co-Evolution ist instabil**: Wenn eine Seite zu dominant wird, bricht die andere zusammen → evolutionäre Stagnation.
- **Balance finden**: Die richtige Kombination aus Populationsgrößen, Geschwindigkeiten und Fitness-Gewichten erfordert viel Experimentieren.
- **Performance bei großen Populationen**: 110 Roboter mit je 5 Sensorstrahlen × 2.500 Frames = 1,375 Millionen Raycast-Operationen pro Generation. Ohne Numba-JIT und Spatial-Grid wäre das nicht machbar.

---

## Fazit

RoboterEvolution zeigt eindrucksvoll, wie aus einfachen Regeln komplexes Verhalten entstehen kann. Kein einziger Roboter wurde programmiert, sich zu bewegen, Batterien zu suchen oder Jägern auszuweichen – all das entsteht durch den evolutionären Druck, der über Generationen hinweg wirkt.

Das Projekt demonstriert, wie **Neuroevolution** als Alternative zu klassischem Reinforcement Learning funktioniert: Statt einen einzelnen Agenten zu trainieren, lässt man eine ganze Population gleichzeitig lernen – und die Natur erledigt den Rest.

---

*Gebaut mit Python, Pygame, NEAT und einer Menge Koffein. 🤖☕*
