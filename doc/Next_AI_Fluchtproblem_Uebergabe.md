# Uebergabe fuer naechste KI: Fluchtproblem der Sammler

## Ausgangsproblem

Die Sammler fluechten nicht sauber vor den Jaegern/Fressern. In der Visualisierung steuern sie teilweise sogar auf Jaeger zu.

## Eingebaute Aenderungen

### 1. CSV-Logging

Datei: `ai/neat_ai.py`

Es wurde ein persistentes Generations-Logging nach `log/evolution_generations.csv` eingebaut.
Beim Spaltenwechsel wird die alte CSV automatisch mit Timestamp archiviert.

Neue bzw. wichtige Felder:

- `escape_ratio`
- `approach_ratio`
- `collector_escape_events`
- `collector_approach_events`
- `collector_danger_frames`
- `collector_sees_hunter_frames`
- `avg_nearest_hunter_dist`
- `avg_fit_escape`
- `avg_fit_approach`
- `avg_fit_hunter_penalty`

Archivierte Vorher-Datei:

- `log/evolution_generations_20260510_155738.csv`

Aktuelle Nachher-Datei:

- `log/evolution_generations.csv`

### 2. Fitness-Fix gegen Jaeger-Annäherung

Dateien:

- `config/config_manager.py`
- `ai/neat_ai.py`
- `core/entities.py`

Neue/angepasste Parameter:

- `fitness_hunter_danger = 0.08`
- `fitness_hunter_approach_penalty = 8.0`

Im Trainingsloop wird mindestens `fitness_hunter_danger >= 0.08` und `fitness_hunter_approach_penalty >= 8.0` verwendet, damit alte `sim_config.json`-Werte den Fix nicht deaktivieren.

In `ai/neat_ai.py` passiert jetzt:

- Naehe zum Jaeger erzeugt einen Distanz-Gradienten-Malus:
  naeher am Jaeger = staerkerer Malus.
- Wenn der Abstand zum Jaeger kleiner wird (`dist_delta < 0`), gibt es eine direkte Annäherungsstrafe.
- Escape-Bonus und Approach-Malus werden getrennt geloggt.

In `core/entities.py` wurden Tracking-Felder ergaenzt:

- `fit_escape`
- `fit_approach`

## Auswertung der Logs

### Vor dem Fix

Datei: `log/evolution_generations_20260510_155738.csv`

- Zeilen: 40
- Weighted `escape_ratio`: ca. `0.4595`
- Weighted `approach_ratio`: ca. `0.5405`
- Durchschnittliche Kills: ca. `25.8`
- Durchschnittlich lebende Sammler: ca. `1.0`
- `avg_fit_hunter_penalty`: `0.0`
- `avg_fit_approach`: `0.0`

Befund:

Die Sammler bewegten sich in Gefahr haeufiger auf Jaeger zu als von ihnen weg. Es gab keinen direkten Malus fuer Annäherung.

### Nach dem Fix

Datei: `log/evolution_generations.csv`

- Zeilen: 26
- Weighted `escape_ratio`: ca. `0.4516`
- Weighted `approach_ratio`: ca. `0.5484`
- Durchschnittliche Kills: ca. `29.2`
- Durchschnittlich lebende Sammler: ca. `0.62`
- `avg_fit_hunter_penalty`: ca. `-4.7`
- `avg_fit_escape`: ca. `189.2`
- `avg_fit_approach`: ca. `-421.7`

Befund:

Die neue Strafe greift messbar, aber das Verhalten ist noch nicht besser. Die Sammler naehern sich weiterhin haeufiger an, als sie fliehen. Kills sind sogar hoeher als vorher. Die Fitnessstrafe ist also sichtbar, aber noch nicht stark/strukturiert genug, um die Evolution schnell umzudrehen.

## Wichtige Diagnose

Die Logs zeigen:

- `collector_danger_frames` ist sehr hoch.
- `collector_sees_hunter_frames` ist deutlich niedriger.

Interpretation:

Sammler sind oft in der Gefahrenzone, aber der Jaeger ist nicht immer als Sensor-Hit sichtbar. Das Netz bekommt also nicht immer ein klares Input-Signal fuer Gefahr, obwohl die Fitnessberechnung bereits Gefahr misst.

Das kann mehrere Ursachen haben:

1. Nur 5 Sensorstrahlen sind fuer Fluchtverhalten eventuell zu grob.
2. Sensor-Update im Turbo nur jeden 4. Frame kann Gefahr verspaeten.
3. Die Fitness weiss ueber `grid.get_nearby(...)`, dass ein Jaeger nah ist, aber das neuronale Netz bekommt nur Ray-Hits. Das ist ein Signal-Mismatch.
4. Batterie-/Survival-Reize koennen weiterhin kurzfristig dominieren.

## Empfohlene naechste Schritte

### Prioritaet 1: Gefahr als direkter Netz-Input

Ergaenze einen oder zwei direkte Inputs fuer Sammler:

- `nearest_hunter_distance_norm`
- optional `nearest_hunter_bearing` oder `hunter_danger_scalar`

Dann muessen Sammler nicht darauf hoffen, dass ein Ray genau den Jaeger trifft.

Achtung:

- Das aendert `num_inputs`.
- Alte Hall-of-Fame-Genome werden inkompatibel.
- Vorher `hall_of_fame.pkl` sichern oder loeschen.

### Prioritaet 2: Annäherungsstrafe staerker machen

Testweise:

- `fitness_hunter_approach_penalty` von `8.0` auf `15.0` oder `20.0`
- `fitness_hunter_danger` von `0.08` auf `0.15`

Zielwerte im Log:

- `approach_ratio < 0.50`
- `escape_ratio > 0.50`
- Kills deutlich unter `20`
- `collectors_alive` steigt ueber `5`

### Prioritaet 3: Sensorik fuer Sammler verbessern

Moegliche Tests:

- `sensor_ray_count` von `5` auf `7` oder `9`
- `collector_sensor_fov` bei `240` lassen oder UI-Grenze passend machen
- Im Turbo Sensoren jeden 2. statt jeden 4. Frame aktualisieren

### Prioritaet 4: Fitness-Breakdown sauber trennen

Aktuell wird Escape-Bonus noch auch in `fit_surv` addiert. Fuer Analyse besser:

- `fit_surv` nur echter Survival-Bonus
- `fit_escape` nur Fluchtbonus

## Befehle zum Pruefen

```powershell
python -m py_compile main.py ai\neat_ai.py ai\hall_of_fame.py config\config_manager.py core\world.py core\entities.py core\sensors.py core\spatial_grid.py core\fast_net.py ui\brain_viewer.py ui\commentary.py ui\hall_of_fame_window.py ui\stats_graph.py
python test_fast_net.py
```

## Kurzfazit

Der erste Fitness-Fix ist drin, aber noch nicht ausreichend. Die naechste wirksame Aenderung sollte wahrscheinlich nicht nur eine staerkere Strafe sein, sondern ein besseres Gefahrensignal im neuronalen Netz: direkte Hunter-Distanz/Bearing-Inputs fuer Sammler.
