import neat, os, sys
sys.path.insert(0, '.')
from core.fast_net import FastNetwork
import numpy as np

# NEAT Population
config_path = os.path.join('ai', 'config-collector.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
pop = neat.Population(config)

# Test alle Genome in der Population
inputs = [0.5] * 26
max_diff = 0.0
n_tested = 0

for genome_id, genome in pop.population.items():
    neat_net = neat.nn.FeedForwardNetwork.create(genome, config)
    fast_net = FastNetwork(neat_net)
    
    neat_out = neat_net.activate(inputs)
    fast_out = fast_net.activate(inputs)
    
    diff = max(abs(a - b) for a, b in zip(neat_out, fast_out))
    if diff > max_diff:
        max_diff = diff
    n_tested += 1

print(f'Getestet: {n_tested} Genome')
print(f'Max Differenz: {max_diff:.12f}')
if max_diff < 1e-6:
    print('Status: OK - Alle Genome identisch!')
else:
    print('Status: FEHLER!')

# Benchmark mit EINEM Netz
genome_id, genome = list(pop.population.items())[0]
neat_net = neat.nn.FeedForwardNetwork.create(genome, config)
fast_net = FastNetwork(neat_net)

# Warmup JIT
for _ in range(100):
    fast_net.activate(inputs)

import time
n = 100000
t0 = time.perf_counter()
for _ in range(n):
    neat_net.activate(inputs)
t_neat = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(n):
    fast_net.activate(inputs)
t_fast = time.perf_counter() - t0

print(f'\nBenchmark ({n} Evaluierungen, gleiches Netz):')
print(f'  neat-python: {t_neat*1000:.1f} ms ({t_neat/n*1e6:.1f} us/eval)')
print(f'  FastNetwork: {t_fast*1000:.1f} ms ({t_fast/n*1e6:.1f} us/eval)')
print(f'  Speedup:     {t_neat/t_fast:.1f}x')

# Hochrechnung fuer 1 Generation (110 Roboter x 2500 Frames)
evals_per_gen = 110 * 2500
print(f'\nHochrechnung fuer 1 Generation ({evals_per_gen} Evaluierungen):')
print(f'  neat-python: {evals_per_gen * t_neat/n:.1f} Sekunden')
print(f'  FastNetwork: {evals_per_gen * t_fast/n:.1f} Sekunden')
print(f'  Gespart:     {evals_per_gen * (t_neat - t_fast)/n:.1f} Sekunden')
