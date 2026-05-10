import neat, os, sys
sys.path.insert(0, '.')
from core.fast_net import FastNetwork
from core.cuda_net import CudaBatchedNetwork
import numpy as np
import time

# NEAT Population
config_path = os.path.join('ai', 'config-collector.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
pop = neat.Population(config)

n_genomes = len(pop.population)
fast_nets = []
for genome_id, genome in pop.population.items():
    neat_net = neat.nn.FeedForwardNetwork.create(genome, config)
    fast_nets.append(FastNetwork(neat_net))

# Init Cuda Batch
print("Initialisiere CudaBatchedNetwork...")
cuda_net = CudaBatchedNetwork(fast_nets)

# Test correctness
inputs = np.random.rand(n_genomes, 26).astype(np.float64)

# CPU
print("Berechne CPU (FastNetwork)...")
cpu_outputs = np.zeros((n_genomes, cuda_net.n_outputs))
for i in range(n_genomes):
    cpu_outputs[i] = fast_nets[i].activate(inputs[i])

# GPU
print("Berechne GPU (CudaBatchedNetwork)...")
gpu_outputs = cuda_net.activate_batch(inputs)

diff = np.max(np.abs(cpu_outputs - gpu_outputs))
print(f"Max Differenz zwischen CPU und GPU: {diff:.12f}")

if diff < 1e-5:
    print("Test BESTANDEN!")
else:
    print("Test FEHLGESCHLAGEN!")
    
# Benchmark
print("\nStarte Benchmark...")
n_evals = 1000

# Warmup GPU
_ = cuda_net.activate_batch(inputs)

t0 = time.perf_counter()
for _ in range(n_evals):
    for i in range(n_genomes):
        fast_nets[i].activate(inputs[i])
t_cpu = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(n_evals):
    cuda_net.activate_batch(inputs)
t_gpu = time.perf_counter() - t0

print(f"CPU Time für {n_evals} Evaluierungen (Batches von {n_genomes}): {t_cpu*1000:.1f} ms")
print(f"GPU Time für {n_evals} Evaluierungen (Batches von {n_genomes}): {t_gpu*1000:.1f} ms")
print(f"Speedup: {t_cpu/t_gpu:.2f}x")
