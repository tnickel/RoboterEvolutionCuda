import sys
import os
import copy
import cupy as cp
import numpy as np
import neat
import pygame
from config.config_manager import SimConfig
from core.entities import Collector, Hunter, Battery
from core.world import World
from core.cuda_sim_mega import CudaSimulation
from core.cuda_net import CudaBatchedNetwork

def setup_dummy_networks(config, n_col, n_hun):
    # Dummy genome objects
    class DummyGenome:
        def __init__(self, key):
            self.key = key
            self.fitness = 0
            self.nodes = {}
            self.connections = {}
            
    col_genomes = [DummyGenome(i) for i in range(n_col)]
    hun_genomes = [DummyGenome(i) for i in range(n_hun)]
    
    from ai.neat_ai import _load_neat_config
    
    num_inputs = config.sensor_ray_count * 5 + 2
    c_config = _load_neat_config("config-collector.txt", n_col, num_inputs)
    h_config = _load_neat_config("config-hunter.txt", n_hun, num_inputs)
                           
    c_pop = neat.Population(c_config)
    h_pop = neat.Population(h_config)
    
    # Grab genomes from initial population
    c_genomes = list(c_pop.population.values())[:n_col]
    h_genomes = list(h_pop.population.values())[:n_hun]
    
    from ai.neat_ai import FastNetwork
    c_fast_nets = [FastNetwork(neat.nn.FeedForwardNetwork.create(g, c_config)) for g in c_genomes]
    h_fast_nets = [FastNetwork(neat.nn.FeedForwardNetwork.create(g, h_config)) for g in h_genomes]
    
    c_net = CudaBatchedNetwork(c_fast_nets)
    h_net = CudaBatchedNetwork(h_fast_nets)
    return c_net, h_net

def run_parity_test():
    print("=== Mega-Kernel Parity Test ===")
    config = SimConfig.load()
    
    n_col = 10
    n_hun = 2
    n_bats = 5
    
    world = World(config)
    batteries = world.batteries
    walls = world.walls
    obstacles = world.obstacles
    
    collectors = [Collector(100.0, 100.0, config) for _ in range(n_col)]
    hunters = [Hunter(200.0, 200.0, config) for _ in range(n_hun)]
    
    c_net, h_net = setup_dummy_networks(config, n_col, n_hun)
    
    sim = CudaSimulation(collectors, hunters, batteries, walls, config, c_net, h_net, obstacles)
    
    # --- Speichere Original-Zustand ---
    orig_rx = sim.d_r_x.copy()
    orig_ry = sim.d_r_y.copy()
    orig_rangle = sim.d_r_angle.copy()
    orig_ralive = sim.d_r_alive.copy()
    orig_renergy = sim.d_r_energy.copy()
    orig_rfitness = sim.d_r_fitness.copy()
    orig_bx = sim.d_b_x.copy()
    orig_by = sim.d_b_y.copy()
    orig_bactive = sim.d_b_active.copy()
    orig_btimer = sim.d_b_timer.copy()
    orig_rndcounter = sim.d_rnd_counter.copy()
    
    orig_prev_hunter_dists = sim.d_prev_hunter_dists.copy()
    orig_sensor_inputs = sim.d_sensor_inputs.copy()
    orig_motor_outputs = sim.d_motor_outputs.copy()
    orig_stats_kills = sim.d_stats_kills.copy()
    orig_stats_bats = sim.d_stats_bats.copy()
    orig_r_eaten = sim.d_r_eaten.copy()
    orig_indiv_kills = sim.d_indiv_kills.copy()
    orig_indiv_bats = sim.d_indiv_bats.copy()
    
    steps = 100
    print(f"Run {steps} steps with CLASSIC generation...")
    
    res1 = sim.run_generation(steps)
    class_fitness_c = res1[0]
    class_fitness_h = res1[1]
    
    class_rx = sim.d_r_x.get()
    class_ry = sim.d_r_y.get()
    
    print("Classic run finished. Fitness[0]:", class_fitness_c[0])
    
    # --- Reset Zustand ---
    sim.d_r_x = orig_rx.copy()
    sim.d_r_y = orig_ry.copy()
    sim.d_r_angle = orig_rangle.copy()
    sim.d_r_alive = orig_ralive.copy()
    sim.d_r_energy = orig_renergy.copy()
    sim.d_r_fitness = orig_rfitness.copy()
    sim.d_b_x = orig_bx.copy()
    sim.d_b_y = orig_by.copy()
    sim.d_b_active = orig_bactive.copy()
    sim.d_b_timer = orig_btimer.copy()
    sim.d_rnd_counter = orig_rndcounter.copy()
    
    sim.d_prev_hunter_dists = orig_prev_hunter_dists.copy()
    sim.d_sensor_inputs = orig_sensor_inputs.copy()
    sim.d_motor_outputs = orig_motor_outputs.copy()
    sim.d_stats_kills = orig_stats_kills.copy()
    sim.d_stats_bats = orig_stats_bats.copy()
    sim.d_r_eaten = orig_r_eaten.copy()
    sim.d_indiv_kills = orig_indiv_kills.copy()
    sim.d_indiv_bats = orig_indiv_bats.copy()
    
    print("Resetting state to original...")
    
    if hasattr(sim, 'run_generation_mega'):
        print(f"Run {steps} steps with MEGA generation...")
        res2 = sim.run_generation_mega(steps)
        mega_fitness_c = res2[0]
        mega_fitness_h = res2[1]
        
        mega_rx = sim.d_r_x.get()
        mega_ry = sim.d_r_y.get()
        
        diff_rx = np.max(np.abs(class_rx - mega_rx))
        diff_ry = np.max(np.abs(class_ry - mega_ry))
        diff_fit = np.max(np.abs(class_fitness_c - mega_fitness_c))
        
        print(f"Classic c_fits: {class_fitness_c[:5]}")
        print(f"Mega c_fits: {mega_fitness_c[:5]}")
        print(f"Classic h_fits: {class_fitness_h[:5]}")
        print(f"Mega h_fits: {mega_fitness_h[:5]}")
        
        print(f"Max Diff X: {diff_rx}")
        print(f"Max Diff Y: {diff_ry}")
        print(f"Max Diff Fitness: {diff_fit}")
        
        if diff_rx < 1e-3 and diff_ry < 1e-3 and diff_fit < 1e-3:
            print("\n>>> TEST PASS: Behavioral Parity 100% achieved! <<<")
        else:
            print("\n>>> TEST FAILED: Divergence detected! <<<")
    else:
        print("run_generation_mega is not implemented yet. Test script ready.")

if __name__ == '__main__':
    pygame.init()
    run_parity_test()
