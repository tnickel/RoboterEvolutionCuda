import sys
import pygame
from config.config_manager import SimConfig
from ai.hall_of_fame import HallOfFame
from ai.neat_ai import run_neat_training

if __name__ == '__main__':
    pygame.init()
    config = SimConfig.load()
    hall = HallOfFame()
    screen = None # Headless screen is handled internally but passing None might crash pygame functions, wait, in neat_ai.py screen is checked `if not training_mode and screen is not None`
    # We still need a dummy display for graph rendering perhaps
    screen = pygame.display.set_mode((100, 100), pygame.HIDDEN)
    
    # Run only 3 generations!
    # I can't easily limit generations here without modifying neat_ai.py, but I can just let it run for 10 seconds.
    run_neat_training(config, screen, pygame.time.Clock(), hall, None, mode='train')
