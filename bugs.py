import numpy as np
import pygame as pg
import time
from simulation import Simulation


WINSIZE = np.array([640, 480])
NUMBUGS = 50
NUMFOOD = 200
BUGSIZE = 10
FOODSIZE = 3
HUNGER_RATE = 0.005
BUG_SIGHT = 1
BUG_SEE_BUG = False
NUMBER_OF_GENERATIONS = 10000
HEADLESS = False
TICK_RATE = 1000


def main_loop(headless=False):
    sim = Simulation(NUMBUGS, NUMFOOD, BUGSIZE, FOODSIZE, HUNGER_RATE, BUG_SIGHT, BUG_SEE_BUG, WINSIZE)
    
    if not headless:
        pg.init()
        screen = pg.display.set_mode(WINSIZE.tolist())
        pg.display.set_caption("Bugs Simulation")
        background_color = (100, 255, 150)
        food_color = (20, 20, 255)
        screen.fill(background_color)
    
    clock = pg.time.Clock()
    running = True
    
    while running:
        if not headless:
            screen.fill(background_color)
            sim.draw(screen, food_color)
            pg.display.update()
        
        sim.update()

        if not headless:
            for event in pg.event.get():
                if event.type in (pg.QUIT, pg.KEYUP) and (event.type == pg.QUIT or event.key == pg.K_ESCAPE):
                    running = False
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    time.sleep(10)
        
        if sim.is_finished():
            running = False
        
        clock.tick(TICK_RATE)

    pg.quit()
    return pg.time.get_ticks()

def run_one_sim(headless=False):
    """Runs a single simulation instance."""
    ticks = main_loop(headless)
    print(f'Ticks: {ticks}')



for i in range(NUMBER_OF_GENERATIONS):
    print(f"Generation {i}")
    run_one_sim(headless=HEADLESS)
