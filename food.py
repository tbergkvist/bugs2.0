import numpy as np
import pygame as pg

class Food:
    def __init__(self, pos, size):
        self.pos = np.array(pos, dtype=float)
        self.size = size

    def draw(self, surface, color):
        pg.draw.circle(surface, color, self.pos.astype(int), self.size)
