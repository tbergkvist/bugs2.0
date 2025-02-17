import numpy as np
import pygame as pg

class Bug:
    def __init__(self, pos, size, sight):
        self.pos = np.array(pos, dtype=float)
        self.size = size
        self.sight = sight
        self.color = (0, 0, 0)

    def decide(self, bugs, foods):
        return np.random.uniform(-1, 1, 2)

    def move(self, direction, winsize):
        self.pos = np.clip(self.pos + direction, [0, 0], winsize)

    def eat(self):
        self.size += 1

    def get_hungry(self, amount):
        self.size -= amount

    def draw(self, surface):
        pg.draw.circle(surface, self.color, self.pos.astype(int), self.size)
