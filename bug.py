import numpy as np
import pygame as pg
from brain import Brain


class Bug:
    def __init__(self, pos, size, sight):
        self.pos = np.array(pos, dtype=float)
        self.size = size
        self.sight = sight
        self.color = (0, 0, 0)
        self.memory = None
        self.brain = Brain(sight * 2)
        self.score = 0

    def percept(self, foods):
        food_vectors = np.array([food.pos - self.pos for food in foods])
        food_vectors = food_vectors[np.argsort(np.linalg.norm(food_vectors, axis=1))][0:self.sight]
        self.memory = food_vectors

    def decide(self):
        x, y = self.brain.decide(self.memory.flatten())
        return np.array([-1 if x.item() < 0.5 else 1, -1 if y.item() < 0.5 else 1])

    def move(self, direction, winsize):
        self.pos = np.clip(self.pos + direction, [0, 0], winsize)

    def eat(self):
        self.size += 0.1
        self.score += 1

    def get_hungry(self, amount):
        self.size -= amount

    def draw(self, surface):
        pg.draw.circle(surface, self.color, self.pos.astype(int), self.size)

    def draw_vectors(self, surface):
        if self.memory is None:
            return
        for vector in self.memory:
            pg.draw.line(surface, self.color, self.pos, self.pos + vector)
