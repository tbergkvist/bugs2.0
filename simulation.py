import numpy as np
from bug import Bug
from food import Food


class Simulation:

    def __init__(self, numbugs, numfood, bugsize, foodsize, hunger_rate, bug_sight, bug_see_bug, winsize):
        self.winsize = np.array(winsize)
        self.bugsize = bugsize
        self.foodsize = foodsize
        self.bugsight = bug_sight
        self.see_bug = bug_see_bug
        self.hunger_rate = hunger_rate
        self.bugs = [Bug(self.random_position(), bugsize, bug_sight) for _ in range(numbugs)]
        self.foods = [Food(self.random_position(), foodsize) for _ in range(numfood)]

    def random_position(self):
        return np.random.randint(0, self.winsize, size=2)

    def update(self):
        self.handle_collisions()
        self.handle_death()
        self.perform_actions()
        self.spawn_food()

    def perform_actions(self):
        for bug in self.bugs:
            bug.move(bug.decide(self.bugs, self.foods), self.winsize)
            bug.get_hungry(self.hunger_rate)

    def handle_collisions(self):
        if not self.foods or not self.bugs:
            return
        bug_positions = np.array([bug.pos for bug in self.bugs])
        bug_sizes = np.array([bug.size for bug in self.bugs])[:, None]
        food_positions = np.array([food.pos for food in self.foods])

        distances = np.linalg.norm(bug_positions[:, None] - food_positions, axis=2)

        collision_indices = np.where(distances < (bug_sizes + self.foodsize))

        food_eaten = set(collision_indices[1])
        self.foods = [food for i, food in enumerate(self.foods) if i not in food_eaten]

        for bug_idx in np.unique(collision_indices[0]):
            self.bugs[bug_idx].eat()

    def handle_death(self):
        self.bugs = [bug for bug in self.bugs if bug.size > 0.5]

    def spawn_food(self):
        if np.random.random() < 0.4:
            self.foods.append(Food(self.random_position(), self.foodsize))

    def draw(self, surface, food_color):
        for bug in self.bugs:
            bug.draw(surface)
        for food in self.foods:
            food.draw(surface, food_color)

    def is_finished(self):
        if len(self.bugs) <= 1:
            return True
        if max([bug.size for bug in self.bugs]) >= self.winsize[0]:
            return True
        return False