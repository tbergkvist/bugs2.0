import numpy as np
from bug import Bug
from food import Food
from brain import Brain


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
        self.bugsave = self.bugs.copy()

    def random_position(self):
        return np.random.randint(0, self.winsize, size=2)

    def update(self):
        self.handle_collisions()
        self.handle_death()
        self.perform_actions()
        self.spawn_food()

    def perform_actions(self):
        for bug in self.bugs:
            bug.percept(self.foods)
            bug.move(bug.decide(), self.winsize)
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
            bug.draw_vectors(surface)
        for food in self.foods:
            food.draw(surface, food_color)

    def is_finished(self):
        if len(self.bugs) <= 1:
            return True
        if len(self.foods) <= 1:
            return True
        if max([bug.size for bug in self.bugs]) >= self.winsize[0]:
            return True
        return False

    def evaluate_bugs(self):
        best_bugs = sorted(self.bugsave, key=lambda x: x.score, reverse=True)
        print("Best score: ", best_bugs[0].score)
        return best_bugs[:len(best_bugs) // 2]

    def crossover(self, parent1, parent2):
        parent1_weights = parent1.brain.state_dict()
        parent2_weights = parent2.brain.state_dict()

        child_brain = Brain(parent1.brain.model[0].in_features)

        child_weights = {}
        for key in parent1_weights:
            child_weights[key] = (parent1_weights[key] + parent2_weights[key]) / 2

        child_brain.load_state_dict(child_weights)

        child = Bug(self.random_position(), self.bugsize, self.bugsight)
        child.brain = child_brain

        return child

    def evolve(self):
        print("Evolving!")
        best_bugs = self.evaluate_bugs()

        new_generation = []
        for i in range(0, len(best_bugs), 2):
            if i + 1 < len(best_bugs):
                parent1 = best_bugs[i]
                parent2 = best_bugs[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
                new_generation.extend([child1, child2])

        return new_generation

    def set_population(self, population):
        if population:
            self.bugs = population
            self.bugsave = self.bugs.copy()
            print("Using saved population!")
