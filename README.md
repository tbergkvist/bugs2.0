# Bugs 2.0
This is a new take on my [old bug simulation](https://github.com/tbergkvist/bugs).

## What is this weird shit?
This is a simulation of bugs. They see the nearest k foods, and use their neural networks to decide what direction to go. If they eat, they grow, if they don't eat, they die.

The bugs in the population are evaluated after each generation and the best half gets to multiply and be the starting population of the next generation.

## Results so far
If they see only the nearest food source, they learn quite fast (often within 50 generations), resulting in individuals with the optimal strategy: go towards the food.

## Changes from my old version:
I completely rebuilt the program using numpy and pytorch. This has made the computations a lot faster.