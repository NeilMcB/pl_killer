# -*- coding utf-8 -*-
"""PL Killer prediction generator.

This script generates a sequence of teams which satisfies the rules of 'PL
Killer' by considering both gameweek odds (short term value) and fixture
difficulty rating (long term value).

It isn't very good :(

TODO(neilmcb): refactor to package?.

Example
-------
TODO(neilmcb): an example should go here, of the form:

    $ python generate_prediction.py argv_1 argv_2

PL Killer
---------
The aim of PL Killer is to select a team each Premier League gameweek who
will their fixture, subject to the constrain that no team can be selected
more than once.

Genetic Algorithm
-----------------
A form of optimisation algorithm inspired by the "survival of the fittest"
in nature. Genetic algorithms first randomly generate a population of
"genes" - sequences of teams in this case - then stochastically sample a
subset of these to serve as "parents" based on their fitness; a measure of
the quality of a given genes teams. Parents are then paired up and spliced
together at a random index in the sequence to give a new generation of
child genes. The process of sampling parents and splicing can then just be
repeated here, or the children can go through a process of mutation where,
at random, teams are swapped around in a sample of genes.

Fixture Difficulty Rating (FDR)
-------------------------------
FDR is an algorithmically generated score for the difficulty of an opponent
for a given team. It takes into account factors such as historical results,
form and various Opta performance metrics. It is used as a proxy for odds
for long term planning as odds are only available a couple of gameweeks in
advance.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd

from src.genetic_algorithm import fitness_function, sample_parents, splice_parents 
plt.style.use('neilmcblane.mplstyle')

## -- Setup --

rng = default_rng()

# Represent teams as ints to save space
teams = np.arange(20)
population_size = 100_000
n_generations = 100

df_fdr = pd.read_csv('fdr.csv', index_col='gw')
# Assign teams missing fixtures with long odds
df_odds = pd.read_csv('odds.csv', index_col='gw').fillna(100)

# Use odds for fitness if available
fixture_scores = np.vstack([df_odds.values, df_fdr.drop(index=df_odds.index).values])
fixture_scores = np.power(1 / fixture_scores.T, np.linspace(2, 0, len(teams))).T


fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(fixture_scores)
ax.set_xlabel("Gameweek")
ax.set_ylabel("Team Score")
ax.set_ylim((0, 1))
ax.set_xlim((0, 19))
ax.set_xticks(np.arange(0, 20, 1))
ax.set_xticklabels(np.arange(1, 21, 1))

fig.savefig('assets/fixture_scores.png', dpi=300)

## -- Algorithm -- TODO(neilmcb): tidy up, monitor performance
population = np.stack([rng.permutation(teams) for _ in np.arange(population_size)])
population_fitness = np.apply_along_axis(lambda gene: fitness_function(gene, fixture_scores), 1, population)

min_fitnesses = -1 * np.ones(n_generations)
max_fitnesses = -1 * np.ones(n_generations)
avg_fitnesses = -1 * np.ones(n_generations)
std_fitnesses = -1 * np.ones(n_generations)

for i in range(n_generations):
    if i % 10 == 0:
        print(f'{i:3} of {n_generations} Gene fitness summary:')
        print(f'\t              Best: {population_fitness.max()}')
        print(f'\t             Worst: {population_fitness.min()}')
        print(f'\t           Average: {population_fitness.mean()}')
        print(f'\tStandard deviation: {population_fitness.std()}')

    min_fitnesses[i] = population_fitness.min()
    max_fitnesses[i] = population_fitness.max()
    avg_fitnesses[i] = population_fitness.mean()
    std_fitnesses[i] = population_fitness.std()

    parents = sample_parents(population, population_fitness, rng)
    splice_indices = rng.choice(len(teams), population_size)

    population = splice_parents(parents, splice_indices)
    population_fitness = np.apply_along_axis(
        lambda gene: fitness_function(gene, fixture_scores), 1, population
    )


fig, ax = plt.subplots(1, figsize=(9, 6))

ax.fill_between(
    np.arange(0, n_generations),
    avg_fitnesses + std_fitnesses,
    avg_fitnesses - std_fitnesses,
    label='Standard Deviation',
    color='C0',
    alpha=0.3,
)
ax.plot(min_fitnesses, label='Generation Worst', color='C14')
ax.plot(avg_fitnesses, label='Generation Average', color='C0')
ax.plot(max_fitnesses, label='Generation Best', color='C7')
ax.axhline(max_fitnesses.max(), color='C7', label='All Time Best', ls='--')

ax.set_xlabel("Generation")
ax.set_ylabel("Gene Fitness")
ax.set_xlim((0, n_generations))
ax.set_ylim((0, 12))

ax.legend()

fig.savefig('assets/fitness_progression.png', dpi=300)

print(df_fdr.columns[population[np.argmax(population_fitness)]])
