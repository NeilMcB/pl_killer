# -*- coding utf-8 -*-
"""PL Killer prediction generator.

This script generates a sequence of teams which satisfies the rules of 'PL
Killer' by considering both gameweek odds (short term value) and fixture
difficulty rating (long term value).

Example
-------
TODO(neilmcb): an example should go here, of the form::

    $ python generate_prediction.py argv_1 argv_2

Genetic Algorithm
-----------------
TODO(neilmcb): explain.

PL Killer
---------
The aim of PL Killer is to select a team each Premier League gameweek who
will their fixture, subject to the constrain that no team can be selected
more than once.

Fixture Difficulty Rating (FDR)
-------------------------------
FDR is an algorithmically generated score for the difficulty of an opponent
for a given team. It takes into account factors such as historical results,
form and various Opta performance metrics. It is used as a proxy for odds
for long term planning as odds are only available a couple of gameweeks in
advance.

TODO(neilmcb): implement in a single script, refactor afterwards.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
plt.style.use('neilmcblane.mplstyle')

## -- Setup --

rng = default_rng()

# Represent teams as ints to save space
teams = np.arange(20)
population_size = 100_000
n_generations = 1_000


df_fdr = pd.read_csv('fdr.csv', index_col='gw')
# Assign teams missing fixtures with long odds
df_odds = pd.read_csv('odds.csv', index_col='gw').fillna(10)

# Use odds for fitness if available
fixture_scores = np.vstack([df_odds.values, df_fdr.drop(index=df_odds.index).values])
fixture_scores = np.power(1 / fixture_scores.T, np.linspace(2, 0, len(teams)))


def fitness_function(gene, fitness_lookup, duplicate_penalty):
    """Assign a fitness value to a given sequence of teams.

    TODO(neilmcb): vectorise?

    Parameters
    ----------
    gene : ndarray
        Sequence of nucleoteams to score.
    fitness_lookup : ndarray
        Lookup table of fitness scores for teams in given positions in
        the sequence. Index order should be [sequence position, team].
    duplicate_penalty : float
        Penalty to assign to the fitness value of `gene` if it contains
        a given team more than once.

    Returns
    -------
        float
    """
    sequence_indexes = np.arange(gene.shape[0])
    lookup_indexes = np.column_stack([sequence_indexes, gene])
    fitness = fitness_lookup[lookup_indexes, gene].sum()

    # Sort and check for matching subsequent values
    sorted_gene = np.sort(gene, axis=None)
    if sorted_gene[:-1][sorted_gene[1:] == sorted_gene[:-1]].shape == (0,):
        fitness -= duplicate_penalty

    return fitness


## -- Algorithm -- TODO(neilmcb): tidy up, monitor performance
population = np.stack([rng.permutation(teams) for _ in np.arange(population_size)])
population_fitness = np.apply_along_axis(
    lambda gene: fitness_function(gene, fixture_scores, 5), 1, population
)
for _ in range(n_generations):
    parent_sample_probabilities = population_fitness / population_fitness.sum()

    parent_indices = rng.choice(population_size, 2*population_size, p=parent_sample_probabilities)
    parents = np.split(population[parent_indices], population_size)

    splice_indices = rng.choice(len(teams), population_size)

    population = np.stack([np.concatenate((parent_m[:splice_index], parent_f[splice_index:])) for splice_index, (parent_m, parent_f) in zip(splice_indices, parents)])
    population_fitness = np.apply_along_axis(lambda gene: fitness_function(gene, fixture_scores, 5), 1, population)

print(population[np.argmax[population_fitness]])
