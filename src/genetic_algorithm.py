"""Genetic Algorithm helper functions.
"""
import numpy as np
from numpy import ndarray

def fitness_function(gene: ndarray, fitness_lookup: ndarray) -> float:
    """Assign a fitness value to a given sequence of teams.

    TODO(neilmcb): vectorise?

    Parameters
    ----------
    gene : (N teams,) ndarray
        Sequence of nucleoteams to score.
    fitness_lookup : (N teams, N teams) ndarray
        Lookup table of fitness scores for teams in given positions in
        the sequence. Index order should be [sequence position, team].

    Returns
    -------
        float
    """
    sequence_indexes = np.arange(gene.shape[0])
    fitness = fitness_lookup[sequence_indexes, gene].sum()

    # Sort and check for matching subsequent values
    sorted_gene = np.sort(gene, axis=None)
    if sorted_gene[:-1][sorted_gene[1:] == sorted_gene[:-1]].shape != (0,):
        fitness = 0

    return fitness


def sample_parents(population: ndarray, population_fitnesses: ndarray, 
                   rng: np.random.Generator) -> ndarray:
    """Stochastically select parents from the population by fitness.

    TODO(neilmcb): manage errors properly.

    Parameters
    ----------
    population : (Population size, N teams) ndarray
        The current generation of genes.
    population_fitnesses : (Population size,) ndarray
        Vector of fitnesses for each gene in the current generation.
    rng : numpy.random.Generator
        Numpy random number generator instance.

    Returns
    -------
        (Population size, 2, N teams) ndarray
    """
    assert population.shape[0] == population_fitnesses.shape[0]

    positive_population_fitness = population_fitnesses + 0.0001  # To avoid NaNs
    parent_sample_probabilities = positive_population_fitness / positive_population_fitness.sum()
    parent_indices_a, parent_indices_b = rng \
        .choice(population.shape[0], 2 * population.shape[0], p=parent_sample_probabilities) \
        .reshape((2, -1))

    return np.stack((population[parent_indices_a], population[parent_indices_b]), axis=1)


def splice_parents(parents: ndarray, indices: ndarray) -> ndarray:
    """Splice parent genes together to produce a child.

    TODO(neilmcb): manage errors properly.
    TODO(neilmcb): vectorise?

    Parameters
    ----------
    parents : (Population size, 2, N teams) ndarray
        Array of paired parent genes.
    indices : (Population size,) ndarray
        Vector of indices with which to splice. Nucleoteams up to this
        point are taken from the first parent and beyond this point
        from the second parent.

    Returns
    -------
        (Population size, N teams) ndarray
    """
    assert parents.shape[1] == 2
    assert parents.shape[0] == indices.shape[0]

    return np.stack(
        [np.concatenate((p_a[:idx], p_b[idx:])) for idx, (p_a, p_b) in zip(indices, parents)]
    )
