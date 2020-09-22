"""Implementation of genetic algorithm components.

TODO(mcbln): extended docstring.

Fitness
-------
TODO(neilmcb): this is still tbd.
"""

class Population:
    """Population of prediction 'genes'.

    This module implements a population of 'genes' representing PL Killer
    predictions. Each gene consists of a sequence of 'nucleoteams', sampled
    from the group of teams made available by the population.

    A population evolves by breeding and mutating its genes. Breeding
    stochastically selects the fittest genes and splices them together to
    create a new generation. Mutating alters nucleoteams in the population's
    genes at random.
    """
    def __init__(self):
        raise NotImplementedError

    def breed(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError
