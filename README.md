# Premier League Killer

PL Killer prediction generator.

This script generates a sequence of teams which satisfies the rules of 'PL
Killer' by considering both gameweek odds (short term value) and fixture
difficulty rating (long term value).

It isn't very good :(
* Genetic diversity dies it pretty quickly.
* Performance of final result is typically worse than the best intiial
permutation.

Potential changes:
* Always room to improve fitness scoring.
* Parent sampling - e.g. tournament selection.
* Carry best gene(s) through without sampling.

## ToDo
* Refactor to package?.
* Implement mutations.
* Refactor plotting/tracking to package.

## Example
To run the algorithm with a population size of 1,000 for 500 generations:
```
python generate_prediction.py 1000 500
```

## PL Killer
The aim of PL Killer is to select a team each Premier League gameweek who
will their fixture, subject to the constraint that no team can be selected
more than once.

## Genetic Algorithm
A form of optimisation algorithm inspired by the "survival of the fittest"
in nature. Genetic algorithms first randomly generate a population of
"genes" - sequences of teams in this case - then stochastically sample a
subset of these to serve as "parents" based on their fitness; a measure of
the quality of a given genes teams. Parents are then paired up and spliced
together at a random index in the sequence to give a new generation of
child genes. The process of sampling parents and splicing can then just be
repeated here, or the children can go through a process of mutation where,
at random, teams are swapped around in a sample of genes.

## Fixture Difficulty Rating (FDR)
FDR is an algorithmically generated score for the difficulty of an opponent
for a given team. It takes into account factors such as historical results,
form and various Opta performance metrics. It is used as a proxy for odds
for long term planning as odds are only available a couple of gameweeks in
advance.
