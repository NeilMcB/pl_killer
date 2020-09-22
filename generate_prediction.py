# -*- coding utf-8 -*-
"""PL Killer prediction generator.

This script generates a sequence of teams which satisfies the rules of 'PL
Killer' by considering both gameweek odds (short term value) and fixture
difficulty rating (long term value).

Example
-------
TODO(neilmcb): an example should go here, of the form::

    $ python generate_prediction.py argv_1 argv_2

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
