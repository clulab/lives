""""Code based on:
https://codereview.stackexchange.com/questions/203468/find-the-intervals-which-have-a-non-empty-intersection-with-a-given-interval
"""

turn = tuple[float,float]

import numpy as np

def find_overlaps(turns:list[turn], query:tuple[turn], index:int) -> list:
    """Find indexes of overlapping turns, excluding current turn

    Parameters
    ----------
    turns: list
        List of tuples of turn starts/ends in seconds
    query: tuple
        Single turn tuple (start[float], end[float])
    index: int
        The index of the query. Used to ignore overlap with query itself.

    Usage
    -----
    for i, k in enumerate(turns):
        print(f"turn {i+1}, {k}")
        print(find_overlaps(turns, k, i))

    Issues
    ------
    Known bug:
        This algorithm will take exact matching boundaries as overlaps.
        This may not be a problem as turns are defined to more than 10 decimals
        For example, these turns would count as overlapping: (0.21,0.30), (0.30,0.40)
        But because we have more decimal places, they do not: (0.210,0.300), (0.301,0.400)

    Tests
    -----
    >>> find_overlaps([(0.20,0.41), (0.21,0.30), (0.28,0.39)], (0.20,0.41), 0)
    [1, 2]
    >>> find_overlaps([(0.20,0.41), (0.21,0.30), (0.28,0.39)], (0.21,0.30), 1)
    [0, 2]
    >>> find_overlaps([(0.20,0.41), (0.21,0.30), (0.28,0.39)], (0.28,0.39), 2)
    [0, 1]
    """
    intervals = np.asarray(turns)
    lower, upper = query
    overlap_indexes =  np.argwhere((lower < intervals[:, 1]) & (intervals[:, 0] < upper))
    res = []
    for i in np.reshape(overlap_indexes, -1):
        if not i == index:
            res.append(i)
    return(res)


if __name__ == "__main__":
    import doctest
    doctest.testmod()