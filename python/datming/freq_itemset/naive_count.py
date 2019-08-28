"""
Naive counting pairs with methods:
    1. Triangle Matrix:
        Keep a matrix, where (row i,col j) maintains the counts of item pair (i,j). The number of items is n.
        -> Keep pair counts in the order: {0,1}{0,2}{0,3},...{0,n-1}{1,2},....
        -> The count of the pair {i,j},(i<j) is at position: i*(2n-i-1)/2+j-i-1
        -> The total memory cost is 4*n^2/2 Bytes

    2. Triples:
        Keep triple (i, j, count) to count item paire (i,j).
        -> The total memory cost is 12 * number of pairs with count>0

This two algorithm is implemented with pure python.
"""


class TriangleMatrixCount(object):
    raise NotImplementedError


class TripleCount(object):
    raise NotImplementedError
