"""
This module is to find frequent item-sets with given support value.
    -> The input file is like:
    {
    itemA, itemB, ...,
    itemA, itemC, ...,
    ...
    }
    Each line is a bucket

    -> The output result is in a dict, which is like:
    {
        the length of item-set :
            {
                frequent item-set: count
            }
    }

    The item-set is presented in frozenset.
"""

from .apriori import APriori
from .pcy import MultiHashPCY
from .random_sampling import RandomSampling
from .toivonen import Toivonen
from .son import SON
from .fpgrowth import FPGrowth
from .eclat import Eclat

__all__ = [
    "APriori", "MultiHashPCY", "RandomSampling",
    "Toivonen", "SON", "FPGrowth", "Eclat"
]


class FrequentItemSet(object):
    pass