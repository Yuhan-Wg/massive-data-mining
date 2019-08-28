"""
Random Sampling:
    Random sample the input data and find out frequent itemsets (lower the threshold).

This algorithm is implemented with pure python.
"""
import random
from datming.freq_itemset.apriori import APriori


class RandomSampling(object):
    def __init__(self,support, sampling_rate, random_seed=None):
        self._support = support
        self._sampling_rate = sampling_rate
        self._random_seed = random_seed
        self._sampled_iterable = None

    def iter_count(self, iterable):
        random.seed(self._random_seed)
        self._sampled_iterable = [bucket for bucket in iterable if random.random() <= self._sampling_rate]
        apriori = APriori(support=self._support * self._sampling_rate)
        yield from apriori.iter_count(iterable = self._sampled_iterable)

    def count(self, iterable, max_size=float("inf")):
        return APriori.count(self, iterable, max_size)

    def get_sampled_iterable(self):
        return self._sampled_iterable


if __name__ == '__main__':
    random_sampling = RandomSampling(1, 0.9,random_seed=None)
    for s in random_sampling.iter_count([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]]):
        print(s)