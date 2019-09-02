"""
Random Sampling:
    Random sample the input data and find out frequent itemsets (lower the threshold).

"""
import random
import numpy as np
from datming.freq_itemset.apriori import APriori
from typing import Iterable, List, Hashable


class RandomSampling(object):
    def __init__(self, support: int,
                 sampling_rate: float,
                 max_set_size: int=float("inf"), random_seed=None):
        self.__support = support
        self.__sampling_rate = sampling_rate
        self.__random_seed = random_seed
        self.__max_set_size = max_set_size

    def predict(self, data: Iterable[List[Hashable]]):
        random.seed(self.__random_seed)
        sampled_data = [bucket for bucket in data
                        if random.random() <= self.__sampling_rate]
        apriori = APriori(support=int(self.__support * self.__sampling_rate),
                          max_set_size=self.__max_set_size)
        return apriori.predict(data=sampled_data)

    def get_sampled_data(self, data):
        random.seed(self.__random_seed)
        sampled_data = [bucket for bucket in data
                        if random.random() <= self.__sampling_rate]
        return sampled_data


if __name__ == '__main__':
    random_sampling = RandomSampling(2, 0.6, random_seed=None)
    result = random_sampling.predict([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]])
    [print(i, result[i]) for i in result]

