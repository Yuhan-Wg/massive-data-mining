"""
Park-Chen-Yu(PCY) Algorithm:
    Similar with A-Priori Algorithm, except:
        Pass1:
            (-> Same as Pass1 in A-Priori Algorithm)
            -> Keep a hash table with many buckets.
            -> Hash item pairs into buckets.
            -> Count occurrences of pairs in each bucket.
            -> Filter out all frequent buckets(represented in countmap).
        Pass2:
            -> The candidate pair should be in a frequent bucket.

------------------------------

Multi-Hash Algorithm:
    Similar to PCY Algorithm, but use several independent hash functions in Pass1.

This algorithm is implemented on single machine, with pure python.
"""
from datming.freq_itemset.apriori import APriori
from datming.utils import hash2vector
from typing import Iterable, List, Hashable, Dict, FrozenSet
from numpy.random import randint


class MultiHashPCY(APriori):
    def __init__(self, support: int, list_n_bucket: List[int],
                 max_set_size: int=float("inf"), seed: int=None):
        super().__init__(support=support, max_set_size=max_set_size)
        self.__list_n_bucket = list_n_bucket
        self.__n_hash = len(list_n_bucket)
        self.__seed = seed if isinstance(seed, int) else randint(0, 2**32-1)

    def predict(self, data: Iterable[List[Hashable]], **kwargs) -> Dict[int, Dict[FrozenSet, int]]:
        hash_maps = [
            [0 for _ in range(i)]
            for i in self.__list_n_bucket
        ]
        return super().predict(data, hash_maps=hash_maps, list_n_buckets=self.__list_n_bucket)

    def _first_pass_count_singletons(self, data: Iterable[List[Hashable]],
                                     **kwargs) -> (Dict[FrozenSet, int], Dict):
        frequent_items, kwargs = super()._first_pass_count_singletons(data, **kwargs)
        hash_maps = kwargs.get("hash_maps")
        bitmaps = [0 for _ in range(self.__n_hash)]
        for i, hash_map in enumerate(hash_maps):
            for idx, val in enumerate(hash_map):
                bitmaps[i] |= (1 << idx if val >= self._support else 0)
        kwargs["bitmaps"] = bitmaps
        del kwargs["hash_maps"]
        return frequent_items, kwargs

    @staticmethod
    def _loop_count_singleton(bucket: List[Hashable],
                              count_map: Dict[Hashable, int],
                              **kwargs) -> (Dict[Hashable, int], Dict):
        count_map, kwargs = APriori._loop_count_singleton(bucket, count_map, **kwargs)

        hash_maps = kwargs.get("hash_maps")
        list_n_buckets = kwargs.get("list_n_buckets")

        for idx, itemA in enumerate(bucket[:-1]):
            for itemB in bucket[idx+1:]:
                hash_codes = hash2vector(frozenset({itemA, itemB}),
                                         length=len(list_n_buckets))
                for i, code in enumerate(hash_codes):
                    hash_maps[i][code % list_n_buckets[i]] += 1
        kwargs["hash_maps"] = hash_maps
        return count_map, kwargs

    @staticmethod
    def _is_pass_the_constraint(item_set: FrozenSet, **kwargs) -> bool:
        list_n_buckets = kwargs["list_n_buckets"]
        bitmaps = kwargs["bitmaps"]

        for i, hash_code in enumerate(hash2vector(item_set,
                                                  length=len(list_n_buckets))):
            if ((bitmaps[i] >> (hash_code % list_n_buckets[i])) & 1) == 0:
                return False
        return True


if __name__ == '__main__':
    pcy = MultiHashPCY(1, [2,3])
    result = pcy.predict([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]])
    [print(i, result[i]) for i in result]