"""
A-Priori Algorithm:
    Two-pass(while counting pairs) approach to reduce the memory cost.
    -> The key idea is to use monotonicity to exclude infrequent item pairs before the second-pass counting.
    -> Monotonicity: if the item set A is frequent only if all subsets of A are frequent.

    Pass1:
        -> Count occurrence of each item.
        -> Filter out all frequent items.
    Pass2:
        -> Renumber the items (if using triangle matrix to count)/ Use triple representation to count.
        -> Construct possible item pairs.
        -> Count item pairs.
        -> Filter out all frequent item pairs.
    Pass3:
        -> Construct possible item triples.
        -> Count triples.
        -> Filter out all frequent triples.
    Pass4 or more:
        -> ....

>> from mmds.frequent_itemset import APriori
>> apriori = APriori(support = 100)
>> frequent_itemsets = apriori.count(iterable<list<item>>)
"""
from collections import defaultdict, namedtuple
from typing import Iterable, List, Hashable, Dict, Set, FrozenSet, Callable


FreqItemset = namedtuple("FreqItemset", ['items', "freq"])


class APriori(object):
    """
    Limited-pass implementation of A-Priori Algorithm.
    """
    def __init__(self, support: int, max_set_size: int=float("inf")):
        """
        :param support: The threshold of frequent itemset
        :param max_set_size: Maximum size of Item-sets.
        """
        self._support = support
        self._max_set_size = max_set_size
        self._freq_itemsets = None

    def predict(self, data: Iterable[List[Hashable]], **kwargs):
        """
        :return: Dict[set_length, Dict[frequent_item_sets, support]]
        """
        freq_itemsets = list()
        for i, item_sets in enumerate(self._iter_predict(data, **kwargs)):
            freq_itemsets.extend([
                FreqItemset(items=items, freq=freq) for items, freq in item_sets.items()
            ])
            if i + 1 >= self._max_set_size:
                break
        self._freq_itemsets = freq_itemsets
        return self

    def frequent_itemsets(self):
        return self._freq_itemsets

    def _iter_predict(self, data: Iterable[List[Hashable]], **kwargs) \
            -> Iterable[Dict[FrozenSet[Hashable], int]]:
        """
        Output frequent item-sets.
        In one iteration, scan all the buckets for one time.
        It is to reduce unnecessary computations if one just wants frequent item-sets in small size.

        :param data: the iterable instance.
            = The instance with implemented __iter__(), like List<List<item>>
        :param kwargs: Other parameters for future sub-classing.
        :return: _frequent_itemsets : the frequency of frequent item-sets
            = dict<length of frequent set, dict<frequent set, frequency>>
        """
        # The first pass to count frequent singletons.
        item_sets, kwargs = \
            self._first_pass_count_singletons(data=data, **kwargs)
        if len(item_sets) > 0:
            yield item_sets
        else:
            return

        # The second pass to count frequent pairs.
        item_sets, kwargs = \
            self._second_pass_count_pairs(data=data,
                                          frequent_items=item_sets,
                                          **kwargs)
        if len(item_sets) > 0:
            yield item_sets
        else:
            return

        set_length = 3
        while True:
            item_sets, _ = \
                self._other_passes_count_triples_or_more(data=data,
                                                         frequent_prev_set=item_sets,
                                                         set_length=set_length,
                                                         **kwargs)
            if len(item_sets) > 0:
                yield item_sets
            else:
                break
            set_length += 1

    def count(self, iterable, max_size=float("inf")):
        """
        Return the frequent item-sets directly with the given max size of frequent item-set.
        Not iterative.

        :param iterable: the iterable instance.
            -> The instance with implemented __iter__(), like <list<item>>
        :param max_size: the maximum length of frequent item-set
        :return= dict<length of frequent set, dict<frequent set, frequency>>
        """
        for size, _ in enumerate(self.predict(iterable)):
            if size + 1 >= max_size:
                break
        return self._frequent_itemsets

    def _first_pass_count_singletons(self, data: Iterable[List[Hashable]],
                                     **kwargs) -> (Dict[FrozenSet, int], Dict):
        """
        One pass to count items in buckets.
        This process will scan all buckets and read the whole file from disk.
        """
        count_map = defaultdict(int)
        for bucket in data:
            dict_count, kwargs = self._loop_count_singleton(bucket=bucket,
                                                            count_map=count_map,
                                                            **kwargs)
        frequent_items = {
            frozenset({key}): val for key, val in count_map.items() if val >= self._support
        }
        return frequent_items, kwargs

    def _second_pass_count_pairs(self, data: Iterable[List[Hashable]],
                                 frequent_items: Dict[FrozenSet, int], **kwargs) -> (Dict[FrozenSet, int], Dict):
        """
        One pass to count candidate pairs in the buckets.
        This process will scan all buckets and read the whole file from disk.
        """
        count_map = defaultdict(int)
        for bucket in data:
            count_map, kwargs = self._loop_count_pair(bucket=bucket,
                                                      frequent_items=frequent_items,
                                                      count_map=count_map,
                                                      is_pass_the_constraint=self._is_pass_the_constraint,
                                                      **kwargs)

        frequent_pairs = {
            key: val for key, val in count_map.items() if val >= self._support
        }
        return frequent_pairs, kwargs

    def _other_passes_count_triples_or_more(self, data: Iterable[List[Hashable]],
                                            frequent_prev_set: Dict[FrozenSet, int],
                                            set_length: int, **kwargs) -> (Dict[FrozenSet, int], Dict):
        """
        One pass to count triples, or bigger item-sets in the buckets.
        All the subsets of the candidate item-set should be frequent. Candidate item-sets will be derived from present
            frequent item-sets.
        This process will scan all buckets and read the whole file from disk.
        """
        candidates = self.find_candidates(frequent_set=frequent_prev_set,
                                          set_length=set_length)

        count_map = defaultdict(int)
        for bucket in data:
            count_map, kwargs = self._loop_count_triples_or_more(bucket=bucket,
                                                                 candidates=candidates,
                                                                 count_map=count_map, **kwargs)

        frequent_next_set = {
            key: val for key, val in count_map.items() if val >= self._support
        }
        return frequent_next_set, kwargs

    @staticmethod
    def _loop_count_singleton(bucket: List[Hashable], count_map: Dict[Hashable, int],
                              **kwargs) -> (Dict[Hashable, int], Dict):
        """
        One loop to count singleton in one bucket.

        :param count_map: dict<item, count>. The map used to count singles.
        :param bucket: list<item>
        :param kwargs: other variables
        :return: count_map, params
        """
        for item in bucket:
            count_map[item] += 1
        return count_map, kwargs

    @staticmethod
    def _loop_count_pair(bucket: List[Hashable],
                         count_map: Dict[FrozenSet, int],
                         frequent_items: Dict[FrozenSet, int],
                         is_pass_the_constraint: Callable, **kwargs) -> (Dict[FrozenSet, int], Dict):
        """
        One loop to count candidate pairs in one bucket.
        Items in the candidate pair should be frequent.
        """
        filtered_bucket = list({
            item for item in bucket
            if frozenset({item}) in frequent_items
        })

        if len(filtered_bucket) < 2:
            return count_map, kwargs

        for idx, itemA in enumerate(filtered_bucket[:-1]):
            for itemB in filtered_bucket[idx+1:]:
                fs = frozenset({itemA, itemB})
                if is_pass_the_constraint(fs, **kwargs):
                    count_map[fs] += 1
        return count_map, kwargs

    @staticmethod
    def _loop_count_triples_or_more(bucket: List[Hashable],
                                    candidates: Set[FrozenSet],
                                    count_map: Dict[FrozenSet, int], **kwargs) -> (Dict[FrozenSet, int], Dict):
        """
        One loop to count candidate item-sets in one bucket.
        Check all candidate one by one.
        """
        set_bucket = set(bucket)
        for candidate in candidates:
            if candidate.issubset(set_bucket):
                # if the candidate is a subset of bucket,
                # then the bucket contains the item-set
                count_map[candidate] += 1
        return count_map, kwargs

    @staticmethod
    def find_candidates(frequent_set: Dict[FrozenSet, int],
                        set_length: int) -> Set[FrozenSet]:
        """
        To find all candidate supersets from frequent sets.
        A candidate item-set require: all its immediate subsets are frequent.
        For a L-size candidate item-set, all its (L-1)-size subsets are frequent. There are L such different subsets,
            so there will be L(L-1) different combinations (consider the order). The union of subsets in each
            combination is this L-size item-set. Therefore, in the below loops, the candidate item-set will be counted
            for L(L-1) times.

        :param frequent_set: set<frozenset<items>>
        :param set_length: the length of candidate set
        :return: set<frozenset<items>>
        """
        candidates = defaultdict(int)
        for setA in frequent_set:
            for setB in frequent_set:
                union_set = setA | setB
                if len(union_set) == set_length:
                    candidates[union_set] += 1
        return set(
            key for key, val in candidates.items() if val >= set_length*(set_length - 1)
        )

    @staticmethod
    def _is_pass_the_constraint(item_set: FrozenSet, **kwargs) -> bool:
        """
        To check if the hashable is qualified to be a candidate.
        Not used in Apriori Algorithm, but used for other algorithms.
        It is for future subclassing.
        :param hashable: the item-set, should be hashable.
        :return = bool: {True: can be a qualified candidate,
                         False: not a qualified candidate}
        """
        return True

if __name__ == '__main__':
    apriori = APriori(2)
    result = apriori.predict([
        [1, 2, 3],
        [2, 3],
        [1, 4],
        [2, 4],
        [1, 2, 3, 4]
    ])
    [print(items) for items in result.frequent_itemsets()]

