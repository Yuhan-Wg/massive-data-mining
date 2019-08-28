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

This algorithm is implemented on single machine, with pure python.

>> from mmds.frequent_itemset import APriori
>> apriori = APriori(support = 100)
>> frequent_itemsets = apriori.count(iterable<list<item>>)
"""
from collections import defaultdict


class APriori(object):
    def __init__(self,support, **params):
        """
        :param support: The threshold of frequent itemset
        :param params: Other parameters.

        :var self._support: The support value to identify frequent itemsets
        :var self._frequent_itemsets: THe dict of frequent itemsets, Dict{set_length: set<frequent itemset>}
        """
        self._support = support
        self._frequent_itemsets = defaultdict(dict)
        self.__dict__.update(params)

    def iter_count(self, iterable, **containers):
        """
        To iteratively output frequent item-sets.
        In one iteration, scan all the buckets for one time.
        It is to reduce unnecessary computations if one just wants frequent item-sets in small size.

        :param iterable: the iterable instance.
            = The instance with implemented __iter__(), like <list<item>>
        :return: _frequent_itemsets : the frequency of frequent item-sets
            = dict<length of frequent set, dict<frequent set, frequency>>

        """
        # The first pass to count frequent singletons.
        self._frequent_itemsets[1], containers = \
            self._first_pass_count_singletons(iterable=iterable,
                                              **containers)
        yield self._frequent_itemsets[1]

        # The second pass to count frequent pairs.
        self._frequent_itemsets[2], containers = \
            self._second_pass_count_pairs(iterable=iterable,
                                          frequent_items=self._frequent_itemsets[1],
                                          **containers)
        yield self._frequent_itemsets[2]

        set_length = 3
        while True:
            _frequent_itemsets, _ = \
                self._other_passes_count_triples_or_more(iterable=iterable,
                                                         frequent_prev_set=self._frequent_itemsets[set_length-1],
                                                         set_length=set_length,
                                                         **containers)
            if len(_frequent_itemsets) > 0:
                self._frequent_itemsets[set_length] = _frequent_itemsets
                yield self._frequent_itemsets[set_length]
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
        for size, _ in enumerate(self.iter_count(iterable)):
            if size + 1 >= max_size:
                break
        return self._frequent_itemsets

    def _first_pass_count_singletons(self, iterable, **containers):
        """
        One pass to count items in buckets.
        This process will scan all buckets and read the whole file from disk.

        :param iterable: iterable<list<item>>
        :param container: other variables (for future inherit)
        :return: dict<frozenset<item>, frequency>, containers
        """
        dict_count = defaultdict(int)
        for bucket in iterable:
            dict_count, containers = self._loop_count_single(bucket=self._process_bucket(bucket),
                                                             dict_count=dict_count,
                                                             **containers)

        frequent_items = {
            frozenset({key}): val for key, val in dict_count.items() if val >= self._support
        }
        return frequent_items, containers

    def _second_pass_count_pairs(self, iterable, frequent_items, **containers):
        """
        One pass to count candidate pairs in the buckets.
        This process will scan all buckets and read the whole file from disk.

        :param iterable: iterable<list<item>>
        :param frequent_items: set<frequent item>
        :param containers: Other variables
        :return: dict<frozenset<item, item>, frequency>, params
        """
        dict_count = defaultdict(int)
        for bucket in iterable:
            dict_count, containers = self._loop_count_pair(bucket=self._process_bucket(bucket),
                                                           frequent_items=frequent_items,
                                                           dict_count=dict_count, **containers)

        frequent_pairs = {
            key: val for key, val in dict_count.items() if val >= self._support
        }
        return frequent_pairs, containers

    def _other_passes_count_triples_or_more(self, iterable, frequent_prev_set, set_length, **containers):
        """
        One pass to count triples, or bigger item-sets in the buckets.
        All the subsets of the candidate item-set should be frequent. Candidate item-sets will be derived from present
            frequent item-sets.
        This process will scan all buckets and read the whole file from disk.

        :param iterable: iterable<list<item>>
        :param frequent_set: set<tuple<itemA,itemB,...>>
        :param set_length: The length of target frequent itemset
        :param containers: other variables
        :return: dict<frozenset<items>, frequency>, containers
        """
        candidates = self.find_candidates(frequent_set=frequent_prev_set,
                                          set_length=set_length)

        dict_count = defaultdict(int)
        for bucket in iterable:
            dict_count, containers = self._loop_count_triples_or_more(bucket=self._process_bucket(bucket),
                                                                      candidates=candidates,
                                                                      dict_count=dict_count, **containers)

        frequent_next_set = {
            key: val for key, val in dict_count.items() if val >= self._support
        }
        return frequent_next_set, containers

    def _loop_count_single(self, bucket, dict_count, **containers):
        """
        One loop to count singleton in one bucket.

        :param _dict: dict<item, count>. The map used to count singles.
        :param containers: other variables
        :param bucket: list<item>
        :return: _dict, params
        """
        for item in bucket:
            dict_count[item] += 1
        return dict_count, containers

    def _loop_count_pair(self, bucket, frequent_items, dict_count, **containers):
        """
        One loop to count candidate pairs in one bucket.
        Items in the candidate pair should be frequent.

        :param bucket:  list<item>
        :param frequent_items: set<item>
        :param dict_count: dict<frozenset<item,item>, count>
        :param containers: other variables (for future inherit)
        :return: dict_count, containers

        """
        filtered_bucket = [
            item for item in bucket
            if frozenset({item}) in frequent_items
        ]

        if len(filtered_bucket) < 2:
            return dict_count, containers

        for idx, itemA in enumerate(filtered_bucket[:-1]):
            for itemB in filtered_bucket[idx+1:]:
                fs = frozenset({itemA, itemB})
                if self._pass_the_constraint(fs, **containers):
                    dict_count[fs] += 1
        return dict_count, containers

    @staticmethod
    def _loop_count_triples_or_more(bucket, candidates, dict_count, **containers):
        """
        One loop to count candidate item-sets in one bucket.
        Check all candidate one by one.

        :param bucket: list<item>
        :param candidates: set<item>
        :param dict_count: dict<frozenset<items>, count>
        :param containers: other variables (for future inherit)
        :return: dict_count, containers
        """
        set_bucket = set(bucket)
        for candidate in candidates:
            if candidate.issubset(set_bucket):
                # if the candidate is a subset of bucket,
                # then the bucket contains the item-set
                dict_count[candidate] += 1
        return dict_count, containers

    @staticmethod
    def find_candidates(frequent_set, set_length):
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

    def _pass_the_constraint(self, hashable, **containers):
        """
        To check if the hashable is qualified to be a candidate.
        Not used in Apriori Algorithm, but used for other algorithms.
        Here is for future inherit.
        :param hashable: the item-set, should be hashable.
        :param containers: other containers (for future inherit)
        :return = bool: {True: can be a qualified candidate,
                         False: not a qualified candidate}
        """
        return True

    @staticmethod
    def _process_bucket(bucket):
        return list(set(bucket))


if __name__ == '__main__':
    apriori = APriori(2)
    iterator = apriori.iter_count([
        [1, 2, 3],
        [2, 3],
        [1, 4],
        [2, 4],
        [1, 2, 3, 4]
    ])
    for s in iterator:
        print(s)
