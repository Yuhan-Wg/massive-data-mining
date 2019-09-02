"""
Toivonen's Algorithm:
    Repeat:
        Pass1:
        -> Random Sample input data and find out frequent itemsets (lower the threshold).
        -> Construct negative border: collection of itemsets that are not frequent but all subsets are frequent in sample.
        Pass2:
        -> Count all candidate frequent itemsets in sample and itemsets in negative border.
        -> If some itemsets in negative border are frequent: repeat the loop.
        -> Or if no itemset in negative border is frequent: break the loop.
    Frequent itemsets from Pass2 is the final frequent itemsets.

This algorithm is implemented with pure python.
"""
from datming.freq_itemset.random_sampling import RandomSampling
from datming.freq_itemset.apriori import APriori
import random


class Toivonen(object):
    def __init__(self, support, sampling_rate, adjust_rate=1, random_seed=None):
        self._support = support
        self._sampling_rate = sampling_rate
        self._adjust_rate = adjust_rate
        self._random_seed = random_seed

    def count(self, iterable):
        random.seed(self._random_seed)
        idx = 1
        while True:
            dict_count_sample_sets, dict_count_negative_border = self._one_iter_find_frequent(iterable)
            if self._check_negative_border(dict_count_negative_border):
                print("Try{0}: Success!".format(idx))
                break
            print("Try{0}: Fail...".format(idx))
            idx+=1

        return {
            i: {key: val for key, val in dict_count_sample_sets[i].items() if val >= self._support}
            for i in dict_count_sample_sets
        }

    def _one_iter_find_frequent(self, iterable):
        dict_of_sample_sets, sampled_iterable = self._find_candidate_frequent_sets(iterable=iterable)
        negative_border = self._generate_negative_border(dict_of_sample_sets, sampled_iterable)
        dict_count_sample_sets, dict_count_negative_border = \
            self._scan_over_whole_file(iterable=iterable,
                                       dict_of_sample_sets=dict_of_sample_sets,
                                       negative_border=negative_border)
        return dict_count_sample_sets, dict_count_negative_border

    def _find_candidate_frequent_sets(self, iterable):
        random_sampling = RandomSampling(support=self._support * self._adjust_rate,
                                         sampling_rate=self._sampling_rate,
                                         random_seed=random.random())
        dict_of_sample_sets = random_sampling.count(iterable=iterable)
        for i in dict_of_sample_sets:
            for key in dict_of_sample_sets[i]:
                dict_of_sample_sets[i][key] = 0
        return dict_of_sample_sets, random_sampling.get_sampled_iterable()

    def _generate_negative_border(self, dict_of_sample_sets, sampled_iterable):
        negative_border = dict()
        # negative border with length 1
        for bucket in sampled_iterable:
            for item in bucket:
                if frozenset({item}) not in dict_of_sample_sets[1]:
                    negative_border[frozenset({item})] = 0

        # negative border with length 2
        for bucket in sampled_iterable:
            filtered_bucket = set(item for item in bucket if frozenset({item}) in dict_of_sample_sets[1])
            for itemA in filtered_bucket:
                for itemB in filtered_bucket:
                    if itemA == itemB: continue
                    if frozenset({itemA,itemB}) not in dict_of_sample_sets[2]:
                        negative_border[frozenset({itemA,itemB})] = 0

        # negative border with length > 2
        for key, val in dict_of_sample_sets.items():
            if key < 2:
                continue
            candidates = APriori.find_candidates(set(val.keys()), set_length=key + 1)
            if key+1 not in dict_of_sample_sets:
                for c in candidates:
                    negative_border[c] = 0
            else:
                for c in candidates:
                    if c not in dict_of_sample_sets[key+1]:
                        negative_border[c] = 0

        return negative_border

    def _scan_over_whole_file(self,iterable, dict_of_sample_sets, negative_border):
        for bucket in iterable:
            set_bucket = set(bucket)
            for item in set_bucket:
                if frozenset({item}) not in dict_of_sample_sets[1] and frozenset({item}) not in negative_border:
                    negative_border[frozenset({item})] = 0
            for i in dict_of_sample_sets:
                for key in dict_of_sample_sets[i]:
                    if key.issubset(set_bucket):
                        dict_of_sample_sets[i][key] += 1
            for key in negative_border:
                if key.issubset(set_bucket):
                    negative_border[key] += 1
        return dict_of_sample_sets, negative_border

    def _check_negative_border(self, dict_count_negative_border):
        for key,val in dict_count_negative_border.items():
            if val >= self._support:
                return False
        return True


if __name__ == '__main__':
    toivonen = Toivonen(support=2, sampling_rate=0.5, adjust_rate=0.8, random_seed=None)
    for s in toivonen.count([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]]).items():
        print(s)