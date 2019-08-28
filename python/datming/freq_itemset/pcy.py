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

This algorithm is implemented on single machine, with pure python.

------------------------------

Multi-Hash Algorithm:
    Similar to PCY Algorithm, but use several independent hash functions in Pass1.

This algorithm is implemented on single machine, with pure python.

------------------------------

Multi-Stage Algorithm:
    Similar to PCY Algorithm, but rehash candidates from Pass1 and count occurrences of buckets in extra passes.

Not implemented.

------------------------------

Combination of  Multi-Hash Algorithm and Multi-Stage Algorithm:
    Each pass in Multi-Stage Algorithm can use more than one hash function.

Not implemented.
"""
from datming.freq_itemset.apriori import APriori


class PCY(APriori):
    def __init__(self, support,num_buckets, **params):
        super().__init__(support=support)
        self._num_buckets = num_buckets

    def hash(self,hashable):
        return hash(hashable) % self._num_buckets

    def iter_count(self, iterable):
        """
        :param iterable: the iterable instance.
            -> The instance with implemented __iter__(), like <list<item>>
        :return: iterator(dict<itemset, count>)
        """
        containers = {
            "count_map":[0 for i in range(self._num_buckets)]
        }
        return super().iter_count(iterable, **containers)

    def _first_pass_count_singletons(self, iterable, **containers):
        frequent_items, containers = super()._first_pass_count_singletons(iterable, **containers)
        count_map = containers.get("count_map")
        bitmap = 0
        for idx, val in enumerate(count_map):
            bitmap |= (1 << idx if val >= self._support else 0)
        containers["bitmap"] = bitmap
        del containers["count_map"]
        return frequent_items, containers

    def _loop_count_single(self, bucket, dict_count, **containers):
        dict_count, containers = super()._loop_count_single(bucket, dict_count, **containers)
        count_map = containers.get("count_map")
        for idx, itemA in enumerate(bucket[:-1]):
            for itemB in bucket[idx+1:]:
                count_map[self.hash(frozenset({itemA, itemB}))] += 1
        return dict_count, containers

    def _pass_the_constraint(self, hashable, **containers):
        return (containers["bitmap"] & (1 << self.hash(hashable))) > 0


class MultiHashPCY(APriori):
    def __init__(self, support, list_num_buckets, **params):
        super(MultiHashPCY, self).__init__(support=support)
        self.num_hash = len(list_num_buckets)
        self.list_num_buckets = list_num_buckets

    def hash(self, hashable):
        current_state = hashable
        for num_buckets in self.list_num_buckets:
            current_state = hash(current_state)
            yield current_state % num_buckets

    def iter_count(self, iterable, **containers):
        containers = {
            "count_map":[[0 for i in range(num_buckets)] for num_buckets in self.list_num_buckets]
        }
        return super(MultiHashPCY, self).iter_count(iterable, **containers)

    def _first_pass_count_singletons(self, iterable, **containers):
        frequent_items, containers = super(MultiHashPCY, self)._first_pass_count_singletons(iterable, **containers)
        count_map = containers.get("count_map")
        bitmap = [0 for i in range(self.num_hash)]
        for seq, counts in enumerate(count_map):
            for idx, val in enumerate(counts):
                bitmap[seq] |= (1 << idx if val >= self._support else 0)
        containers["bitmap"] = bitmap
        del containers["count_map"]
        return frequent_items, containers

    def _loop_count_single(self, bucket, dict_count, **containers):
        dict_count, containers = super(MultiHashPCY, self)._loop_count_single(bucket, dict_count, **containers)
        count_map = containers.get("count_map")
        for idx, itemA in enumerate(bucket[:-1]):
            for itemB in bucket[idx+1:]:
                for i, hash_code in enumerate(self.hash(frozenset({itemA,itemB}))):
                    count_map[i][hash_code] += 1
        return dict_count, containers

    def _pass_the_constraint(self, hashable, **containers):
        for i, hash_code in enumerate(self.hash(frozenset(hashable))):
            if (containers["bitmap"][i] & (1 << hash_code)) == 0:
                return False
        return True


if __name__ == '__main__':
    pcy = MultiHashPCY(1,[2,3])
    for s in pcy.iter_count([[1,2,3],[2,3],[1,4],[2,4],[1,2,3,4]]):
        print(s)