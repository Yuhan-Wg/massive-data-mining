"""
Minhash Signature
    Is to construct signature of an itemset:
        -> Randomly shuffle the indexes of items with a random seed.
        -> Find the smallest shuffled item index in an itemset as an element of the signature.
        -> Repeat last two steps and construct the entire signature.
    Probability(h(itemset i) = h(itemset j)) = Jaccard(itemset i, itemset j)
    -> h(i) is the smallest shuffled item index in itemset i.
    -> Which means Signature(i,j)~Jaccard(i,j)

Locality-Sensitive Hashing(LSH)
    Is used to reduce the number of pair comparisons.
        -> If two itemsets are similar, their signatures should be similar.
        -> Two signatures are similar if and only if most of elements in the signatures have same values.
        -> If two signatures are similar, their element samples are similar too.
        -> Divide the signature into bands and hash band samples into buckets.
        -> Candidate pairs are those which are hashed into same bucket at least one band.

This algorithm is implemented with spark.
"""
from pyspark import SparkContext
import random
import numpy as np
from datming.similar_items import jaccard_similarity


class LSHJaccard(object):
    def __init__(self, threshold=0., n_bands=20, signature_range=10**4, signature_length=200, random_seed=None):
        self._range = signature_range
        self._length = signature_length
        self._random_seed = random_seed
        self._threshold = threshold
        self._n_bands = n_bands
        self._n_rows = self._length // self._n_bands
        self._length = self._n_rows * self._n_bands

    def run(self, rdd):
        """
        :param rdd: RDD<key, list<item>>The rdd of key-value pairs where key represents the item Id,
         and the value is the business Id
        :return: (item A, item B, similarity)
        """
        rdd_signatures = self._compute_signature(rdd).cache()
        signature_table = dict(rdd_signatures.collect())  # Can be optimized with HBase
        rdd_candidates = self._lsh_hash(rdd_signatures)
        similar_items = rdd_candidates.map(
            lambda items: self._lsh_compute_similarity(items[0], items[1], signature_table)
        ).filter(lambda x: x[2] >= self._threshold)
        return similar_items

    def _compute_signature(self, rdd_file):
        _range = self._range
        _length = self._length
        _random_seed = self._random_seed
        min_hash = self._min_hash

        def _signature(key_values):
            key, values = key_values
            signature = [_range for _ in range(_length)]
            for item in values:
                for i, s in enumerate(min_hash(item, _random_seed)):
                    signature[i] = min(s, signature[i])
            return key, signature

        return rdd_file.map(_signature)

    def _min_hash(self, hashable, _random_seed):
        if _random_seed is not None:
            random.seed(_random_seed + hash(hashable))
        else:
            random.seed(hash(hashable))
        yield from (
            random.randint(1, self._range) for _ in range(self._length)
        )

    def _lsh_hash(self, rdd_file):
        return rdd_file.flatMap(
            lambda key_values: self._lsh_divide(*key_values)
        ).aggregateByKey(
            [], lambda u, v: u + [v], lambda u1, u2: u1 + u2
        ).flatMap(
            lambda key_values: self._lsh_candidates(key_values[1])
        ).distinct()

    def _lsh_divide(self, key, values):
        for i in range(self._n_bands):
            yield (tuple(values[i*self._n_rows:(i+1)*self._n_rows] + [i]), key)

    @staticmethod
    def _lsh_candidates(iterator):
        items = list(iterator)
        if len(items) < 2:
            return []

        items.sort()
        for i, itemA in enumerate(items[:-1]):
            for itemB in items[i+1:]:
                yield (itemA, itemB)

    @staticmethod
    def _lsh_compute_similarity(item_a, item_b, signature_table):
        if item_a not in signature_table or item_b not in signature_table:
            return item_a, item_b, 0
        signatureA = signature_table[item_a]
        signatureB = signature_table[item_b]
        return (
            item_a, item_b,
            sum(int(sA == sB) for sA, sB in zip(signatureA, signatureB)) / len(signatureA)
        )


if __name__ == '__main__':
    test_buckets = [
            set(np.random.randint(1, 20, i))
            for i in np.random.randint(1, 10, 1000)
        ]
    sc = SparkContext.getOrCreate()
    test_rdd = sc.parallelize([
        (item, business)
        for item, buckets in enumerate(test_buckets)
        for business in buckets
    ]).groupByKey().map(lambda u: (u[0], list(u[1]))).cache()

    threshold = 0.5
    lsh_result_temp = LSHJaccard(threshold, 20, signature_length=40).run(rdd=test_rdd).collect()

    lsh_result = set([
        (i, j) for i, j, _ in lsh_result_temp
    ])
    print("number of LSH-selected pairs: ", len(lsh_result))

    truth = set()
    for i, arr1 in enumerate(test_buckets[:-1]):
        for j, arr2 in enumerate(test_buckets[i + 1:]):
            if jaccard_similarity(arr1, arr2) >= threshold:
                truth.add((i, j + i + 1))
    print("number of true pairs: ", len(truth))

    print("TP rate=", len(lsh_result & truth) / len(truth))
    print("FN rate=", len(truth - lsh_result) / len(truth))

    sum_square_error = 0
    count = 0
    for itemA, itemB, lsh_similarity in lsh_result_temp:
        truth = len(set(test_buckets[itemA]) & set(test_buckets[itemB])) /\
                len(set(test_buckets[itemA]) | set(test_buckets[itemB]))
        sum_square_error += (lsh_similarity - truth) ** 2
        count += 1
    print("RMSE=",np.sqrt(sum_square_error/count))
