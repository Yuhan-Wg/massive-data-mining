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

>> from pyspark import SparkContext
>> from datming.similarity import JaccardSimilarity
>> data = SparkContext.getOrCreate().parallelize([(1, (1,2,3)), (2, (1,2,4)), (3, (1, 3))])
>> print(JaccardSimilarity(threshold=0.5, n_bands=10, signature_length=20, seed=0).predict(data).collect())
[(1, 2, 0.55), (1, 3, 0.65)]

"""
from pyspark import SparkContext, RDD
import random
import numpy as np
from types import GeneratorType
from typing import Iterator, Hashable
from datming.utils import hash2vector, join_multiple_keys


__all__ = [
    "jaccard_similarity", "jaccard_distance", "JaccardSimilarity"
]


def jaccard_similarity(set_a: set, set_b: set) -> float:
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_distance(set_a: set, set_b: set) -> float:
    return 1 - jaccard_similarity(set_a, set_b)


class JaccardSimilarityLSH(object):
    """
    The implementation with Spark, optimized by Locality-Sensitive Hashing (LSH).
    """
    def __init__(self, threshold: float=0.,
                 n_bands: int=20, hashing_range: int=2**32-1, signature_length: int=200,
                 seed: int=None):
        """
        :param threshold: Minimum similarity value to count a pair of items as similar items.
        :param n_bands: Number of bands in LSH.
        :param hashing_range: The hashing range in LSH.
        :param signature_length: The length of signature in LSH.
        :param seed: random seed.
        """
        self.__seed = seed if isinstance(seed, int) else random.randint(0, 2**32-1)
        self.__threshold = threshold

        self.__hashing_range = hashing_range
        self.__n_bands = n_bands
        self.__n_rows = signature_length // n_bands
        self.__signature_length = self.__n_rows * self.__n_bands

    def _lsh_predict(self, data: RDD) -> RDD:
        """
        :param data: RDD<(Hashable, Iterator<Hashable>)>
            = RDD<(item, content)>
        :return: RDD<(Hashable, Hashable, float)>
            = RDD<(item A, item B, similarity)>
        """
        signature = self.__compute_signature(data).cache()
        # signature: RDD<Hashable, tuple<int>>

        pair_candidates = self.__find_candidates(signature)
        # pair_candidates: RDD<(Hashable, Hashable)>

        similar_items = self.__compute_similarity(signature, pair_candidates)
        # similarity_items: RDD<(Hashable, Hashable, float)>

        threshold = self.__threshold
        similar_items = similar_items.filter(
            lambda u: u[2] >= threshold
        )
        return similar_items

    def __compute_signature(self, data: RDD) -> RDD:
        """
        Compute signature for items.
        :param data: RDD<(Hashable, Iterator<Hashable>)>
            = RDD<(item, content)>
        :return: RDD<(Hashable, tuple<int>)>
            = RDD<(item, signature)>
        """
        hashing_range = self.__hashing_range
        signature_length = self.__signature_length
        random_seed = self.__seed
        min_hash_func = self.__min_hash

        def _signature(key_values: (Hashable, Iterator)) -> (Hashable, tuple):
            """
            Compute signature for each item
            :return (Hashable, tuple<int>)
                = (item, signature)
            """
            item, content = key_values
            signature = [hashing_range for _ in range(signature_length)]
            for element in content:
                for index_i, hashed_value in enumerate(
                        min_hash_func(element, signature_length, hashing_range, random_seed)
                ):
                    signature[index_i] = min(hashed_value, signature[index_i])
            return item, tuple(signature)

        return data.map(_signature)

    def __find_candidates(self, signature: RDD) -> RDD:
        """
        Generate candidates from signatures.
        :param signature: RDD<(Hashable, tuple<int>)>
        :return: RDD<(Hashable, Hashable)>
            Item pairs which are candidates for computing similarity later.
        """
        divide = self.__divide_signature
        generate = self.__generate_candidates
        n_bands, n_rows = self.__n_bands, self.__n_rows

        return signature.flatMap(
            lambda key_values: divide(*key_values, n_bands=n_bands, n_rows=n_rows)
        ).aggregateByKey(
            tuple(), lambda u, v: u + (v,), lambda u1, u2: u1 + u2
        ).flatMap(
            lambda key_values: generate(key_values[1])
        ).distinct()

    def __compute_similarity(self, signature: RDD, pair_candidates: RDD) -> RDD:
        """
        Compute similarity between items in pairs.
        :param signature: RDD<(Hashable, tuple<int>)>
        :param pair_candidates: RDD<(Hashable, Hashable)>
        :return: RDD<(Hashable, Hashable, float)>
            = RDD<(item, item, similarity)>
        """
        pair_candidates = pair_candidates.map(lambda u: (u, 1))
        # pair_candidates: RDD<((Hashable, Hashable), 1)>
        # signature: RDD<int, tuple<int>>

        joint = join_multiple_keys(left=pair_candidates, right=signature, n=2)
        # joint: RDD<((Hashable, Hashable), (1, tuple<int>, tuple<int>))>

        hamming = self.__hamming_similarity
        similarity = joint.map(
            lambda u: u[0] + (hamming(*u[1][1:3]),)
        )
        return similarity

    @staticmethod
    def __min_hash(hashable: int, signature_length: int,
                   hashing_range: int, random_seed: int) -> GeneratorType:
        """
        :return: Generator<int>
        """
        return hash2vector(obj=hashable, length=signature_length,
                           min_value=0, max_value=hashing_range, seed=random_seed)

    @staticmethod
    def __divide_signature(key: Hashable, values: tuple,
                           n_bands: int, n_rows: int) -> (tuple, Hashable):
        """
        Divide signatures into bands. Each band has rows.
        """
        for i in range(n_bands):
            yield ((i,) + values[i*n_rows:(i + 1) * n_rows], key)

    @staticmethod
    def __generate_candidates(iterator: Iterator) -> (Hashable, Hashable):
        """
        Generate all possible pairs from an iterator.
        """
        items = list(iterator)
        if len(items) < 2:
            return []
        items.sort()
        for i, item_a in enumerate(items[:-1]):
            for item_b in items[i+1:]:
                yield (item_a, item_b)

    @staticmethod
    def __hamming_similarity(signature_a: tuple, signature_b: tuple) -> float:
        return sum(
            int(s_a == s_b) for s_a, s_b in zip(signature_a, signature_b)
        ) / len(signature_a)


class JaccardSimilarity(JaccardSimilarityLSH):
    """
    Implementation of calculating Jaccard Similarity between items.
    """
    def __init__(self, mode="lsh", **kwargs):
        if mode == "lsh":
            JaccardSimilarityLSH.__init__(self, **kwargs)
            self.mode = mode
        else:
            raise NotImplementedError(
                "Other Implementations are not available yet."
            )

    def predict(self, data: RDD):
        if self.mode == "lsh":
            return self._lsh_predict(data)
        else:
            raise NotImplementedError


