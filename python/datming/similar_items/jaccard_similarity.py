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
from pyspark import SparkContext, RDD
import random
import numpy as np
from datming.utils import hash2vector, join_multiple_keys


def jaccard_similarity(set_a: set, set_b: set) -> float:
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_distance(set_a: set, set_b: set) -> float:
    return 1 - jaccard_similarity(set_a, set_b)


class JaccardSimilarityLSH(object):
    """
    The implementation with Spark, optimized by Locality-Sensitive Hashing (LSH).
    """
    def __init__(self, threshold: float=0., n_bands: int=20,
                 hashing_range: int=2**32-1, signature_length: int=200,
                 seed: int=None):
        """
        :param threshold: Minimum similarity value to count a pair of items as similar items.
        :param n_bands: Number of bands in LSH.
        :param hashing_range: The hashing range in LSH.
        :param signature_length: The length of signature in LSH.
        :param seed: random seed.
        """
        self.__seed = seed if isinstance(seed) else random.randint(0, 2**32-1)
        self.__threshold = threshold

        self.__hashing_range = hashing_range
        self.__n_bands = n_bands
        self.__n_rows = signature_length // n_bands
        self.__signature_length = self.__n_rows * self.__n_bands

    def _lsh_predict(self, data: RDD[(int, list[int])]) -> RDD[(int, int, float)]:
        """
        :param data: RDD<(int, list<int>)>
            The rdd with key-value pairs containing item (int) as the key
                                                and vector (list) as the value.
        :return: RDD<(int, int, float)>
            RDD<(item A, item B, similarity)>
        """
        signature = self.__compute_signature(data).cache()
        # signature: RDD<int, tuple<int>>

        pair_candidates = self.__find_candidates(signature)
        # pair_candidates: RDD<(int, int)>

        similar_items = self.__compute_similarity(pair_candidates, signature)
        # similarity_items: RDD<(int, int, float)>

        threshold = self.__threshold
        similar_items = similar_items.filter(
            lambda u: u[2] >= threshold
        )
        return similar_items

    def __compute_signature(self, data: RDD[(int, list[int])]) -> RDD[(int, tuple[int])]:
        """
        Compute signature for each set.
        :return: RDD<(int, tuple<int>)>
        """
        hashing_range = self.__hashing_range
        signature_length = self.__signature_length
        random_seed = self.__seed
        min_hash_func = self.__min_hash

        def _signature(key_values):
            key, values = key_values
            signature = [hashing_range for _ in range(signature_length)]
            for element in values:
                for index_i, hashing_value in enumerate(
                        min_hash_func(element, signature_length, hashing_range, random_seed)
                ):
                    signature[i] = min(hashing_value, signature[index_i])
            return key, signature
        return data.map(_signature)

    @staticmethod
    def __min_hash(hashable, signature_length: int, hashing_range: int, random_seed: int) -> int:
        return hash2vector(obj=hashable, length=signature_length,
                           min_value=0, max_value=hashing_range, seed=random_seed)

    def __find_candidates(self, signature: RDD[(int, tuple[int])]) -> RDD[(int, int)]:
        """
        Generate candidates from signatures.
        :param signature: RDD<(int, tuple<int>)>
        :return:
        """
        divide = self.__divide_signature
        generate = self.__generate_candidates
        n_bands, n_rows = self.__n_bands, self.__n_rows

        return signature.flatMap(
            lambda key_values: divide(*key_values, n_bands=n_bands, n_rows=n_rows)
        ).aggregateByKey(
            tuple, lambda u, v: u + (v,), lambda u1, u2: u1 + u2
        ).flatMap(
            lambda key_values: generate(key_values[1])
        ).distinct()

    @staticmethod
    def __divide_signature(key: int, values: tuple[int], n_bands: int, n_rows: int) -> (tuple[int], int):
        """
        Divide signatures into bands. Each band has rows.
        """
        for i in range(n_bands):
            yield ((i,) + values[i*n_rows:(i + 1) * n_rows], key)

    @staticmethod
    def __generate_candidates(iterator) -> (int, int):
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

    def __compute_similarity(self, pair_candidates: RDD[(int, int)],
                             signature: RDD[(int, tuple[int])]) -> RDD[(int, int, float)]:
        """
        Compute similarity between items in pairs.
        :return: RDD<(int, int, float)>
            = RDD<(item, item, similarity)>
        """
        pair_candidates = pair_candidates.map(lambda u: (u, 1))
        # pair_candidates: RDD<((int, int), 1)>
        # signature: RDD<int, tuple<int>>

        joint = join_multiple_keys(left=pair_candidates, right=signature, n=2)
        # joint: RDD<((int, int), (1, tuple, tuple))>

        hamming = self.__hamming_similarity
        similarity = joint.map(
            lambda u: u[0] + (hamming(*u[1:3]), )
        )
        return similarity

    @staticmethod
    def __hamming_similarity(signature_a: tuple[int], signature_b: tuple[int]) -> float:
        return sum(int(s_a == s_b) for s_a, s_b in zip(signature_a, signature_b)) / len(signature_a)


class JaccardSimilarity(JaccardSimilarityLSH):
    """
    Implementation of calculating Jaccard Similarity between items.
    """
    def __init__(self, mode="lsh", **kwargs):
        if mode == "lsh":
            JaccardSimilarityLSH.__init__(self, **kwargs)
            self.predict = self._lsh_predict
        else:
            raise NotImplementedError(
                "Other Implementations are not available yet."
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

    lsh_result_temp = JaccardSimilarity(0.5, 20, signature_length=40).predict(data=test_rdd).collect()

    lsh_result = set([
        (i, j) for i, j, _ in lsh_result_temp
    ])
    print("number of LSH-selected pairs: ", len(lsh_result))

    truth = set()
    for i, arr1 in enumerate(test_buckets[:-1]):
        for j, arr2 in enumerate(test_buckets[i + 1:]):
            if jaccard_similarity(arr1, arr2) >= 0.5:
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
