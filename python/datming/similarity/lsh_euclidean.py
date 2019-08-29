"""
LSH for euclidean distance.

"""
from pyspark import SparkContext
from datming.similarity import euclidean_distance
import numpy as np


class LSHEuclidean(object):
    def __init__(self, block_size, maximum_distance=float("inf"),
                 n_bands=-1, num_buckets=200, random_seed=None, n_decimal=5):
        self._block_size = block_size
        self._maximum_distance = maximum_distance

        self._n_bands = int(n_bands) if 1 <= n_bands <= num_buckets else int(num_buckets)
        self._n_rows = num_buckets // self._n_bands
        self._num_buckets = self._n_rows * self._n_bands
        self._random_seed = random_seed
        self._n_decimal = n_decimal

    def run(self, rdd):
        """

        :param rdd: rdd<item_id, numpy.array>
        :return:
        """
        self._init_hyperplanes(rdd)
        return rdd.flatMap(self._compute_buckets).aggregateByKey(
            [], lambda u, v: u + [v], lambda u1, u2: u1 + u2
        ).flatMap(lambda x: self._compute_similarity(x[1])).distinct()

    def _init_hyperplanes(self, rdd):
        """
        Muller, Mervin E. "A note on a method for generating points uniformly on n-dimensional spheres."
        Communications of the ACM 2.4 (1959): 19-20.
        :return:
        """
        vec_len = len(rdd.take(1)[0][1])
        np.random.seed(self._random_seed)
        self._hyperplanes = np.random.randn(self._num_buckets, vec_len)
        self._hyperplanes = self._hyperplanes / np.linalg.norm(self._hyperplanes, axis=1).reshape(-1, 1)

    def _compute_buckets(self, key_values):
        key, values = key_values
        blocks = np.floor(np.dot(self._hyperplanes, values) / self._block_size)
        for i in range(self._n_bands):
            yield (
                (i, tuple(blocks[i*self._n_rows:(i+1)*self._n_rows])), key_values
            )

    def _compute_similarity(self, list_of_key_values):
        if len(list_of_key_values) < 2:
            return []
        list_of_key_values.sort(key=lambda x: x[0])
        for idxA, (keyA, valA) in enumerate(list_of_key_values[:-1]):
            for keyB, valB in list_of_key_values[idxA+1:]:
                distance = euclidean_distance(valA, valB)
                if distance <= self._maximum_distance:
                    yield (keyA, keyB, round(distance, self._n_decimal))


if __name__ == '__main__':
    test_data = [
        np.random.randn(5) for i in range(1000)
    ]
    sc = SparkContext.getOrCreate()
    test_rdd = sc.parallelize(
        [(i, arr) for i, arr in enumerate(test_data)]
    )

    threshold = 1

    lsh_result = LSHEuclidean(
        block_size=8, maximum_distance=threshold, n_bands=10, num_buckets=50
    ).run(rdd=test_rdd).collect()
    lsh_result = set([
        (i, j) for i, j, _ in lsh_result
    ])
    print("number of LSH-selected pairs: ", len(lsh_result))

    truth = set()
    for i, arr1 in enumerate(test_data[:-1]):
        for j, arr2 in enumerate(test_data[i+1:]):
            if euclidean_distance(arr1, arr2) <= threshold:
                truth.add((i, j+i+1))
    print("number of true pairs: ", len(truth))

    print("TP rate=", len(lsh_result & truth) / len(truth))
    print("FN rate=", len(truth - lsh_result) / len(truth))





