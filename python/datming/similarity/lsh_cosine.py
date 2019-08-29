"""
LSH for cosine distance.

"""
from pyspark import SparkContext
import numpy as np
from datming.similarity import cosine_similarity


class LSHCosine(object):
    def __init__(self, is_similarity=True, minimum_similarity=0, maximum_distance=-1,
                 n_bands=-1, sketch_length=200, random_seed=None, n_decimal=5):
        self._is_similarity = is_similarity

        self._threshold = minimum_similarity
        if not self._is_similarity and maximum_distance >= 0:
            self._threshold = 1 - maximum_distance / 180

        self._n_bands = int(n_bands) if 1 <= n_bands <= sketch_length else int(sketch_length)
        self._n_rows = sketch_length // self._n_bands
        self._sketch_len = self._n_rows * self._n_bands
        self._random_seed = random_seed
        self._n_decimal = n_decimal

    def run(self, rdd):
        """
        :param rdd: rdd< item_id, numpy.array >
            the lengths of all numpy.arrays should be same or the program will throw out error.
        :return:
        """
        self._init_hyperplanes(rdd)
        return rdd.flatMap(self._compute_sketch).\
            aggregateByKey([], lambda u, v: u + [v], lambda u1, u2: u1 + u2).\
            flatMap(lambda x: self._compute_similarity(x[1])).\
            distinct()

    def _init_hyperplanes(self, rdd):
        """
        Muller, Mervin E. "A note on a method for generating points uniformly on n-dimensional spheres."
        Communications of the ACM 2.4 (1959): 19-20.
        :return:
        """
        vec_len = len(rdd.take(1)[0][1])
        np.random.seed(self._random_seed)
        self._hyperplanes = np.random.randn(self._sketch_len, vec_len)
        self._hyperplanes = self._hyperplanes / np.linalg.norm(self._hyperplanes, axis=1).reshape(-1, 1)

    def _compute_sketch(self, key_values):
        _, values = key_values
        sketch = ((self._hyperplanes * values.reshape(1, -1)).sum(axis=1) > 0).reshape(self._n_bands, self._n_rows)
        for i in range(self._n_bands):
            code = 0
            for j in range(self._n_rows):
                if sketch[i, j]:
                    code |= 1 << j
            yield ((i, code), key_values)

    def _compute_similarity(self, list_of_key_values):
        if len(list_of_key_values) < 2:
            return []
        list_of_key_values.sort(key=lambda x: x[0])
        for idxA, (keyA, valA) in enumerate(list_of_key_values[:-1]):
            for keyB, valB in list_of_key_values[idxA+1:]:
                similarity = cosine_similarity(valA, valB)
                if similarity < self._threshold:
                    continue
                if self._is_similarity:
                    similarity = round(similarity, self._n_decimal)
                    yield (keyA, keyB, similarity)
                else:
                    distance = round((1 - similarity) * 180, self._n_decimal)
                    yield (keyA, keyB, distance)


if __name__ == '__main__':
    test_data = [
        np.random.randn(5) for i in range(1000)
    ]
    sc = SparkContext.getOrCreate()
    test_rdd = sc.parallelize(
        [(i, arr) for i, arr in enumerate(test_data)]
    )

    threshold = 0.9

    lsh_result = LSHCosine(
        minimum_similarity=threshold, n_bands=10, sketch_length=50
    ).run(rdd=test_rdd).collect()
    lsh_result = set([
        (i, j) for i, j, _ in lsh_result
    ])
    print("number of LSH-selected pairs: ", len(lsh_result))

    truth = set()
    for i, arr1 in enumerate(test_data[:-1]):
        for j, arr2 in enumerate(test_data[i+1:]):
            if cosine_similarity(arr1, arr2) >= threshold:
                truth.add((i, j+i+1))
    print("number of true pairs: ", len(truth))

    print("TP rate=", len(lsh_result & truth) / len(truth))
    print("FN rate=", len(truth - lsh_result) / len(truth))
