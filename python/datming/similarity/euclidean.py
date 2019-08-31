"""
LSH for euclidean distance.

"""
from pyspark import SparkContext, RDD
from datming.utils import join_multiple_keys
import numpy as np


__all__ = [
    "EuclideanDistance"
]


class EuclideanDistanceLSH(object):
    """
    Find item pairs between which Euclidean Distance is closed enough.
    """
    def __init__(self, n_dimension: int, threshold: int,
                 block_size: int=1, n_bands: int=20, signature_length: int=200,
                 random_seed: int=None, n_partitions=5):
        """
        :param n_dimension: Dimension of vector
        :param block_size: size of block to split the dimensions.
        :param threshold: Maximum distance to consider a pair of vectors as similar vectors.
        :param n_partitions: Maximum number of partitions during the computation.
        """
        self.__block_size = block_size
        self.__n_dim = n_dimension
        self.__threshold = threshold

        self.__n_bands = n_bands
        self.__n_rows = signature_length // n_bands
        self.__signature_length = self.__n_rows * self.__n_bands
        self.__random_seed = (random_seed if isinstance(random_seed, int)
                             else np.random.randint(0, 2**32-1))
        self.__n_partitions = n_partitions

    def _lsh_predict(self, data: RDD) -> RDD:
        """
        :param data: RDD<(int, np.array)>
            = RDD<(id, vector)>
        :return: RDD<(int, int, float)>
            = RDD<(id, id, distance)>
        """
        hyperplanes = self.__init_hyperplanes(
            self.__n_dim, self.__signature_length, self.__random_seed
        )

        candidates = self.__compute_candidates(
            data, hyperplanes,
            self.__block_size, self.__n_bands, self.__n_rows, self.__n_partitions
        )
        similarity = self.__compute_similarity(
            data, candidates
        )
        threshold = self.__threshold
        similarity = similarity.filter(lambda u: u[2] <= threshold).cache()
        similarity.count()
        return similarity

    @staticmethod
    def __init_hyperplanes(n_dim: int, signature_length: int,
                           random_seed: int):
        """
        Initialize random n-D Unit vectors.
        Muller, Mervin E. "A note on a method for generating points uniformly on n-dimensional spheres."
        Communications of the ACM 2.4 (1959): 19-20.
        """
        np.random.seed(random_seed)
        hyperplanes = np.random.randn(signature_length, n_dim)
        hyperplanes = (hyperplanes / np.linalg.norm(hyperplanes, axis=1)
                       .reshape(-1, 1))
        return hyperplanes

    @staticmethod
    def __compute_candidates(data, hyperplanes,
                             block_size, n_bands, n_rows, num_partitions):
        """
        Compute signatures, group items according to signature and generate candidate pairs.
        """
        def compute(generator_of_key_values):
            for key, values in generator_of_key_values:
                blocks = np.floor(
                    np.dot(hyperplanes, values) / block_size
                )
                for i in range(n_bands):
                    yield (
                        (i, tuple(blocks[i*n_rows:(i+1)*n_rows])), key
                    )

        def generate_pairs(list_of_keys: list):
            if len(list_of_keys) < 2:
                return []
            list_of_keys.sort()
            for idxA, keyA in enumerate(list_of_keys[:-1]):
                for keyB in list_of_keys[idxA+1:]:
                    yield ((keyA, keyB), -1)

        candidates = (data
                      .mapPartitions(compute)
                      .coalesce(num_partitions)
                      .aggregateByKey(list(), lambda u, v: u + [v], lambda u1, u2: u1 + u2)
                      .map(lambda u: u[1])
                      .flatMap(generate_pairs)
                      .distinct()
                      .coalesce(num_partitions)
                      .cache()
                      )
        return candidates

    @staticmethod
    def __compute_similarity(data, candidates):
        def compute(key_values):
            (key1, key2), (_, vector1, vector2) = key_values
            return key1, key2, euclidean_distance(vector1, vector2)

        similarity = (join_multiple_keys(left=candidates, right=data, n=2)
                      .map(compute)
                      )
        return similarity


class Euclidean(EuclideanDistanceLSH):
    def __init__(self, mode: str="lsh", **kwargs):
        self.mode = mode.lower()
        if mode.lower() == "lsh":
            EuclideanDistanceLSH.__init__(self, **kwargs)
        else:
            raise NotImplementedError

    def predict(self, data: RDD) -> RDD:
        if self.mode == "lsh":
            return self._lsh_predict(data)
        else:
            raise NotImplementedError


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def test_case_with_random_data():
    test_data = [
        np.random.randn(5) for _ in range(1000)
    ]
    sc = SparkContext.getOrCreate()
    test_rdd = sc.parallelize(
        [(i, arr) for i, arr in enumerate(test_data)]
    )

    _threshold = 1

    lsh_result = Euclidean(
        block_size=8, n_dimension=5, threshold=_threshold, n_bands=10, signature_length=50
    ).predict(data=test_rdd).collect()
    lsh_result = set([
        (i, j) for i, j, _ in lsh_result
    ])
    print("number of LSH-selected pairs: ", len(lsh_result))

    truth = set()
    for i, arr1 in enumerate(test_data[:-1]):
        for j, arr2 in enumerate(test_data[i + 1:]):
            if euclidean_distance(arr1, arr2) <= _threshold:
                truth.add((i, j + i + 1))
    print("number of true pairs: ", len(truth))

    print("TP rate=", len(lsh_result & truth) / len(truth))
    print("FN rate=", len(truth - lsh_result) / len(truth))

if __name__ == '__main__':
    test_case_with_random_data()





