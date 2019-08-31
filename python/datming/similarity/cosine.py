"""
LSH for cosine distance.

"""
from pyspark import SparkContext, RDD
import numpy as np
from types import GeneratorType
from datming.utils import join_multiple_keys


__all__ = [
    "CosineSimilarity"
]


class CosineSimilarityLSH(object):
    def __init__(self, n_dimension: int, threshold: float=0.,
                 n_bands: int=20, signature_length: int=200,
                 random_seed: int=None, n_partitions: int=None):
        self.__threshold = threshold
        self.__n_dim = n_dimension

        self.__n_bands = n_bands
        self.__n_rows = signature_length // self.__n_bands
        self.__signature_length = self.__n_rows * self.__n_bands
        self.__random_seed = (random_seed if isinstance(random_seed, int)
                              else np.random.randint(0, 2**32-1))
        self.__n_partitions = n_partitions

    def _lsh_predict(self, data):
        """
        :param data: RDD<(Hashable, numpy.array)>
             = RDD<(item, vector)>
            the lengths of all numpy.arrays should be same or the program will throw out error.
        :return:
        """
        hyperplanes = self.__init_hyperplanes(
            self.__n_dim, self.__signature_length, self.__random_seed
        )

        if self.__n_partitions is None:
            self.__n_partitions = data.getNumPartitions()

        candidates = self.__compute_candidates(
            data, hyperplanes, self.__n_bands, self.__n_rows, self.__n_partitions
        )
        similarity = self.__compute_similarity(
            data, candidates, self.__n_partitions
        )

        threshold = self.__threshold
        similarity = similarity.filter(lambda u: u[2] >= threshold).cache()
        similarity.count()
        return similarity

    @staticmethod
    def __init_hyperplanes(vector_length, signature_length, random_seed):
        """
        Initialize unit vectors which are uniformly distributed on the n-D sphere.
        Reference: Muller, Mervin E. "A note on a method for generating points
        uniformly on n-dimensional spheres." Communications of the ACM 2.4 (1959): 19-20.
        :return:
        """
        np.random.seed(random_seed)
        hyperplanes = np.random.randn(signature_length, vector_length)
        hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1).reshape(-1, 1)
        return hyperplanes

    @staticmethod
    def __compute_candidates(data: RDD, hyperplanes: np.array,
                             n_bands: int, n_rows: int, num_partitions: int)-> RDD:
        """
        Generate signatures, hash keys into buckets according to signature
         and group keys in the same bucket.
        :return: RDD<Tuple[Hashable]>
            = RDD<Tuple[key]>
        """
        def compute(generator_of_key_values: GeneratorType):
            for key_values in generator_of_key_values:
                key, values = key_values
                signature = (
                    ((hyperplanes * values.reshape(1, -1)).sum(axis=1) > 0).reshape(n_bands, n_rows)
                )
                for i in range(n_bands):
                    code = sum([1 << j for j in range(n_rows) if signature[i, j]])
                    yield ((i, code), key)

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
    def __compute_similarity(data: RDD, candidates: RDD, num_partitions: int) -> RDD:
        """
        Compute similarity.
        """
        def compute(key_values):
            (key1, key2), (_, vector1, vector2) = key_values
            return key1, key2, cosine_similarity(vector1, vector2)

        similarity = (join_multiple_keys(left=candidates, right=data, n=2)
                      .map(compute)
                      )
        return similarity


class CosineSimilarity(CosineSimilarityLSH):
    def __init__(self, mode="lsh", **kwargs):
        self.mode = mode
        if mode == "lsh":
            CosineSimilarityLSH.__init__(self, **kwargs)
        else:
            raise NotImplementedError

    def predict(self, data: RDD) -> RDD:
        if self.mode == "lsh":
            return self._lsh_predict(data)
        else:
            raise NotImplementedError


def cosine_similarity(arr1, arr2):
    return 1 - cosine_distance(arr1, arr2) / 180


def cosine_distance(arr1, arr2):
    return np.arccos((arr1 * arr2).sum() / (np.linalg.norm(arr1) * np.linalg.norm(arr2))) * 180 / np.pi


def test_cosine_similarity_with_random_data():
    test_data = [
        np.random.randn(5) for _ in range(1000)
    ]
    sc = SparkContext.getOrCreate()
    test_rdd = sc.parallelize(
        [(i, arr) for i, arr in enumerate(test_data)]
    )

    threshold = 0.9

    lsh_result = CosineSimilarity(
        threshold=threshold, vector_length=5,
        n_bands=10, signature_length=50
    ).predict(data=test_rdd).collect()
    lsh_result = set([
        (i, j) for i, j, _ in lsh_result
    ])
    print("number of LSH-selected pairs: ", len(lsh_result))

    truth = set()
    for i, arr1 in enumerate(test_data[:-1]):
        for j, arr2 in enumerate(test_data[i + 1:]):
            if cosine_similarity(arr1, arr2) >= threshold:
                truth.add((i, j + i + 1))
    print("number of true pairs: ", len(truth))
    print("TP rate=", len(lsh_result & truth) / len(truth))
    print("FN rate=", len(truth - lsh_result) / len(truth))


if __name__ == '__main__':
    test_cosine_similarity_with_random_data()
