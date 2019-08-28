"""
K-Means
"""
from pyspark import SparkContext
import numpy as np


class KMeans(object):
    def __init__(self, n_clusters, n_iterations, epsilon=10**-2, scale=True):
        self._n_clusters = n_clusters
        self._n_iterations = n_iterations
        self._epsilon = epsilon
        self.centers = self.labels = None
        self._is_scale = scale

    def fit_transform(self, vectors):
        """
        :param vectors: RDD<row, np.array>
        :return:
        """
        centers = self._sampling(vectors, self._n_clusters)
        if self._is_scale:
            vectors = self._scale(vectors)
        vectors = vectors.map(lambda u: (-1, u)).cache()
        vectors, error = self._assign(vectors, centers)
        n_loop = 0
        while error > self._epsilon and n_loop < self._n_iterations:
            centers = self._center(vectors)
            vectors, error = self._assign(vectors, centers)
            vectors = vectors.cache()
            n_loop += 1

        self.centers = centers
        self.labels = vectors.map(lambda u: (u[0], u[1][0]))
        return self

    @staticmethod
    def _scale(vectors):
        _norm = vectors.map(lambda u: u[1] ** 2).reduce(lambda u1, u2: u1+u2) ** 0.5
        return vectors.map(lambda u: (u[0], u[1]/_norm))

    @staticmethod
    def _sampling(vectors, _n_clusters):
        def _random_choose(iterator):
            buffer = []
            for _, arr in iterator:
                buffer.append(arr)
            if len(buffer) < 1:
                return []
            buffer = np.asarray(buffer)
            return buffer[np.random.choice(buffer.shape[0], min(_n_clusters, buffer.shape[0]), replace=False)]

        samples = np.asarray(vectors.mapPartitions(_random_choose).collect())
        return samples[np.random.choice(samples.shape[0], _n_clusters, replace=False)]

    @staticmethod
    def _assign(vectors, centers):
        def _assign_single_row(row):
            _, (row_idx, arr) = row
            error = np.sum((centers - arr) ** 2, axis=1)
            return error.argmin(), error.min(), (row_idx, arr)
        vectors = vectors.map(_assign_single_row).cache()
        return vectors.map(lambda u: (u[0], u[2])), vectors.map(lambda u: u[1]).mean()

    @staticmethod
    def _center(vectors):
        def _average(key_iterator_of_elements):
            key, iterator_of_elements = key_iterator_of_elements
            count, sum_of_arr = 0, 0
            for _, arr in iterator_of_elements:
                count += 1
                sum_of_arr += arr
            return sum_of_arr/count
        return np.asarray(vectors.groupByKey().map(_average).collect())


if __name__ == '__main__':
    sc = SparkContext.getOrCreate()
    rdd = sc.parallelize([(i, np.random.rand(5)) for i in range(100)])
    print(KMeans(n_clusters=5, n_iterations=30).fit_transform(rdd).labels.collect())
