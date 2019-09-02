"""
Spectral Clustering
"""
from pyspark import SparkContext
from collections import defaultdict
import heapq
import numpy as np
from numpy.random import RandomState, rand
import matplotlib.pyplot as plt
from scipy.linalg import eigh, norm
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datming.clustering import KMeans


class SpectralClustering(object):
    class Distributed(object):
        def __init__(self, weight=False, normalize="none", scale=True,
                     n_clusters=6, n_iteration=None, n_clustering_iteration=15, n_block=10,
                     random_state=None, edges=None, epsilon=10**-2, clustering_epsilon=10**-2):
            if normalize not in ["none", "symmetric", "random_walk"]:
                raise ValueError("Please specify normalize type with "
                                 "argument normalize = (none, symmetric, random_walk)")
            else:
                self._normalize = normalize

            self._n_clusters = n_clusters
            self._n_iteration = n_iteration if n_iteration else n_clusters
            self._n_clustering_iteration = n_clustering_iteration
            self._n_block = n_block
            self._weight = weight
            self._random_state = random_state
            self._is_scale = scale
            self._D, self._A, self._L = None, None, None
            self._n = None
            self._clusters = None
            self._epsilon = epsilon
            self._clustering_epsilon = clustering_epsilon
            if edges is not None:
                self.add_edges(edges)

        def add_edges(self, edges):
            """
            Convert edges into matrix, and compute Graph Laplacian.
            The matrix in this implementation is represented like:
                (key, value)
                key = (index_of_row_block, index_of_column_block)
                value = (index_of_row, index_of_column, weight or matrix value)
            :param edges: RDD[(nodeA, nodeB)] or RDD[(nodeA, nodeB, weight)]
            :return:
            """
            _hash_func = self._hash
            _n_block = self._n_block
            if not self._weight:
                edges = edges.map(lambda u: (
                    (_hash_func(u[0]) % _n_block, _hash_func(u[1]) % _n_block), (u[0], u[1], -1)
                ))  # 1 means constant weight
            else:
                edges = edges.map(lambda u: (
                    (_hash_func(u[0]) % _n_block, _hash_func(u[1]) % _n_block), (u[0], u[1], -u[2])
                ))
            self._A = edges

            def _accumulate_nodes(key_iterator_of_node_weight):
                key, iterator_of_node_weight = key_iterator_of_node_weight
                weight_map = dict()
                for node, _, weight in iterator_of_node_weight:
                    weight_map[node] = weight_map.get(node, 0) + weight

                for node in weight_map:
                    yield (key, (node, node, weight_map[node]))

            self._D = edges.flatMap(lambda u: [
                ((u[0][0], u[0][0]), (u[1][0], u[1][0], -u[1][2])),
                ((u[0][1], u[0][1]), (u[1][1], u[1][1], -u[1][2]))
            ]).groupByKey().flatMap(_accumulate_nodes)
            self._L = self._A.union(self._D).cache()

        def run(self):
            """
            :return:
            """
            """
            vectors: RDD<(block_idx, (row, col, val))> -> RDD<row, array>
            """
            vectors = self.heigen().cache()
            _dim = vectors.map(lambda u: u[1][1]).max() + 1

            def _combine(key_iterator_of_elements):
                """
                :param key_iterator_of_elements: (key, iter<element>)
                    element = (row, column, value)
                :return: iter<row, array>
                """
                key, iterator_of_elements = key_iterator_of_elements
                buffer = dict()
                for row, column, value in iterator_of_elements:
                    if row not in buffer:
                        buffer[row] = np.zeros(_dim)
                    buffer[row][column] += value
                for row in buffer:
                    yield (row, buffer[row])

            vectors = vectors.groupByKey().flatMap(_combine).cache()
            model = KMeans(n_clusters=self._n_clusters,
                           n_iterations=self._n_clustering_iteration,
                           epsilon=self._clustering_epsilon,
                           scale=self._is_scale)
            model.fit_transform(vectors)
            return model.labels.groupByKey().map(lambda u: list(u[1])).collect()

        def heigen(self):
            """
            :return: RDD<(block_idx, (row, col, value))>
            """
            """
            Initialization.
            """
            #  Initialize v as a normalized random n-vector.
            _hash_func = self._hash
            _random_state = self._random_state
            init_random_vector = self._D.map(lambda u: (
                u[0][0], (u[1][0], RandomState(
                    seed=(_hash_func(u[1][0]) + _hash_func(_random_state)) if _random_state is not None else None
                ).rand())))
            l2_norm_of_random_vector = self._rdd_l2_norm_of_vector(init_random_vector)

            # v_prev = v_i-1, v_now = v_i
            input_matrix = self._L
            basis_vector_prev = None
            basis_vector_now = init_random_vector.map(
                lambda u: (u[0], (u[1][0], u[1][1]/l2_norm_of_random_vector))
            ).cache()
            basis_vector_matrix = basis_vector_now.map(
                lambda u: (u[0], (u[1][0], 0, u[1][1]))
            ).cache()
            beta_prev = 0.
            reduced_tri_diagonal_matrix = np.zeros((self._n_iteration, self._n_iteration))
            """
            Loop: i = 1 -> n_iteration-1
            In the i-th loop:
                reduced_tri_diagonal_matrix_rank_i.shape == (i, i)
            """
            for i in range(1, self._n_iteration):
                _temp_vector = self._rdd_matrix_dot_vector(matrix=input_matrix, vector=basis_vector_now).cache()
                alpha_now = self._rdd_vector_dot_vector(basis_vector_now, _temp_vector)
                _temp_vector = self._rdd_vector_plus_vector(
                    vector_a=self._rdd_vector_plus_vector(
                        vector_a=_temp_vector, vector_b=basis_vector_prev, coefficient_b=-beta_prev
                    ),
                    vector_b=basis_vector_now,
                    coefficient_b=-alpha_now
                ).cache()  # _temp_vector = _temp_vector - beta_prev * basis_vector_prev - alpha_i * basis_vector_now
                beta_now = self._rdd_l2_norm_of_vector(_temp_vector)
                reduced_tri_diagonal_matrix[i-1, i-1] = alpha_now
                reduced_tri_diagonal_matrix[i-1, i] = reduced_tri_diagonal_matrix[i, i-1] = beta_now
                reduced_tri_diagonal_matrix_rank_i = reduced_tri_diagonal_matrix[0:i, 0:i]
                _, _eigen_vectors = eigh(reduced_tri_diagonal_matrix_rank_i)

                is_selectively_orthogonalized = False
                for j in range(0, i):
                    if beta_now * abs(_eigen_vectors[i-1, j]) <= \
                                    (self._epsilon ** 0.5) * norm(reduced_tri_diagonal_matrix_rank_i, ord=2):
                        _r = self._rdd_matrix_dot_numpy_vector(
                            rdd_matrix=basis_vector_matrix, numpy_vector=np.asarray(_eigen_vectors[:, j]).reshape(-1)
                        )
                        _temp_vector = self._rdd_vector_plus_vector(
                            vector_a=_temp_vector,
                            vector_b=_r,
                            coefficient_b=-self._rdd_vector_dot_vector(_r, _temp_vector)
                        )
                        is_selectively_orthogonalized = True

                if is_selectively_orthogonalized:
                    beta_now = self._rdd_l2_norm_of_vector(_temp_vector)
                if beta_now == 0:
                    reduced_tri_diagonal_matrix = reduced_tri_diagonal_matrix[0:i, 0:i]
                    break

                basis_vector_prev = basis_vector_now
                basis_vector_now = _temp_vector.map(lambda u: (u[0], (u[1][0], u[1][1]/beta_now))).cache()
                basis_vector_matrix = basis_vector_matrix.union(
                    basis_vector_now.map(lambda u: (u[0], (u[1][0], i, u[1][1])))
                ).cache()
                beta_prev = beta_now

            _, _eigen_vectors = eigh(reduced_tri_diagonal_matrix)
            return self._rdd_matrix_dot_numpy_matrix(
                rdd_matrix=basis_vector_matrix, numpy_matrix=_eigen_vectors)

        @staticmethod
        def _hash(hashable):
            """
            Use a __hash function to divide nodes into blocks.
            """
            return hash(hashable)

        @staticmethod
        def _rdd_matrix_dot_vector(matrix, vector):
            """
            block_idx_2/column in matrix matches block_idx/row
            :param matrix: RDD<(block_idx_1, block_idx_2), (row, column, value)>
            :param vector: RDD<block_idx, (row, value)>
            :return: RDD<block_idx, (row, value)>
            """
            def _mat_mul(key_iterator_of_elements):
                """
                :param key_iterator_of_elements: (key, iter<element>)
                    element = (block_idx_1, (row, column, value)) or (row, value)
                :return: iter<(block_idx, (row, value))>
                """
                matrix_elements = defaultdict(list)
                vector_elements = defaultdict(int)
                for element in key_iterator_of_elements[1]:
                    if isinstance(element[1], tuple):
                        # matrix element
                        block_idx_1, (row, column, value) = element
                        matrix_elements[column].append(
                            (block_idx_1, row, value)
                        )
                    elif isinstance(element[1], (float, int)):
                        # vector element
                        row, value = element
                        vector_elements[row] += value
                    else:
                        # wrong type
                        continue

                for idx in matrix_elements:
                    if idx not in vector_elements:
                        continue
                    for block_idx_1, row, value in matrix_elements[idx]:
                        yield (block_idx_1, (row, value * vector_elements[idx]))

            def _add_up(key_iterator_of_elements):
                """
                :param key_iterator_of_elements: (key, iter(elements))
                    element = (row, value)
                :return: iter<(block_idx, (row, value))>
                """
                block_idx, iterator_of_elements = key_iterator_of_elements
                elements = defaultdict(int)
                for row, value in iterator_of_elements:
                    elements[row] += value

                for row in elements:
                    yield (block_idx, (row, elements[row]))

            _temp_matrix = matrix.map(lambda u: (u[0][1], (u[0][0], u[1])))\
                .union(vector)\
                .groupByKey().flatMap(_mat_mul)\
                .groupByKey().flatMap(_add_up)
            return _temp_matrix

        @staticmethod
        def _rdd_vector_dot_vector(vector_a, vector_b):
            """
            :param vector_a: RDD<block_idx, (row, value)>
            :param vector_b: RDD<block_idx, (row, value)>
            :return: float/int
            """
            def _vec_mul(key_iterator_of_elements):
                """
                Note: For each row, there is at almost two values (from vector_a and vector_b, respectively),
                    so there is no need to distinguish them.
                :param key_iterator_of_elements: (key, iter(elements))
                    element = (row, value)
                :return: float
                """
                _, iterator_of_elements = key_iterator_of_elements
                buffer = dict()
                for row, value in iterator_of_elements:
                    if row not in buffer:
                        buffer[row] = value
                    else:
                        yield buffer[row] * value
                        del buffer[row]

            dot_result = vector_a.union(vector_b).groupByKey().flatMap(_vec_mul).sum()
            return dot_result

        @staticmethod
        def _rdd_vector_plus_vector(vector_a, vector_b, coefficient_a=1., coefficient_b=1.):
            """
            :param vector_a: RDD<block_idx, (row, value)>
            :param vector_b: RDD<block_idx, (row, value)>
            :param coefficient_a: number
            :param coefficient_b: number
            :return: RDD<block_idx, (row, value)>
            """
            if coefficient_b == 0 and coefficient_a == 0:
                return vector_a.map(lambda u: (u[0], (u[1][0], 0)))
            elif coefficient_b == 0 and coefficient_a == 1:
                return vector_a
            elif coefficient_a == 0 and coefficient_b == 1:
                return vector_b
            elif coefficient_b == 0:
                return vector_a.map(lambda u: (u[0], (u[1][0], u[1][1] * coefficient_a)))
            elif coefficient_a == 0:
                return vector_b.map(lambda u: (u[0], (u[1][0], u[1][1] * coefficient_b)))
            else:
                coefficients = [coefficient_a, coefficient_b]

                def _add_up(key_iterator_of_elements):
                    """
                    :param key_iterator_of_elements: (key, iter<element>)
                        element = (flag, (row, value)), flat = 0 or 1
                    :return: iter<(block_idx, (row, value))>
                    """
                    block_idx, iterator_of_elements = key_iterator_of_elements
                    elements = defaultdict(int)
                    for flag, (row, value) in iterator_of_elements:
                        elements[row] += coefficients[flag] * value

                    for row in elements:
                        yield (block_idx, (row, elements[row]))

                _temp_vector_a = vector_a.map(lambda u: (u[0], (0, u[1])))
                _temp_vector_b = vector_b.map(lambda u: (u[0], (1, u[1])))
                return _temp_vector_a.union(_temp_vector_b).groupByKey().flatMap(_add_up)

        @staticmethod
        def _rdd_l2_norm_of_vector(vector):
            return vector.map(lambda u: u[1][1] ** 2).sum() ** 0.5

        @staticmethod
        def _rdd_matrix_dot_numpy_vector(rdd_matrix, numpy_vector):
            """
            :param rdd_matrix: RDD<block_idx_of_row, (row, column, value)>
            :param numpy_vector: np.array
            :return: RDD<block_idx, (row, value)>
            """
            def _mat_mul_vec(key_iterator_of_elements):
                """

                :param key_iterator_of_elements: (key, iter<element>)
                        element = (row, column, value)
                :return: iter<(block_idx, (row, value))>
                """
                block_idx_of_row, iterator_of_elements = key_iterator_of_elements
                buffer = defaultdict(int)
                for row, column, value in iterator_of_elements:
                    buffer[row] += value * numpy_vector[column]
                for row in buffer:
                    yield (block_idx_of_row, (row, buffer[row]))

            return rdd_matrix.groupByKey().flatMap(_mat_mul_vec)

        @staticmethod
        def _rdd_matrix_dot_numpy_matrix(rdd_matrix, numpy_matrix):
            """
            :param rdd_matrix: RDD<block_idx_of_row, (row, column, value)>
            :param numpy_matrix: np.matrix
            :return: RDD<block_idx_of_row, (row, column, value)>
            """
            def _mat_mul_mat(key_iterator_of_elements):
                """

                :param key_iterator_of_elements: (key, iter<element>)
                        element = (row, column, value)
                :return: iter<(block_idx, (row, col, value))>
                """
                block_idx_of_row, iterator_of_elements = key_iterator_of_elements
                buffer = dict()
                for row, column, value in iterator_of_elements:
                    if row not in buffer:
                        buffer[row] = np.zeros(numpy_matrix.shape[1])
                    buffer[row] += value * numpy_matrix[column, :]
                for row in buffer:
                    for col, val in enumerate(buffer[row]):
                        yield (block_idx_of_row, (row, col, val))
            return rdd_matrix.groupByKey().flatMap(_mat_mul_mat)

    class Local(object):
        def __init__(self, weight=False, normalize="none", scaled=True,
                     n_clusters=6, n_dimension=None, random_state=None, edges=None):
            if normalize not in ["none", "symmetric", "random_walk"]:
                raise ValueError("Please specify normalize type with argument "
                                 "normalize = (none, symmetric, random_walk)")
            else:
                self._normalize = normalize

            self._n_clusters = n_clusters
            self._n_dimension = n_dimension if n_dimension else n_clusters
            self._weight = weight
            self._random_state = random_state
            self._scaled = scaled
            self._D = None
            self._A = None
            self._clusters = None
            self._vectors = None

            if edges is not None:
                self.add_edges(edges)

        def add_edges(self, edges):
            """
            :param edges: Iterable[(nodeA, nodeB)] or Iterable[(nodeA, nodeB, weight)]
            :return:
            """
            array_edges = np.array(edges)
            if self._weight is False:
                array_edges = np.concatenate((array_edges[:, 0:2], np.ones((array_edges.shape[0], 1))), axis=1)
            else:
                array_edges = array_edges[:, 0:3]

            self._A = coo_matrix(
                (array_edges[:, 2], (array_edges[:, 0].astype(int), array_edges[:, 1].astype(int)))
            ).toarray()
            self._A = self._A + self._A.transpose()
            self._D = np.asarray(np.sum(self._A, axis=0)).reshape(-1)
            self._D[self._D < 1] = 1

        def run(self):
            if self._normalize == "none":
                """
                Un-normalized Graph Laplacian.
                    L[:,:] = D - A
                It's to solve:
                    L[:,:] v[:,i] = w[i] * v[:,i]
                w is eigenvalue and v is eigenvector.
                """
                _, vectors = eigh(np.diag(self._D) - self._A, eigvals=(1, self._n_dimension))
            elif self._normalize == "symmetric":
                """
                Normalized Graph Laplacian.
                    L_sym = (D^-0.5) L (D^-0.5) = I - (D^-0.5) A (D^-0.5)
                It's to solve:
                    L_sym[:,:] v[:,i] = w[i] v[:,i]
                """
                inverse_d = 1/self._D ** 0.5
                _, vectors = eigh(
                    np.identity(self._D.shape[0]) - inverse_d.reshape((-1, 1)) * self._A * inverse_d,
                    eigvals=(1, self._n_dimension))
            elif self._normalize == "random_walk":
                """
                Normalized Graph Laplacian according to Shi and Malik (2000).
                    L_rw = (D^-1) L = I - (D^-1) A
                It's to solve:
                    L[:,:] v[:,i] = w[i] D[:,:] v[:,i]
                """
                _, vectors = eigh(np.diag(self._D) - self._A,
                                  b=np.diag(self._D),
                                  eigvals=(1, self._n_dimension))

            self._vectors = StandardScaler().fit_transform(vectors) if self._scaled else vectors
            self._clusters = KMeans(n_clusters=self._n_clusters, random_state=self._random_state).fit(self._vectors)
            return self._clusters.labels_

        def plot(self, i, figsize=(10, 6)):
            plt.figure(figsize=figsize)
            plt.scatter([i for i in range(1, self._vectors.shape[0]+1)],
                        sorted(self._vectors[:, i]))
            plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    num_nodes, num_edges, num_communities = 500, 1000, 5
    connecting_strength_among_communities = 0.01

    list_of_nodes = [i for i in range(num_nodes)]
    list_of_edges = list()
    count = 0
    while count < num_edges:
        edge = tuple(np.random.choice(list_of_nodes, 2))
        if edge[0]//(num_nodes//num_communities) != edge[1]//(num_nodes//num_communities) \
                and np.random.rand() < 1 - connecting_strength_among_communities:
            continue
        else:
            list_of_edges.append(edge)
            count += 1
    sc = SparkContext.getOrCreate()
    _rdd_edges = sc.parallelize(list_of_edges)
    print(len(list_of_edges))
    scl = SpectralClustering.Distributed(n_clusters=5,
                                         normalize="none",
                                         n_iteration=5,
                                         n_clustering_iteration=20,
                                         block_size=20,
                                         edges=_rdd_edges,
                                         epsilon=10 ** -2, clustering_epsilon=10 ** -4)
    labels = scl.run()
    for v in labels:
        print(sorted(v))
    # scl.plot(1)

    #for i in range(5):
    #    print([idx for idx, e in enumerate(labels) if e == i])

