"""
TrustRank, HITS
Authority: how important the page is about topic?
Hub: does it provide many pointers to the topic?
"""
from pyspark import SparkContext
from datming.graph.ranking.page_rank import PageRank
from datming.graph.community import generate_test_case
from collections import defaultdict


class TrustRank(object):
    class Distributed(PageRank.Distributed):
        def __init__(self, n_block=None, hash_func=hash, n_iteration=30,
                     num_partitions=None, links=None):
            super().__init__(
                n_block=n_block, hash_func=hash_func, n_iteration=n_iteration,
                teleports=None, num_partitions=num_partitions, links=links)
            self.authority = None
            self.hubbiness = None

        def run(self):
            """
            Run the algorithm.

            :return: model
            """
            """
            Initialize.
            """
            n_block = self._n_block \
                if self._n_block is not None else self._links.getNumPartitions()
            num_partitions = self.num_partitions \
                if self.num_partitions is not None else self._links.getNumPartitons()

            vector_h, _ = self._init_page_rank_vector(
                n_block=n_block, links=self._links, hash_func=self._hash_func, teleports=self._teleports
            )
            vector_a = vector_h
            transition_matrix_h2a, transition_matrix_a2h = self._init_page_rank_links(
                links=self._links, hash_func=self._hash_func, n_block=n_block
            )

            """
            Transit
            """
            for n_loop in range(self._n_iteration):
                vector_a = self._transit(self._n_block, transition_matrix_h2a, vector_h)\
                    .coalesce(num_partitions).cache()
                vector_h = self._transit(self._n_block, transition_matrix_a2h, vector_a)\
                    .coalesce(num_partitions).cache()
            self.hubbiness = vector_h.map(lambda u: u[1]).collect()
            self.authority = vector_a.map(lambda u: u[1]).collect()
            return self

        @staticmethod
        def _init_page_rank_links(n_block, links, hash_func=hash):
            """
            Initialize the graph or transition matrix from links.

            :param n_block: int
            :param links: RDD<int, int>
            :param hash_func: function: hashable->int
            :return: RDD<(int, int), (int, int, float)>
                RDD<(row_block, column_block), (row, column, value)>
            """

            def enumerate_elements(key_iterator):
                """
                :param key_iterator: (key, iterator<e>)
                    key: int, from_node
                    e: int, to_node
                :return:
                """
                from_node, _iterator = key_iterator
                to_nodes = list(_iterator)
                count = len(to_nodes)
                yield from (
                    ((hash_func(from_node) % n_block, hash_func(e) % n_block), (from_node, e, 1)) for e in to_nodes
                )
            transition_matrix_h2a = links.groupByKey().flatMap(enumerate_elements).cache()
            transition_matrix_a2h = transition_matrix_h2a.map(
                lambda u: ((u[0][1], u[0][0]), (u[1][1], u[1][0], u[1][2]))
            )
            return transition_matrix_h2a, transition_matrix_a2h

        @staticmethod
        def _transit(n_block, matrix, vector):
            """
            Matrix multiplication, transit the vector: v -> M v.
            And add teleportation or tax with damping factor.

            :param matrix: RDD<(int, int), (int, int, float)>
                RDD<(row_block, column_block), (row, column)>
            :param vector: RDD<int, (int, float)>
                RDD<row_block, (row, value)>
            :param vector_0: RDD<int, (int, float)>
                RDD<row_block, (row, value)>
            :return: RDD<int, (int, float)>
                RDD<row_block, (row, value)>
            """

            def _mat_mul(key_iterator):
                """
                :param key_iterator: ((row_block, column_block), iterator<e>)
                    e: (row, column, value)=(int, int, float) from transition matrix
                        or (row, value)=(int, float) from vector
                    column in matrix matches row in vector
                :return: iterator<e>
                    e: (row_block, (row, value))=(int, (int, float))
                """
                (row_block, _), _iterator = key_iterator
                matrix_elements = defaultdict(list)
                vector_elements = defaultdict(float)
                for e in _iterator:
                    if len(e) == 3:
                        row, column, value = e
                        matrix_elements[column].append((row, value))
                    elif len(e) == 2:
                        row, value = e
                        vector_elements[row] = value

                output_vector = defaultdict(float)
                for column in matrix_elements:
                    for row, value in matrix_elements[column]:
                        output_vector[row] += value * vector_elements[column]
                yield from (
                    (row_block, (row, output_vector[row])) for row in output_vector
                )

            def _add_up(key_iterator):
                """
                :param key_iterator: (row_block, iterator<e>)
                    row_block: int
                    e: (row, value)=(int, float)
                :return: iterator<e>
                    e: (row_block, (row, value))=(int, (int, float))
                """
                row_block, _iterator = key_iterator
                output_vector = defaultdict(float)
                for row, value in _iterator:
                    output_vector[row] += value
                yield from (
                    (row_block, (row, output_vector[row])) for row in output_vector
                )

            temp_vector = vector.flatMap(
                lambda u: [((row_block, u[0]), u[1]) for row_block in range(n_block)]
            )
            vector = matrix.union(temp_vector) \
                .groupByKey().flatMap(_mat_mul) \
                .groupByKey().flatMap(_add_up).cache()
            max_value = vector.map(lambda u: u[1][1]).max()
            vector = vector.map(lambda u: (u[0], (u[1][0], u[1][1]/max_value)))
            return vector

if __name__ == '__main__':
    list_of_links = generate_test_case(num_nodes=500, num_edges=10000,
                                       connecting_strength_among_communities=1)
    rdd_links = SparkContext.getOrCreate().parallelize(list_of_links, 40)
    tr = TrustRank.Distributed(n_block=10, n_iteration=20,
                               num_partitions=20, links=rdd_links).run()
    print("Hubbiness:", tr.hubbiness)
    print("Authority:", tr.authority)
