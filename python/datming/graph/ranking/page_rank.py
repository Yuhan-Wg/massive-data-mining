"""
PageRank
"""
from pyspark import SparkContext, RDD
from collections import defaultdict
from datming.graph.community import generate_test_case
import numpy as np


class PageRank(object):
    class Distributed(object):
        def __init__(self, n_block=None, hash_func=hash, n_iteration=30,
                     teleports="0", damping=0.8, num_partitions=None, links=None, topics=None):
            self._n_block = n_block
            self._n_iteration = n_iteration
            self._hash_func = hash_func
            self._teleports = "N" if teleports != "0" else "0"
            self._damping = damping
            self.num_partitions = num_partitions
            self.page_rank = None
            self._links = None
            self._topics = None
            if links is not None:
                self.add_links(links)
            if topics is not None:
                self.add_topics(topics)

        def add_links(self, links):
            """
            :param links: RDD<(int, int)>
                (int, int): (from_node, to_node)
            """
            if self._links is None:
                self._links = links.distinct()
            elif isinstance(links, RDD):
                self._links = self._links.union(links).distinct()
            else:
                raise ValueError("Input parameter links should be an RDD.")

        def add_topics(self, topics):
            """
            :param topics: RDD<int>
                int: node
            :return:
            """
            if self._topics is None:
                self._topics = topics.distinct()
            elif isinstance(topics, RDD):
                self._topics = self._topics.union(topics).distinct()
            else:
                raise ValueError("Input parameter topics should be an RDD.")

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
            num_partitions = self.num_partitions\
                if self.num_partitions is not None else self._links.getNumPartitons()

            vector, init_vector = self._init_page_rank_vector(
                n_block=n_block, links=self._links, topics=self._topics,
                hash_func=self._hash_func, teleports=self._teleports
            )
            transition_matrix = self._init_page_rank_links(
                links=self._links, hash_func=self._hash_func, n_block=n_block
            ).coalesce(num_partitions).cache()

            """
            Transit
            """
            for n_loop in range(self._n_iteration):
                vector = self._transit(self._n_block, transition_matrix, vector, init_vector, damping=self._damping)
                vector = vector.coalesce(num_partitions).cache()
            self.page_rank = vector.map(lambda u: u[1]).collect()
            return self

        @staticmethod
        def _init_page_rank_vector(n_block, nodes=None, links=None, topics=None, hash_func=hash, teleports="0"):
            """
            Initialize the page-ranks with even value (1 in this implementation).

            :param nodes: RDD<int>
            :param links: RDD<int, int>
            :param hash_func: function: hashable->int
            :param n_block: int
            :return: RDD<int, (int, float)>
            """

            if nodes is not None:
                vector = nodes.distinct().map(lambda u: (hash_func(u) % n_block, (u, 1.))).cache()
            elif links is not None:
                nodes = links.flatMap(lambda u: [u[0], u[1]]).distinct()
                vector = nodes.map(lambda u: (hash_func(u) % n_block, (u, 1.))).cache()
            else:
                raise ValueError("At least one of nodes/links should be specified.")

            if isinstance(topics, RDD):
                init_vector = topics.map(lambda u: (hash_func(u) % n_block, (u, 1.))).cache()
            else:
                init_vector = vector if teleports == "0" else None

            return vector, init_vector

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
                    ((hash_func(e) % n_block, hash_func(from_node) % n_block), (e, from_node, 1./count))
                    for e in to_nodes
                )

            return links.groupByKey().flatMap(enumerate_elements)

        @staticmethod
        def _transit(n_block, matrix, vector, vector_0=None, damping=0.8):
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

            def _add_up(key_iterator, _damping=1.):
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
                    (row_block, (row, _damping * output_vector[row])) for row in output_vector
                )

            if vector_0 is None:
                vector_0 = vector
            vector_0 = vector_0.map(lambda u: (u[0], (u[1][0], (1-damping) * u[1][1])))
            temp_vector = vector.flatMap(
                lambda u: [((row_block, u[0]), u[1]) for row_block in range(n_block)]
            )
            vector = matrix.union(temp_vector)\
                .groupByKey().flatMap(_mat_mul)\
                .groupByKey().flatMap(lambda u: _add_up(u, damping))\
                .union(vector_0)\
                .groupByKey().flatMap(_add_up)
            return vector


if __name__ == '__main__':
    list_of_links = generate_test_case(num_nodes=500, num_edges=10000,
                                       connecting_strength_among_communities=1)
    topics = SparkContext.getOrCreate().parallelize(np.random.randint(0, 500, 50))
    rdd_links = SparkContext.getOrCreate().parallelize(list_of_links).repartition(20)
    print(
        PageRank.Distributed(n_block=10, n_iteration=20, damping=0.8,
                             num_partitions=20, links=rdd_links, topics=None).run().page_rank
    )

