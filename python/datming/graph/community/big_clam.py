"""
Big-CLAM
"""
from pyspark import SparkContext
import numpy as np
import pandas as pd

from datming.graph.community import generate_test_case
__all__ = ["BigCLAM"]


class BigCLAM(object):
    class Distributed(object):
        def __init__(self, n_cluster, n_iteration=float("inf"), learning_rate=0.001, n_block=1,
                     likelihood_criteria=float("-inf"), epsilon=0.0001,
                     edges=None, hash_func=None):
            if n_iteration == float("inf") and likelihood_criteria == float("inf"):
                raise ValueError("At least one in n_iteration and likelihood_criteria need to be specified ")
            self._n_cluster = n_cluster
            self._n_iteration = n_iteration
            self._n_block = n_block
            self._learning_rate = learning_rate
            self._likelihood_criteria = likelihood_criteria
            self._epsilon = epsilon
            self._hash_func = hash_func if hash_func is not None else hash
            self._edges = None
            self.membership_matrix = None
            self._block_members = None
            self.labels = None
            if edges is not None:
                self.add_edges(edges)

        def add_edges(self, edges):
            """
            Initialize the community and vectors.
            :param edges: RDD<(int, int)> -> RDD<(int, int), (int, int)>
                The input edges (RDD<(int, int)>) contain node pairs. The nodes are presented in integers which are the
                 indexes of nodes (from 1 to N).
                The processed edges (RDD<(int, int), (int, int)>) contain block indexes and node pairs. The key is a sorted
                 pair of the block indexes of nodes and the value is a pair of the nodes.
            :return: None
            """
            """
            Initialize the community
            """
            n_block = self._n_block
            _hash = self._hash_func
            self._edges = edges.map(
                lambda u: ((_hash(u[0]) % n_block, _hash(u[1]) % n_block), (u[0], u[1]))
                if _hash(u[0]) % n_block <= _hash(u[1]) % n_block else
                ((_hash(u[1]) % n_block, _hash(u[0]) % n_block), (u[1], u[0]))
            ).groupByKey().map(lambda block_edges: (block_edges[0], list(block_edges[1]))).cache()
            """
            Initialize the membership matrix.
            :var self._block_members: dict<int, list<int>>. Cache the nodes in each block.
            :var self._membership_matrix: pandas.DataFrame. The community membership strength matrix.
            """
            n_cluster = self._n_cluster
            nodes = edges.flatMap(lambda u: [u[0], u[1]]).distinct().cache()
            self._block_members = dict(nodes.map(lambda u: (_hash(u) % n_block, u))
                                       .groupByKey()
                                       .map(lambda block_nodes: (block_nodes[0], list(block_nodes[1])))
                                       .collect())
            self.membership_matrix = pd.DataFrame(
                np.random.rand(nodes.count(), n_cluster), index=nodes.collect()
            )

        def run(self, is_print=True):
            """
            Run the Big-CLAM algorithm.
            :return: The model.

            :var self._edges: RDD<(int, int), (int, int)>
                (key, value) pair.
                key = sorted(block index 1, block index 2)
                value = (node 1, node 2)
            :var self._block_members: dict<int, list<int>>.
                Map: block index -> nodes in the block
            :var self._membership_matrix: pandas.DataFrame
                data = the community membership strength matrix (float)
                index = nodes
            """
            if self._edges is None:
                raise ValueError("Please add edges to the model and then _lsh_predict the algorithm.")
            membership_matrix = self.membership_matrix
            n_loop, likelihood_prev, improvement = 0, float("-inf"), 1
            while n_loop < self._n_iteration and improvement > self._likelihood_criteria:
                membership_matrix, likelihood = self._gradient_descent(membership_matrix)
                membership_matrix[membership_matrix < 0] = self._epsilon
                n_loop += 1
                improvement = (likelihood - likelihood_prev)/abs(likelihood)
                likelihood_prev = likelihood
                if is_print:
                    print("Loop {0}: likelihood(in log)={1}".format(n_loop, likelihood))
            self.membership_matrix = membership_matrix
            self.labels = membership_matrix.apply(np.argmax, axis=1)
            return self

        def _gradient_descent(self, membership_matrix):
            """
            :param membership_matrix: pandas.DataFrame
            :return: pandas.DataFrame, float
            """
            edges = self._edges
            members = self._block_members
            learning_rate = self._learning_rate
            """
            Cache the sum vector of matrix
            """
            over_all_strength = membership_matrix.sum(axis=0)

            """
            Update.
            df_update: SUM_v{Fv / (1-exp(-Fu Fv)))}
            over_all_strength: SUM_v{Fv}
            membership_matrix.loc[members[block_idx]]: Fu
            """
            calculate_gradient = self._calculate_gradient
            updates = edges.flatMap(
                lambda u: calculate_gradient(u, membership_matrix.loc[members[u[0][0]]]) if u[0][0] == u[0][1]
                else calculate_gradient(
                    u, membership_matrix.loc[members[u[0][0]]], membership_matrix.loc[members[u[0][1]]]
                )
            ).cache()
            likelihood = updates.map(lambda u: u[2]).sum()
            likelihood = (likelihood
                          - over_all_strength.dot(over_all_strength)
                          + (membership_matrix * membership_matrix).sum().sum())
            updates = updates.map(lambda u: (u[0], u[1])).reduceByKey(lambda x, y: x+y).collect()
            for block_idx, df_update in updates:
                membership_matrix.loc[members[block_idx]] += learning_rate * (df_update - over_all_strength +
                                                                              membership_matrix.loc[members[block_idx]])
            return membership_matrix, likelihood

        @staticmethod
        def _calculate_gradient(key_value, matrix1, matrix2=None):
            """
            :param key_value: ((int, int), list<(int, int)>)
                (int, int): sorted(block index 1, block index 2)
                list<(int, int)>: [(node 1, node 2),...]
            :param matrix1: pandas.DataFrame
                pandas.DataFrame: part of the community membership strength matrix
            :param matrix2:pandas.DataFrame
                pandas.DataFrame: part of the community membership strength matrix
            :return: RDD<int, pandas.DataFrame, float>
                int: block index
                pandas.DataFrame 1: update to membership_matrix
                float: likelihood
            """
            (block_idx1, block_idx2), nodes = key_value
            if block_idx1 == block_idx2:
                matrix2 = matrix1
            likelihood = 0

            updated_df1 = pd.DataFrame(np.zeros(matrix1.shape), index=matrix1.index)
            updated_df2 = pd.DataFrame(np.zeros(matrix2.shape), index=matrix2.index)
            for node1, node2 in nodes:
                # gradient
                dot_product = matrix1.loc[node1].dot(matrix2.loc[node2])
                coefficient = 1/(1-np.exp(-dot_product))
                updated_df1.loc[node1] += coefficient * matrix2.loc[node2]
                updated_df2.loc[node2] += coefficient * matrix1.loc[node1]
                # likelihood
                likelihood += np.log(1-np.exp(-dot_product)) + dot_product

            if block_idx1 == block_idx2:
                return [(block_idx1, updated_df1 + updated_df2, likelihood)]
            else:
                return [
                    (block_idx1, updated_df1, likelihood),
                    (block_idx2, updated_df2, likelihood)
                ]

    class Local(object):
        raise NotImplementedError


if __name__ == '__main__':
    sc = SparkContext.getOrCreate()
    list_of_edges = generate_test_case()
    _rdd_edges = sc.parallelize(list_of_edges)
    print(len(list_of_edges))
    big_clam = BigCLAM.Distributed(n_cluster=5, n_iteration=100,
                                   learning_rate=0.0001, n_block=10, likelihood_criteria=0.001,
                                   edges=_rdd_edges).run(is_print=True)

    labels = big_clam.labels
    for i in big_clam.membership_matrix.columns:
        print(labels.loc[labels == i].index)



