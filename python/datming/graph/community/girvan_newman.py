"""
Implementation of Girvan-Newman Algorithm. (details of the algoritm: chapter 10 in mmds)

> gn = GirvanNewman(nodes=_rdd_nodes, edges=_rdd_edges)
> gn.run(print_result=True, plot_modularity_curve=True)
> best_q = gn.get_best_q()
> best_community = gn.get_best_community()
"""
from pyspark import SparkContext
from collections import namedtuple, deque
import heapq
import numpy as np
import matplotlib.pyplot as plt


class GirvanNewman(object):
    ScoreLevel = namedtuple("ScoreLevel", ("score", "level"))
    CommunityInfo = namedtuple("CommunityInfo", ("iteration", "q", "num_community"))

    def __init__(self, nodes=None, edges=None, removing_step=5, early_stopping=20):
        self._num_of_removed_edges_in_pass = removing_step
        self._early_stopping = early_stopping
        self._nodes = nodes
        self._edges = edges
        self._num_of_edges, self._num_of_nodes = 0, 0
        self._best_q, self._best_community = 0, list()
        self._modularity_curve = list()

    def add_nodes(self, nodes):
        """
        Add nodes to the community
        :param nodes: RDD<node>
        """
        if self._nodes is None:
            self._nodes = nodes
        else:
            self._nodes = self._nodes.union(nodes)

    def add_edges(self, edges):
        """
        Add edges to the community
        :param edges: RDD<(node, node)>
        """
        edges = edges.filter(lambda pair: pair[0] != pair[1])
        if self._edges is None:
            self._edges = edges
        else:
            self._edges = self._edges.union(edges)

    def graph_info(self):
        """
        Print Graph Information (nodes/edges number)
        """
        self._num_of_nodes = self._nodes.count()
        self._num_of_edges = self._edges.count()
        print(
            "Number of nodes =", self._num_of_nodes, "\n",
            "Number of edges =", self._num_of_edges
        )

    def get_best_q(self):
        return self._best_q

    def get_best_community(self):
        return self._best_community

    def get_modularity_curve(self):
        return self._modularity_curve

    def print_result(self):
        """
        Print the results after running the algorithm.
        """
        print("Best modularity(Q)=", self._best_q)
        idx = 1
        singulars = set()
        for _community in self._best_community:
            if len(_community) > 1:
                print("Community %s:" % idx, _community)
                idx += 1
            elif len(_community) == 1:
                singulars |= _community
        print("Singulars :", singulars)

    def plot_modularity_curve(self):
        """
        Plot the curve:
        ->upper: modularity vs iteration
        ->lower: number of communities vs iteration
        """
        plt.figure("212", figsize=(10, 8))
        plt.subplot("211")
        plt.plot(
            [c.iteration for c in self._modularity_curve],
            [c.q for c in self._modularity_curve]
        )
        plt.ylabel("Modularity(Q)")
        plt.subplot("212")
        plt.plot(
            [c.iteration for c in self._modularity_curve],
            [c.num_community for c in self._modularity_curve]
        )
        plt.ylabel("Number of Communities")
        plt.xlabel("Iteration")
        plt.show()

    def run(self, print_result=False, plot_modularity_curve=False):
        """
        The main function to compute communities of community.
        """
        if self._check_nodes_edges():
            self.graph_info()
        else:
            return False

        edges = self._edges.flatMap(
            lambda pair: [(pair[0], pair[1]), (pair[1], pair[0])]
        ).aggregateByKey(
            set(), lambda u, v: u | {v}, lambda u1, u2: u1 | u2
        )
        self._best_q, self._best_community = self._iterate_removing_edges(
            rdd_nodes=self._nodes,
            list_edges=edges.collect(),
            num_of_edges=self._num_of_edges
        )

        if print_result:
            self.print_result()
        if plot_modularity_curve:
            self.plot_modularity_curve()

    def _check_nodes_edges(self):
        """
        Check if the nodes and edges have been inputted.
        :return: bool(True/False)
        """
        if self._nodes is None:
            print("No node input. Please add nodes with add_nodes method")
            return False
        if self._edges is None:
            print("No edge input. Please add edges with add_edges method")
            return False
        return True

    def _iterate_removing_edges(self, rdd_nodes, list_edges, num_of_edges):
        """
        Iterated process of removing edges:
        -> Step 0: Broadcast edges to partitions
        -> Step 1: Compute the betweeness of edges
        -> Step 2: Remove k edges with highest betweenness
        -> Step 3: Compute modularity of the community to see if it reaches the maximum peak.
        -> Iterate step 1~3
        -----------------------------------------
        :param rdd_nodes: RDD<node>
        :param list_edges: List<(node, set<node>)>
        :param num_of_edges: Int
        :return: best_q, best_community
        """
        sc = SparkContext.getOrCreate()

        """
        Broadcast edges
        """
        broadcast_edges = sc.broadcast(list_edges)

        """
        edges: dict<node, list<node>>. Used for search the connected nodes.
        node_degrees: dict<node, int>. The degrees (degree = number of edges connected to the node ) of nodes.
        excluded_edges: list<(node, node)>. The edges to be removed from origin edges.
        list_nodes: list<node>. Node list.
        """
        edges = dict(list_edges)
        node_degrees = {node: len(neighbors) for node, neighbors in list_edges}
        excluded_edges = list()
        list_nodes = rdd_nodes.collect()

        best_q = -10  # a number smaller than -1 is ok
        iteration = 0
        while True:
            iteration += 1
            """
            Calculate the betweenness scores of edges.
            For each loop, the removed edges will be updated. These will be treated as changes to the origin edges /
            for the purpose of reducing communication cost.
            """
            broadcast_excluded_edges = sc.broadcast(excluded_edges)
            betweenness = self._compute_betweenness(
                rdd_nodes, broadcast_edges, broadcast_excluded_edges
            )
            """
            Find the edge with highest score, and remove that edge. 
            (Broadcast the removed edges)
            """
            top_k_edges = self._top_k(betweenness, self._num_of_removed_edges_in_pass)
            for removed, _ in top_k_edges:
                excluded_edges.append(removed)
                self._remove_edge(edges, removed)

            """
            Calculate modularity (Q), and decide next step: breaking the loop or entering the next loop.
            """
            next_q, next_community = self._modularity(list_nodes, edges, num_of_edges, node_degrees)
            self._modularity_curve.append(
                self.CommunityInfo(iteration=iteration, q=next_q, num_community=len(next_community))
            )
            if next_q > best_q:
                best_community = next_community
                best_q = next_q
                best_iteration = iteration
            elif next_q == best_q:
                best_iteration = iteration
            elif iteration >= best_iteration + self._early_stopping or len(top_k_edges) == 0:
                break

        return best_q, best_community

    @classmethod
    def _compute_betweenness(cls, nodes, broadcast_edges, broadcast_excluded_edges):
        """
        calculate credits with each root, and sum up all credits to get betweenness.
        :param nodes: RDD<node>
        :param broadcast_edges: (broadcast)list<(node, set<node>)>
        :param broadcast_excluded_edges: (broadcast)list<(node, node)>
        :return: RDD<((node, node), float)>
        """
        betweenness = nodes.mapPartitions(
            lambda roots: cls._calculate_credits(roots, broadcast_edges, broadcast_excluded_edges)
        ).reduceByKey(lambda x, y: x+y)
        return betweenness

    @classmethod
    def _calculate_credits(cls, iterator_roots, bc_edges, bc_excluded_edges):
        """
        Regard one node as root, then calculate credits of edges
        :param iterator_roots: Iterator<node>
        :param bc_edges: (broadcast)list<(node, set<node>)>
        :param bc_excluded_edges: (broadcast)list<(node, node)>
        :return:
        """
        edges = dict(bc_edges.value)
        excluded_edges = bc_excluded_edges.value

        # Remove edges.
        for excluded_edge in excluded_edges:
            cls._remove_edge(edges, excluded_edge)

        for root in iterator_roots:
            if root not in edges:
                continue

            # Label score to each node in each level
            # -> score = the number of shortest paths that reach the node from root
            scores = cls._credit_label_scores(root, edges)
            yield from cls._credit_assign_credits(root, edges, scores)

    @classmethod
    def _credit_label_scores(cls, root, edges):
        """
        Label scores to nodes: The numbers of shortest paths from the root to nodes.
        :param root: node
        :param edges: dict<node, set<node>>
        :return: dict<node, ScoreLevel(score, level)>
        """
        scores = {root: cls.ScoreLevel(score=1, level=0)}
        prev_layer = deque([root])
        while len(prev_layer) > 0:
            node = prev_layer.pop()
            parent_score, parent_level = scores[node]
            for e in edges[node]:
                if e not in scores:
                    scores[e] = cls.ScoreLevel(score=parent_score, level=parent_level+1)
                    prev_layer.appendleft(e)
                elif scores.get(e).level == parent_level+1:
                    child_score, child_level = scores.get(e)
                    scores[e] = cls.ScoreLevel(score=parent_score + child_score, level=child_level)
        return scores

    @staticmethod
    def _credit_assign_credits(root, edges, scores):
        """
        Assign credits to DAGs, from leaves to roots.
        :param root: node
        :param edges: dict<node, set<node>>
        :param scores: dict<node, ScoreLevel(score, level)>
        :return: list<(edge, credit(or betweenness))>
        """
        credit_map = {}
        dags = []

        # Travel the whole tree
        def _get_credits(node):
            if node in credit_map:
                return credit_map[node]
            score, level = scores.get(node)
            credit = 1
            for child in edges[node]:
                child_score, child_level = scores.get(child)
                if level - child_level == -1:
                    credit += _get_credits(child) * score / child_score
            for parent in edges[node]:
                parent_score, parent_level = scores.get(parent)
                if level - parent_level == 1:
                    dags.append((tuple(sorted([node, parent])), credit * parent_score / score / 2))
            credit_map[node] = credit
            return credit

        _get_credits(root)
        return dags

    @staticmethod
    def _top_k(betweenness, k):
        """
        Find edges with top-k highest betweenness
        :param betweenness: RDD<(edge, betweenness score)>
        :param k: int
        :return: list<(edge, betweenness)>
        """
        if k == 1:
            removed = betweenness.reduce(lambda x, y: x if x[1] > y[1] else y)
            return [removed]
        else:
            top_k_s = betweenness.mapPartitions(
                lambda iterator: heapq.nlargest(k, iterator, key=lambda key_value: key_value[1])
            ).collect()
            return heapq.nlargest(k, top_k_s, key=lambda key_value: key_value[1])

    @staticmethod
    def _remove_edge(edges, excluded_edge):
        """
        Remove an edge from edges.
        :param edges: dict<node, set<node>>
        :param excluded_edge: (node, node)
        :return:
        """
        start, to = excluded_edge
        if start in edges:
            try:
                edges[start].remove(to)
            except KeyError:
                # (start, to) is not in the edges.
                pass
            if len(edges[start]) == 0:
                del edges[start]
        if to in edges:
            try:
                edges[to].remove(start)
            except KeyError:
                # (start, to) is not in the edges.
                pass
            if len(edges[to]) == 0:
                del edges[to]

    @staticmethod
    def _modularity(nodes, edges, num_of_edges, node_degrees):
        """
        Calculate modularity of the community
        :param nodes: list<node>
        :param edges: dict<node, list<node>>
        :param num_of_edges: int
        :param node_degrees: dict<node, int>
        :return: modularity: int, communities: list<set<node>>
        """
        """
        Find communities from the community.
        """
        # Group nodes
        communities, grouped_nodes = list(), set()

        # Travel the entire community
        def _travel_nodes(node):
            if node in grouped_nodes:
                return []
            elif node not in edges:
                grouped_nodes.add(node)
                yield node
            else:
                grouped_nodes.add(node)
                yield node
                for _neighbor in edges[node]:
                    yield from _travel_nodes(_neighbor)

        for _node in nodes:
            if _node in grouped_nodes:
                continue
            else:
                communities.append(set(_travel_nodes(_node)))

        """
        Calculate modularity(Q) of the community.
        -> Q is between -1~1, Q>0.3~0.7 means significant community structure
        """
        # Calculate Modularity Q
        modularity = 0
        for _group in communities:
            if len(_group) == 1:
                continue
            for _node1 in _group:
                for _node2 in _group:
                    modularity -= node_degrees.get(_node1, 0) * node_degrees.get(_node2, 0) / (2 * num_of_edges)
                    if _node2 in edges[_node1]:
                        modularity += 1
        modularity = modularity / (2 * num_of_edges)
        return modularity, communities


if __name__ == '__main__':
    np.random.seed(0)
    num_nodes, num_edges, num_communities = 500, 5000, 5
    connecting_strength_among_communities = 0.01

    sc = SparkContext.getOrCreate()
    list_of_nodes = [i for i in range(num_nodes)]
    list_of_edges = list()
    for i in range(5000):
        edge = tuple(np.random.choice(list_of_nodes, 2))
        if edge[0]//(num_nodes//num_communities) != edge[1]//(num_nodes//num_communities) \
                and np.random.rand() < 1 - connecting_strength_among_communities:
            continue
        else:
            list_of_edges.append(edge)
    print(len(list_of_edges))
    _rdd_nodes = sc.parallelize(list_of_nodes)
    _rdd_edges = sc.parallelize(list_of_edges)
    GirvanNewman(nodes=_rdd_nodes, edges=_rdd_edges).run(print_result=True, plot_modularity_curve=True)




