import numpy as np
# from .girvan_newman import GirvanNewman
# from .big_clam import BigCLAM
# from .spectral_clustering import SpectralClustering

__all__ = ["GirvanNewman", "BigCLAM", "SpectralClustering",
           "generate_test_case"]


def generate_test_case(num_nodes=500, num_edges=1000, num_communities=5,
                       connecting_strength_among_communities=0.01, random_state=None):
    """
    :param num_nodes: int
    :param num_edges: int
    :param num_communities: int
    :param connecting_strength_among_communities: float
    :param random_state: None or int or 1-d array_like
    :return: list<(int, int)>
    """
    np.random.seed(random_state)
    list_of_nodes = [i for i in range(num_nodes)]
    list_of_edges = list()
    count = 0
    while count < num_edges:
        edge = tuple(np.random.choice(list_of_nodes, 2))
        if edge[0] // (num_nodes // num_communities) != edge[1] // (num_nodes // num_communities) \
                and np.random.rand() > connecting_strength_among_communities:
            continue
        else:
            list_of_edges.append(edge)
            count += 1
    return list_of_edges
