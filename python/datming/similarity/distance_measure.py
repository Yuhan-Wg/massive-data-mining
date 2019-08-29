"""
Distance Measures:
    -> Jaccard Distance
    -> Euclidean Distance
    -> Consine Distance
    -> Edit Distance
    -> Hamming Distance
    -> Pearson Correlation
"""
import numpy as np





def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)


def cosine_similarity(arr1, arr2):
    return 1 - cosine_distance(arr1, arr2) / 180


def cosine_distance(arr1, arr2):
    return np.arccos((arr1 * arr2).sum() / (np.linalg.norm(arr1) * np.linalg.norm(arr2))) * 180 / np.pi

