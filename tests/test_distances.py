import numpy as np

from clustering.distances import pairwise_distance_matrix


def test_distance_matrix_symmetry():
    trajs = [np.array([[0, 0], [1, 1]]), np.array([[0, 0], [2, 2]])]
    D = pairwise_distance_matrix(trajs, metric="euclidean")
    assert D.shape == (2, 2)
    assert np.allclose(D, D.T)
