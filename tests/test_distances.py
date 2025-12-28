import numpy as np

from clustering.distances import pairwise_distance_matrix


def test_distance_matrix_symmetry():
    X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    D = pairwise_distance_matrix(X, metric="euclidean")
    assert D.shape == (2, 2)
    assert np.allclose(D, D.T)
