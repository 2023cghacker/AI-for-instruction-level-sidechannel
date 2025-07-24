import numpy as np


def normalize_to_01(matrix):
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_matrix


def Component_Standardlized(X, Y):
    normalized_X = normalize_to_01(X)
    return normalized_X, Y


if __name__ == '__main__':
    print("Component_Standardlized")
