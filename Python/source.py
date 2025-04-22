#!/usr/bin/env python3

import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel

class GammaExpKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, s0=1.0, gamma0=1.0):
        self.s0 = s0
        self.gamma0 = gamma0

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        distance_sq = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        K = np.exp(- (distance_sq / self.s0) ** self.gamma0)

        if eval_gradient:
            # Gradient computation is not implemented for simplicity
            raise ValueError("Gradient computation is not implemented")

        return K

    def diag(self, X):
        return np.full(X.shape[0], 1.0)

    def is_stationary(self):
        return True

def find_eigen_threshold_last(matrix, threshold=0.95):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    total_sum = sum(eigenvalues)
    cumulative_sum = np.cumsum(eigenvalues)
    number = np.argmax(cumulative_sum / total_sum >= threshold) + 1
    Psi_matrix = eigenvectors[:, :number]
    V_matrix = np.diag(eigenvalues[:number])
    return number, Psi_matrix, V_matrix

def mexican_hat(t, loc, sigma, c):
    """Mexican Hat function (Ricker Wavelet)."""
    return (c * 2 / (np.sqrt(3 * sigma) * (np.pi ** 0.25))) * (1 - ((t - loc) ** 2 / sigma ** 2)) * np.exp(
        -(t - loc) ** 2 / (2 * sigma ** 2))



