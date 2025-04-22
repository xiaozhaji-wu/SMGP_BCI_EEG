#!/usr/bin/env python3

from source import *
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import copy
import numpy as np

# specify dimensions
T = 30
E = 2
L = 20
I = 10
J = 12
s1 = 6.5
s0 = 9.5
gamma = 1.5

# specify kernel matrix
x = np.linspace(0, T-1, T).reshape(-1, 1)
kernel_1 = GammaExpKernel(s0=s1, gamma0=gamma)
kernel_0 = GammaExpKernel(s0=s0, gamma0=gamma)
kernel_omega = GammaExpKernel(s0=0.8, gamma0=0.6)
K_1 = kernel_1(x)
K_0 = kernel_0(x)
K_omega = kernel_omega(x)
# if __name__ == "__main__":
#     x1 = np.random.multivariate_normal(np.zeros(T), K_1, 5)
#     x0 = np.random.multivariate_normal(np.zeros(T), K_0, 5)
#     time = np.arange(1, T + 1)
#     plt.plot(time, x1.T)
#     plt.show()

# np.linalg.eigvals(K_1)

# specify some hyper-parameters
L_1, Psi_1, V_Psi_1 = find_eigen_threshold_last(K_1)
L_0, Psi_0, V_Psi_0 = find_eigen_threshold_last(K_0)
rho_zeta = 0.5
Sigma_zeta_e = np.zeros((30, 30))
np.fill_diagonal(Sigma_zeta_e, 1)
Sigma_zeta_e_copy = copy.deepcopy(Sigma_zeta_e)

# generate simulation signals
t = np.linspace(0, 1000, T).reshape((T, 1))
Beta_0_true = mexican_hat(t, 300, 100, 5)
Beta_1_true = np.where((t < 200) | (t > 400), mexican_hat(t, 300, 100,5), mexican_hat(t, 300, 100,15))
Beta_0_1_true = mexican_hat(t, 300, 100, 5)
Beta_1_1_true = np.where((t < 200) | (t > 400), mexican_hat(t, 300, 100,5), mexican_hat(t, 300, 100,15))

B_0_2_true = mexican_hat(t, 700, 100, 5)
B_1_2_true = np.where((t < 600) | (t > 800), mexican_hat(t, 700, 100,5), mexican_hat(t, 700, 100,15))
Beta_0_2_true = mexican_hat(t, 700, 100, 5)
Beta_1_2_true = np.where((t < 600) | (t > 800), mexican_hat(t, 700, 100,5), mexican_hat(t, 700, 100,15))

# plt.plot(t, Beta_0_1_true)
# plt.plot(t, Beta_1_1_true)
# plt.show()
#
# plt.plot(t, Beta_0_2_true)
# plt.plot(t, Beta_1_2_true)
# plt.show()

Mu_0_true = np.concatenate((Beta_0_1_true, Beta_0_2_true), axis=1).T
Mu_1_true = np.concatenate((Beta_1_1_true, Beta_1_2_true), axis=1).T

psi_1_true = 1
psi_2_true = 1

I_unit = np.identity(T)
zeta_e_true = np.zeros((T, 1))
zeta_e_true[6: 12] = 1
zeta_e_true_diag = np.diag(zeta_e_true.flatten())
zeta_1_true = np.zeros((T, 1))
zeta_2_true = np.zeros((T, 1))
zeta_1_true[6: 12] = 1
zeta_1_true_diag = np.diag(zeta_1_true.flatten())
zeta_2_true[18: 24] = 1
zeta_2_true_diag = np.diag(zeta_2_true.flatten())

if __name__ == "__main__":
    np.savetxt('Beta_0_1_true.csv', Beta_0_1_true.flatten(), delimiter=',')
    np.savetxt('Beta_1_1_true.csv', Beta_1_1_true.flatten(), delimiter=',')
    np.savetxt('Beta_0_2_true.csv', Beta_0_2_true.flatten(), delimiter=',')
    np.savetxt('Beta_1_2_true.csv', Beta_1_2_true.flatten(), delimiter=',')
    np.savetxt('zeta_1_true.csv', zeta_1_true.flatten(), delimiter=',')
    np.savetxt('zeta_2_true.csv', zeta_2_true.flatten(), delimiter=',')

sigma_rho_sq_1_true = 4
sigma_rho_sq_2_true = 4
sigma_rho_1_true = np.sqrt(sigma_rho_sq_1_true)
sigma_rho_2_true = np.sqrt(sigma_rho_sq_2_true)
sigma_rho_true = np.diag([sigma_rho_1_true, sigma_rho_2_true])
rho_e_true = 0.6
C_e_true = np.eye(E)
C_e_true[C_e_true == 0] = rho_e_true
C_e_true = np.dot(np.dot(sigma_rho_true, C_e_true), sigma_rho_true)
rho_t_true = 0.6
rho_correlation_matrix_true = toeplitz([rho_t_true ** i for i in range(T)])
C_t_true = rho_correlation_matrix_true

Lambda_1_true = psi_1_true * (zeta_e_true_diag @ Psi_1)
Lambda_0_true = psi_2_true * (I_unit - zeta_e_true_diag) @ Psi_0
Theta_0_true, _, _, _ = np.linalg.lstsq(psi_2_true * Psi_0, Beta_0_true, rcond=None)
Theta_1_true, _, _, _ = np.linalg.lstsq(Lambda_1_true, (Beta_1_true - Lambda_0_true @ Theta_0_true), rcond=None)
Alpha_1_true = psi_1_true * Psi_1 @ Theta_1_true
Lambda_1_1_true = psi_1_true * (zeta_1_true_diag @ Psi_1)
Lambda_0_1_true = psi_2_true * (I_unit - zeta_1_true_diag) @ Psi_0
Theta_0_1_true, _, _, _ = np.linalg.lstsq(psi_2_true * Psi_0, Beta_0_1_true, rcond=None)
Theta_1_1_true, _, _, _ = np.linalg.lstsq(Lambda_1_1_true, (Beta_1_1_true - Lambda_0_1_true @ Theta_0_1_true), rcond=None)
Alpha_1_1_true = psi_1_true * Psi_1 @ Theta_1_1_true
Lambda_1_2_true = psi_1_true * (zeta_2_true_diag @ Psi_1)
Lambda_0_2_true = psi_2_true * (I_unit - zeta_2_true_diag) @ Psi_0
Theta_0_2_true, _, _, _ = np.linalg.lstsq(psi_2_true * Psi_0, Beta_0_2_true, rcond=None)
Theta_1_2_true, _, _, _ = np.linalg.lstsq(Lambda_1_2_true, (Beta_1_2_true - Lambda_0_2_true @ Theta_0_2_true), rcond=None)
Alpha_1_2_true = psi_1_true * Psi_1 @ Theta_1_2_true

