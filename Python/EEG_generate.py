#!/usr/bin/env python3

from source import *
import numpy as np


# specify dimensions
T = 25
E = 2
L = 19
I = 15
J = 12
s1 = 6
s0 = 10
gamma = 1.5

# specify kernel matrix
x = np.linspace(0, T-1, T).reshape(-1, 1)
kernel_1 = GammaExpKernel(s0=s1, gamma0=gamma)
kernel_0 = GammaExpKernel(s0=s0, gamma0=gamma)
kernel_omega = GammaExpKernel(s0=0.8, gamma0=0.6)
K_1 = kernel_1(x)
K_0 = kernel_0(x)
K_omega = kernel_omega(x)

# specify some hyper-parameters
L_1, Psi_1, V_Psi_1 = find_eigen_threshold_last(K_1)
L_0, Psi_0, V_Psi_0 = find_eigen_threshold_last(K_0)