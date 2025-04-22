#!/usr/bin/env python3

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import pickle
from jax import random
from self_Func import TruncatedNormal, GammaExponentialKernel

def model(X1, X0, T, E, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0):
    # sampling kernel variance
    psi = {}
    for s in range(2):
        for e in range(1, E + 1):
            psi_e = f"psi_{s}_{e}"
            psi[psi_e] = numpyro.sample(psi_e, dist.LogNormal(0.0, 1.0))
            # psi[psi_e] = 1.0

    L = {}
    Psi = {}
    V_Psi = {}
    for s in range(2):
        if s == 0:
            L[f"L_{s}"] = L_0
            Psi[f"Psi_{s}"] = jnp.array(Psi_0)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_0)
        elif s == 1:
            L[f"L_{s}"] = L_1
            Psi[f"Psi_{s}"] = jnp.array(Psi_1)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_1)

    theta_mean = {}
    theta_cov = {}
    theta = {}
    alpha = {}
    for s in range(2):
        for e in range(1, E + 1):
            theta_mean_s_e = f"theta_mean_{s}_{e}"
            theta_cov_s_e = f"theta_cov_{s}_{e}"
            theta_s_e = f"theta_{s}_{e}"
            alpha_s_e = f"alpha_{s}_{e}"
            theta_mean[theta_mean_s_e] = jnp.zeros(L[f"L_{s}"])
            theta_cov[theta_cov_s_e] = V_Psi[f"V_Psi_{s}"]
            theta[theta_s_e] = numpyro.sample(theta_s_e,
                                              dist.MultivariateNormal(theta_mean[theta_mean_s_e],
                                                                      theta_cov[theta_cov_s_e]))
            # theta[f"theta_{0}_{1}"] = jnp.array(Theta_0_1_true.flatten())
            # theta[f"theta_{1}_{1}"] = jnp.array(Theta_1_1_true.flatten())
            # theta[f"theta_{0}_{2}"] = jnp.array(Theta_0_2_true.flatten())
            # theta[f"theta_{1}_{2}"] = jnp.array(Theta_1_2_true.flatten())
            alpha[alpha_s_e] = psi[f"psi_{s}_{e}"] * jnp.dot(Psi[f"Psi_{s}"], theta[f"theta_{s}_{e}"])

    # sampling psi_omega and rho_omega
    psi_omega = {}
    rho_omega = {}
    # psi_omega_same = numpyro.sample("psi_omega", dist.InverseGamma(3, 30))
    psi_omega_same = 40
    rho_omega_same = 0.5
    for e in range(1, E + 1):
        # psi_omega_e = f"psi_omega_{e}"
        # psi_omega[psi_omega_e] = numpyro.sample(psi_omega_e, dist.InverseGamma(3, 20))
        psi_omega_e = f"psi_omega_{e}"
        psi_omega[psi_omega_e] = psi_omega_same
        rho_omega_e = f"rho_omega_{e}"
        # rho_omega[rho_omega_e] = numpyro.sample(rho_omega_e, dist.Uniform(0.0, 1.0))
        rho_omega[rho_omega_e] = rho_omega_same

    # define omega prior
    omega_mean = {}
    omega_cov = {}
    for e in range(1, E + 1):
        omega_mean_e = f"omega_mean_{e}"
        omega_mean[omega_mean_e] = jnp.zeros(T)

        omega_cov_e = f"omega_cov_{e}"
        cor_matrix = jnp.zeros((T, T))
        cor_matrix = cor_matrix.at[jnp.diag_indices(T)].set(1)
        cor_matrix = cor_matrix.at[jnp.arange(T - 1), jnp.arange(1, T)].set(rho_omega[f"rho_omega_{e}"])
        cor_matrix = cor_matrix.at[jnp.arange(1, T), jnp.arange(T - 1)].set(rho_omega[f"rho_omega_{e}"])
        omega_cov[omega_cov_e] = psi_omega[f"psi_omega_{e}"] * cor_matrix

    # sampling omega
    omega = {}
    for e in range(1, E + 1):
        omega_e = f"omega_{e}"
        omega[omega_e] = numpyro.sample(omega_e, dist.MultivariateNormal(omega_mean[f"omega_mean_{e}"],
                                                                         omega_cov[f"omega_cov_{e}"]))

    # calculate zeta
    zeta = {}
    for e in range(1, E + 1):
        zeta_e = f"zeta_{e}"
        zeta[zeta_e] = dist.Normal(0, 1).cdf(omega[f"omega_{e}"])

    # calculate beta
    beta = {}
    for s in range(2):
        for e in range(1, E + 1):
            beta_s_e = f"beta_{s}_{e}"
            if s == 0:
                beta[beta_s_e] = alpha[f"alpha_{s}_{e}"]
            elif s == 1:
                beta[beta_s_e] = (
                        jnp.diag(zeta[f"zeta_{e}"]) @ alpha[f"alpha_{s}_{e}"] + jnp.diag(1 - zeta[f"zeta_{e}"]) @
                        alpha[f"alpha_{0}_{e}"])

    # calculate Mu
    Mu = {}
    for s in range(2):
        Mu_s = f"Mu_{s}"
        Mu[Mu_s] = jnp.stack([beta[f"beta_{s}_{e}"] for e in range(1, E + 1)])

    # sampling sigma_rho
    sigma_rho = {}
    for e in range(1, E + 1):
        sigma_rho_e = f"sigma_rho_{e}"
        sigma_rho[sigma_rho_e] = numpyro.sample(sigma_rho_e, dist.HalfCauchy(5.0))
        # sigma_rho[f"sigma_rho_{1}"] = 3.0
        # sigma_rho[f"sigma_rho_{2}"] = 4.0

    # calculate sigma_rho_sq
    sigma_rho_sq = {}
    for e in range(1, E + 1):
        sigma_rho_sq_e = f"sigma_rho_sq_{e}"
        sigma_rho_sq[sigma_rho_sq_e] = sigma_rho[f"sigma_rho_{e}"] ** 2

    # sampling rho_t and rho_e
    rho_t = numpyro.sample("rho_t", TruncatedNormal(0.5, 1.0, 0.0, 1.0))
    rho_e = numpyro.sample("rho_e", dist.Uniform(0.0, 1.0))
    # rho_t = 0.6
    # rho_e = 0.6

    # calculate C_e and Sigma_e
    rho_e_valid = jnp.clip(rho_e, 0, 0.99)
    C_e = rho_e_valid * jnp.ones((E, E)) + (1 - rho_e_valid) * jnp.eye(E)
    sigma_rho_vec = jnp.array([sigma_rho[f"sigma_rho_{e}"] for e in range(1, E + 1)])
    sigma_rho_valid = jnp.diag(sigma_rho_vec)
    Sigma_e = sigma_rho_valid @ C_e @ sigma_rho_valid

    with numpyro.plate("data1", N1):
        temp_diff = X1 - Mu["Mu_1"]
        numpyro.sample("obs1_1", dist.MultivariateNormal(jnp.zeros(E), Sigma_e), obs=temp_diff[:, :, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, :, t - 1]
            curr_col = temp_diff[:, :, t]
            numpyro.sample(f"obs1_{t + 1}", dist.MultivariateNormal(rho_t * prev_col, Sigma_e), obs=curr_col)

    with numpyro.plate("data0", N0):
        temp_diff = X0 - Mu["Mu_0"]
        numpyro.sample("obs0_1", dist.MultivariateNormal(jnp.zeros(E), Sigma_e), obs=temp_diff[:, :, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, :, t - 1]
            curr_col = temp_diff[:, :, t]
            numpyro.sample(f"obs0_{t + 1}", dist.MultivariateNormal(rho_t * prev_col, Sigma_e), obs=curr_col)

def model_ref(X1, X0, T, E, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0):
    # sampling kernel variance
    psi = {}
    for s in range(2):
        for e in range(1, E + 1):
            psi_e = f"psi_{s}_{e}"
            psi[psi_e] = numpyro.sample(psi_e, dist.LogNormal(0.0, 1.0))

    # # define kernel with sampled variance
    # kernel = {}
    # for s in range(2):
    #     for e in range(1, E + 1):
    #         kernel_s_e = f"kernel_{s}_{e}"
    #         lengthscale = s1 if s == 1 else s0
    #         kernel[kernel_s_e] = GammaExponentialKernel(lengthscale=lengthscale, gamma=gamma)
    #         kernel[kernel_s_e].variance = 1

    L = {}
    Psi = {}
    V_Psi = {}
    for s in range(2):
        if s == 0:
            L[f"L_{s}"] = L_0
            Psi[f"Psi_{s}"] = jnp.array(Psi_0)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_0)
        elif s == 1:
            L[f"L_{s}"] = L_1
            Psi[f"Psi_{s}"] = jnp.array(Psi_1)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_1)

    theta_mean = {}
    theta_cov = {}
    theta = {}
    alpha = {}
    for s in range(2):
        for e in range(1, E + 1):
            theta_mean_s_e = f"theta_mean_{s}_{e}"
            theta_cov_s_e = f"theta_cov_{s}_{e}"
            theta_s_e = f"theta_{s}_{e}"
            alpha_s_e = f"alpha_{s}_{e}"
            theta_mean[theta_mean_s_e] = jnp.zeros(L[f"L_{s}"])
            theta_cov[theta_cov_s_e] = V_Psi[f"V_Psi_{s}"]
            theta[theta_s_e] = numpyro.sample(theta_s_e,
                                              dist.MultivariateNormal(theta_mean[theta_mean_s_e],
                                                                      theta_cov[theta_cov_s_e]))
            alpha[alpha_s_e] = psi[f"psi_{s}_{e}"] * jnp.dot(Psi[f"Psi_{s}"], theta[theta_s_e])
    #
    # # define GP prior
    # alpha_gp_mean = {}
    # alpha_gp_cov = {}
    # for s in range(2):
    #     for e in range(1, E + 1):
    #         alpha_gp_mean_s_e = f"alpha_gp_mean_{s}_{e}"
    #         alpha_gp_cov_s_e = f"alpha_gp_cov_{s}_{e}"
    #         alpha_gp_mean[alpha_gp_mean_s_e] = jnp.zeros(T)
    #         alpha_gp_cov[alpha_gp_cov_s_e] = kernel[f"kernel_{s}_{e}"](jnp.arange(T).reshape(-1, 1))
    #
    # # sampling GP alpha
    # alpha = {}
    # for s in range(2):
    #     for e in range(1, E + 1):
    #         alpha_s_e = f"alpha_{s}_{e}"
    #         alpha[alpha_s_e] = numpyro.sample(alpha_s_e,
    #                                           dist.MultivariateNormal(alpha_gp_mean[f"alpha_gp_mean_{s}_{e}"],
    #                                                                   alpha_gp_cov[f"alpha_gp_cov_{s}_{e}"]))

    # calculate beta
    beta = {}
    for s in range(2):
        for e in range(1, E + 1):
            beta_s_e = f"beta_{s}_{e}"
            beta[beta_s_e] = alpha[f"alpha_{s}_{e}"]

    # calculate Mu
    Mu = {}
    for s in range(2):
        Mu_s = f"Mu_{s}"
        Mu[Mu_s] = jnp.stack([beta[f"beta_{s}_{e}"] for e in range(1, E + 1)])

    # sampling sigma_rho
    sigma_rho = {}
    for e in range(1, E + 1):
        sigma_rho_e = f"sigma_rho_{e}"
        sigma_rho[sigma_rho_e] = numpyro.sample(sigma_rho_e, dist.HalfCauchy(5.0))

    # calculate sigma_rho_sq
    sigma_rho_sq = {}
    for e in range(1, E + 1):
        sigma_rho_sq_e = f"sigma_rho_sq_{e}"
        sigma_rho_sq[sigma_rho_sq_e] = sigma_rho[f"sigma_rho_{e}"] ** 2

    # sampling rho_t and rho_e
    rho_t = numpyro.sample("rho_t", TruncatedNormal(0.5, 1.0, 0.0, 1.0))
    rho_e = numpyro.sample("rho_e", dist.Uniform(0.0, 1.0))

    # calculate C_e and Sigma_e
    rho_e_valid = jnp.clip(rho_e, 0, 0.99)
    C_e = rho_e_valid * jnp.ones((E, E)) + (1 - rho_e_valid) * jnp.eye(E)
    sigma_rho_vec = jnp.array([sigma_rho[f"sigma_rho_{e}"] for e in range(1, E + 1)])
    sigma_rho_valid = jnp.diag(sigma_rho_vec)
    Sigma_e = sigma_rho_valid @ C_e @ sigma_rho_valid

    with numpyro.plate("data1", N1):
        temp_diff = X1 - Mu["Mu_1"]
        numpyro.sample("obs1_1", dist.MultivariateNormal(jnp.zeros(E), Sigma_e), obs=temp_diff[:, :, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, :, t - 1]
            curr_col = temp_diff[:, :, t]
            numpyro.sample(f"obs1_{t + 1}", dist.MultivariateNormal(rho_t * prev_col, Sigma_e), obs=curr_col)

    with numpyro.plate("data0", N0):
        temp_diff = X0 - Mu["Mu_0"]
        numpyro.sample("obs0_1", dist.MultivariateNormal(jnp.zeros(E), Sigma_e), obs=temp_diff[:, :, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, :, t - 1]
            curr_col = temp_diff[:, :, t]
            numpyro.sample(f"obs0_{t + 1}", dist.MultivariateNormal(rho_t * prev_col, Sigma_e), obs=curr_col)

def model_single(X1, X0, T, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0):
    # sampling kernel variance
    psi = {}
    for s in range(2):
        psi_s = f"psi_{s}"
        psi[psi_s] = numpyro.sample(psi_s, dist.LogNormal(0.0, 1.0))

    # # define kernel with sampled variance
    # kernel = {}
    # for s in range(2):
    #     kernel_s = f"kernel_{s}"
    #     lengthscale = s1 if s == 1 else s0
    #     kernel[kernel_s] = GammaExponentialKernel(lengthscale=lengthscale, gamma=gamma)
    #     kernel[kernel_s].variance = psi[f"psi_{s}"]

    # # define GP prior
    # alpha_gp_mean = {}
    # alpha_gp_cov = {}
    # for s in range(2):
    #     alpha_gp_mean_s = f"alpha_gp_mean_{s}"
    #     alpha_gp_cov_s = f"alpha_gp_cov_{s}"
    #     alpha_gp_mean[alpha_gp_mean_s] = jnp.zeros(T)
    #     alpha_gp_cov[alpha_gp_cov_s] = kernel[f"kernel_{s}"](jnp.arange(T).reshape(-1, 1))
    #
    # # sampling GP alpha
    # alpha = {}
    # for s in range(2):
    #     alpha_s = f"alpha_{s}"
    #     alpha[alpha_s] = numpyro.sample(alpha_s, dist.MultivariateNormal(alpha_gp_mean[f"alpha_gp_mean_{s}"],
    #                                                                      alpha_gp_cov[f"alpha_gp_cov_{s}"]))

    L = {}
    Psi = {}
    V_Psi = {}
    for s in range(2):
        if s == 0:
            L[f"L_{s}"] = L_0
            Psi[f"Psi_{s}"] = jnp.array(Psi_0)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_0)
        elif s == 1:
            L[f"L_{s}"] = L_1
            Psi[f"Psi_{s}"] = jnp.array(Psi_1)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_1)

    theta_mean = {}
    theta_cov = {}
    theta = {}
    alpha = {}
    for s in range(2):
        theta_mean_s = f"theta_mean_{s}"
        theta_cov_s = f"theta_cov_{s}"
        theta_s = f"theta_{s}"
        alpha_s = f"alpha_{s}"
        theta_mean[theta_mean_s] = jnp.zeros(L[f"L_{s}"])
        theta_cov[theta_cov_s] = V_Psi[f"V_Psi_{s}"]
        theta[theta_s] = numpyro.sample(theta_s,
                                        dist.MultivariateNormal(theta_mean[theta_mean_s],
                                                                theta_cov[theta_cov_s]))
        alpha[alpha_s] = psi[f"psi_{s}"] * jnp.dot(Psi[f"Psi_{s}"], theta[theta_s])

    # sampling psi_omega and rho_omega
    psi_omega = {}
    rho_omega = {}
    psi_omega_same = 40
    rho_omega_same = 0.5

    psi_omega_e = f"psi_omega"
    # psi_omega[psi_omega_e] = numpyro.sample(psi_omega_e, dist.LogNormal(0.0, 1.0))
    psi_omega[psi_omega_e] = psi_omega_same
    rho_omega_e = f"rho_omega"
    # rho_omega[rho_omega_e] = numpyro.sample(rho_omega_e, dist.Uniform(0.0, 1.0))
    rho_omega[rho_omega_e] = rho_omega_same

    # define omega prior
    omega_mean = {}
    omega_cov = {}

    omega_mean_e = f"omega_mean"
    omega_mean[omega_mean_e] = jnp.zeros(T)

    omega_cov_e = f"omega_cov"
    cor_matrix = jnp.zeros((T, T))
    cor_matrix = cor_matrix.at[jnp.diag_indices(T)].set(1)
    cor_matrix = cor_matrix.at[jnp.arange(T - 1), jnp.arange(1, T)].set(rho_omega[f"rho_omega"])
    cor_matrix = cor_matrix.at[jnp.arange(1, T), jnp.arange(T - 1)].set(rho_omega[f"rho_omega"])
    omega_cov[omega_cov_e] = psi_omega[f"psi_omega"] * cor_matrix

    # sampling omega
    omega = {}

    omega_e = f"omega"
    omega[omega_e] = numpyro.sample(omega_e, dist.MultivariateNormal(omega_mean[f"omega_mean"],
                                                                     omega_cov[f"omega_cov"]))

    # calculate zeta
    zeta = {}

    zeta_e = f"zeta"
    zeta[zeta_e] = dist.Normal(0, 1).cdf(omega[f"omega"])

    # calculate beta
    beta = {}
    for s in range(2):

        beta_s = f"beta_{s}"
        if s == 0:
            beta[beta_s] = alpha[f"alpha_{s}"]
        elif s == 1:
            beta[beta_s] = (jnp.diag(zeta[f"zeta"]) @ alpha[f"alpha_{s}"] + jnp.diag(1 - zeta[f"zeta"]) @
                            alpha[f"alpha_{0}"])

    # calculate Mu
    Mu = {}
    for s in range(2):
        Mu_s = f"Mu_{s}"
        Mu[Mu_s] = beta[f"beta_{s}"]

    # sampling sigma_rho
    sigma_rho = {}

    sigma_rho_e = f"sigma_rho"
    sigma_rho[sigma_rho_e] = numpyro.sample(sigma_rho_e, dist.HalfCauchy(5.0))

    # calculate sigma_rho_sq
    sigma_rho_sq = {}

    sigma_rho_sq_e = f"sigma_rho_sq"
    sigma_rho_sq[sigma_rho_sq_e] = sigma_rho[f"sigma_rho"] ** 2

    # sampling rho_t
    rho_t = numpyro.sample("rho_t", TruncatedNormal(0.5, 1.0, 0.0, 1.0))

    with numpyro.plate("data1", N1):
        temp_diff = X1 - Mu["Mu_1"]
        numpyro.sample("obs1_1", dist.Normal(0, sigma_rho[f"sigma_rho"]), obs=temp_diff[:, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, t - 1]
            curr_col = temp_diff[:, t]
            numpyro.sample(f"obs1_{t + 1}", dist.Normal(rho_t * prev_col, sigma_rho[f"sigma_rho"]), obs=curr_col)

    with numpyro.plate("data0", N0):
        temp_diff = X0 - Mu["Mu_0"]
        numpyro.sample("obs0_1", dist.Normal(0, sigma_rho[f"sigma_rho"]), obs=temp_diff[:, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, t - 1]
            curr_col = temp_diff[:, t]
            numpyro.sample(f"obs0_{t + 1}", dist.Normal(rho_t * prev_col, sigma_rho[f"sigma_rho"]), obs=curr_col)

def model_single_ref(X1, X0, T, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0):
    # sampling kernel variance
    psi = {}
    for s in range(2):
        psi_s = f"psi_{s}"
        psi[psi_s] = numpyro.sample(psi_s, dist.LogNormal(0.0, 1.0))

    L = {}
    Psi = {}
    V_Psi = {}
    for s in range(2):
        if s == 0:
            L[f"L_{s}"] = L_0
            Psi[f"Psi_{s}"] = jnp.array(Psi_0)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_0)
        elif s == 1:
            L[f"L_{s}"] = L_1
            Psi[f"Psi_{s}"] = jnp.array(Psi_1)
            V_Psi[f"V_Psi_{s}"] = jnp.array(V_Psi_1)

    theta_mean = {}
    theta_cov = {}
    theta = {}
    alpha = {}
    for s in range(2):
        theta_mean_s = f"theta_mean_{s}"
        theta_cov_s = f"theta_cov_{s}"
        theta_s = f"theta_{s}"
        alpha_s = f"alpha_{s}"
        theta_mean[theta_mean_s] = jnp.zeros(L[f"L_{s}"])
        theta_cov[theta_cov_s] = V_Psi[f"V_Psi_{s}"]
        theta[theta_s] = numpyro.sample(theta_s,
                                        dist.MultivariateNormal(theta_mean[theta_mean_s],
                                                                theta_cov[theta_cov_s]))
        alpha[alpha_s] = psi[f"psi_{s}"] * jnp.dot(Psi[f"Psi_{s}"], theta[theta_s])

    # calculate beta
    beta = {}
    for s in range(2):
        beta_s = f"beta_{s}"
        beta[beta_s] = alpha[f"alpha_{s}"]

    # calculate Mu
    Mu = {}
    for s in range(2):
        Mu_s = f"Mu_{s}"
        Mu[Mu_s] = beta[f"beta_{s}"]

    # sampling sigma_rho
    sigma_rho = {}

    sigma_rho_e = f"sigma_rho"
    sigma_rho[sigma_rho_e] = numpyro.sample(sigma_rho_e, dist.HalfCauchy(5.0))

    # calculate sigma_rho_sq
    sigma_rho_sq = {}

    sigma_rho_sq_e = f"sigma_rho_sq"
    sigma_rho_sq[sigma_rho_sq_e] = sigma_rho[f"sigma_rho"] ** 2

    # sampling rho_t
    rho_t = numpyro.sample("rho_t", TruncatedNormal(0.5, 1.0, 0.0, 1.0))

    with numpyro.plate("data1", N1):
        temp_diff = X1 - Mu["Mu_1"]
        numpyro.sample("obs1_1", dist.Normal(0, sigma_rho[f"sigma_rho"]), obs=temp_diff[:, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, t - 1]
            curr_col = temp_diff[:, t]
            numpyro.sample(f"obs1_{t + 1}", dist.Normal(rho_t * prev_col, sigma_rho[f"sigma_rho"]), obs=curr_col)

    with numpyro.plate("data0", N0):
        temp_diff = X0 - Mu["Mu_0"]
        numpyro.sample("obs0_1", dist.Normal(0, sigma_rho[f"sigma_rho"]), obs=temp_diff[:, 0])
        for t in range(1, T):
            prev_col = temp_diff[:, t - 1]
            curr_col = temp_diff[:, t]
            numpyro.sample(f"obs0_{t + 1}", dist.Normal(rho_t * prev_col, sigma_rho[f"sigma_rho"]), obs=curr_col)

def run_mcmc(model, X1, X0, T, E, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0, seed):
    rng_key = random.PRNGKey(seed)
    N1 = N1
    N0 = N0
    T = T
    E = E
    X1 = jnp.array(X1)
    X0 = jnp.array(X0)
    s1 = s1
    s0 = s0
    gamma = gamma

    # define NUTS sampling kernel
    kernel = NUTS(model, step_size=0.1)

    # MCMC sampling settings
    mcmc = MCMC(kernel, num_samples=1000, num_warmup=2000, num_chains=2)

    # MCMC sampling
    mcmc.run(rng_key, X1, X0, T, E, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0)

    # get samples
    samples = mcmc.get_samples()

    # print samples
    return samples

def run_mcmc_single(model, X1, X0, T, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0, seed):
    rng_key = random.PRNGKey(seed)
    N1 = N1
    N0 = N0
    T = T
    X1 = jnp.array(X1)
    X0 = jnp.array(X0)
    s1 = s1
    s0 = s0
    gamma = gamma

    # define NUTS sampling kernel
    kernel = NUTS(model, step_size=0.1)

    # MCMC sampling settings
    mcmc = MCMC(kernel, num_samples=1000, num_warmup=3000, num_chains=2)

    # MCMC sampling
    mcmc.run(rng_key, X1, X0, T, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0)

    # get samples
    samples = mcmc.get_samples()

    # print samples
    return samples
