#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import matrix_normal

def generate_multi_data(Mu_1_true, Mu_0_true, C_t_true, C_e_true, L, I, J, T, E, seed=0):
    np.random.seed(seed)

    # Generate matrix normal data for X_Y_1 and X_Y_0
    X_Y_1 = matrix_normal.rvs(mean=Mu_1_true, rowcov=C_e_true, colcov=C_t_true, size=(L * I * int(J / 6)))
    X_Y_0 = matrix_normal.rvs(mean=Mu_0_true, rowcov=C_e_true, colcov=C_t_true, size=(L * I * int(5 * J / 6)))

    # Reshape to match the expected dimensions
    X_Y_1 = X_Y_1.reshape((L * I * int(J / 6), E, T))
    X_Y_0 = X_Y_0.reshape((L * I * int(5 * J / 6), E, T))

    # Create Y column
    Y_1 = np.ones(L * I * int(J / 6))
    Y_0 = np.zeros(L * I * int(5 * J / 6))

    # Create Letter and Sequence columns
    Letter_1 = np.repeat(np.arange(1, L + 1), I * int(J / 6))
    Letter_0 = np.repeat(np.arange(1, L + 1), I * int(5 * J / 6))

    Sequence_1 = np.tile(np.repeat(np.arange(1, I + 1), int(J / 6)), L)
    Sequence_0 = np.tile(np.repeat(np.arange(1, I + 1), int(5 * J / 6)), L)

    # Create Flash column
    selected_target_flash_pairs = set()

    target_flashes = np.zeros((L, 2), dtype=int)
    for l in range(L):
        while True:
            flash_1 = np.random.choice(range(1, 7))  # 1-6
            flash_2 = np.random.choice(range(7, 13))  # 7-12
            if (flash_1, flash_2) not in selected_target_flash_pairs:
                target_flashes[l] = [flash_1, flash_2]
                selected_target_flash_pairs.add((flash_1, flash_2))
                break

    # Non-target flashes
    nontarget_flashes = np.zeros((L, I, J - 2), dtype=int)
    for l in range(L):
        available_flashes_1_6 = list(set(range(1, 7)) - {target_flashes[l, 0]})
        available_flashes_7_12 = list(set(range(7, 13)) - {target_flashes[l, 1]})
        available_flashes = available_flashes_1_6 + available_flashes_7_12
        for i in range(I):
            nontarget_flashes[l, i] = available_flashes

    Flash_0 = nontarget_flashes.flatten()

    # Target flashes
    Flash_1 = np.empty(L * I * int(J / 6))
    for l in range(L):
        for i in range(I):
            index = l * I * int(J / 6) + i * int(J / 6)
            Flash_1[index] = target_flashes[l, 0]
            Flash_1[index + 1] = target_flashes[l, 1]

    # Create channel column
    Channel_1 = np.tile(np.arange(1, E + 1), L * I * int(J / 6))
    Channel_0 = np.tile(np.arange(1, E + 1), L * I * int(5 * J / 6))

    # Create DataFrame
    df_1 = pd.DataFrame(X_Y_1.reshape(-1, T), columns=[f'Feature_{i + 1}' for i in range(T)])
    df_0 = pd.DataFrame(X_Y_0.reshape(-1, T), columns=[f'Feature_{i + 1}' for i in range(T)])

    df_1['Y'] = np.repeat(Y_1, E)
    df_1['Letter'] = np.repeat(Letter_1, E)
    df_1['Sequence'] = np.repeat(Sequence_1, E)
    df_1['Flash'] = np.repeat(Flash_1, E)
    df_1['Channel'] = Channel_1

    df_0['Y'] = np.repeat(Y_0, E)
    df_0['Letter'] = np.repeat(Letter_0, E)
    df_0['Sequence'] = np.repeat(Sequence_0, E)
    df_0['Flash'] = np.repeat(Flash_0, E)
    df_0['Channel'] = Channel_0

    df = pd.concat([df_1, df_0], axis=0).reset_index(drop=True)

    return df
