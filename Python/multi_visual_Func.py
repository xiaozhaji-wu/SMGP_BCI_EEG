import pandas as pd
import numpy as np
from scipy.stats import matrix_normal, norm


def prob_score(odf, X, Mu_1_mean, Mu_0_mean, C_e_mean, C_t_mean, E):
    prob_target = []
    prob_non_target = []

    for i in range(0, X.shape[0], E):
        row_matrix = X[i:i + E, :]

        s1 = matrix_normal.logpdf(row_matrix, mean=Mu_1_mean, rowcov=C_e_mean, colcov=C_t_mean)
        s0 = matrix_normal.logpdf(row_matrix, mean=Mu_0_mean, rowcov=C_e_mean, colcov=C_t_mean)

        prob_target.append(s1)
        prob_non_target.append(s0)

    prob_target = np.array(prob_target)
    prob_non_target = np.array(prob_non_target)
    odf['prob_target'] = prob_target
    odf['prob_non_target'] = prob_non_target
    return odf

def prob_score_swLDA(odf, X, b_inmodel, mean_1, mean_0, std, E):
    prob_target = []
    prob_non_target = []

    X_expand_df = pd.DataFrame(X.values.reshape(-1, E * X.shape[1]))
    X_expand = X_expand_df.to_numpy()

    for i in range(0, X_expand.shape[0]):
        row_matrix = X_expand[i, :]
        row_score = np.matmul(np.array(row_matrix), b_inmodel)

        s1 = norm.logpdf(row_score, loc=mean_1, scale=std)
        s0 = norm.logpdf(row_score, loc=mean_0, scale=std)

        prob_target.append(s1)
        prob_non_target.append(s0)

    prob_target = np.array(prob_target)
    prob_non_target = np.array(prob_non_target)
    odf['prob_target'] = prob_target
    odf['prob_non_target'] = prob_non_target
    return odf

def true_f(odf, unique_letters):
    true_flashes = []

    for letter in unique_letters:
        i = int(odf[(odf['Letter'] == letter) & (odf['Sequence'] == 1) & (odf["Y"] == 1)]['Flash'].values[0])
        j = int(odf[(odf['Letter'] == letter) & (odf['Sequence'] == 1) & (odf["Y"] == 1)]['Flash'].values[1])
        if i > j:
            i, j = j, i
        flash_combination = f"flash:{i},{j}"
        true_flashes.append({'Letter': letter, 'Flash_Combination': flash_combination})

    true_flashes = pd.DataFrame(true_flashes)
    return true_flashes

def cum_prob_score(df_sub, unique_letters, unique_sequences):
    cum_prob_score_dic = {f"flash:{i},{j}": [] for i in range(1, 7) for j in range(7, 13)}
    cum_prob_score_dic['Letter'] = []
    cum_prob_score_dic['Sequence'] = []

    for letter in unique_letters:
        for sequence in unique_sequences:
            subset = df_sub[(df_sub['Letter'] == letter) & (df_sub['Sequence'] == sequence)]
            if len(subset) == 12:
                cum_prob_score = {}
                for i in range(1, 7):
                    for j in range(7, 13):
                        cum_prob_score[f'flash:{i},{j}'] = subset[(subset['Flash'] == i) | (subset['Flash'] == j)][
                                                               'prob_target'].sum() + \
                                                           subset[(subset['Flash'] != i) & (subset['Flash'] != j)][
                                                               'prob_non_target'].sum()
                        cum_prob_score_dic[f"flash:{i},{j}"].append(cum_prob_score[f'flash:{i},{j}'])
                cum_prob_score_dic['Letter'].append(letter)
                cum_prob_score_dic['Sequence'].append(sequence)

    cum_prob_score_df = pd.DataFrame(cum_prob_score_dic)

    return cum_prob_score_df

def max_cum_prob_score(cum_prob_score_df):
    sum_cum_prob_score = pd.DataFrame()
    for i in range(1, 7):
        for j in range(7, 13):
            sum_cum_prob_score[f'flash:{i},{j}'] = cum_prob_score_df.sort_values('Sequence').groupby('Letter')[
                f'flash:{i},{j}'].cumsum()
    sum_cum_prob_score['Letter'] = cum_prob_score_df['Letter']
    sum_cum_prob_score['Sequence'] = cum_prob_score_df['Sequence']
    sum_cum_prob_score = sum_cum_prob_score.sort_index()

    max_cum_prob_score = pd.DataFrame()
    max_columns = sum_cum_prob_score.drop(columns=['Letter', 'Sequence']).idxmax(axis=1)
    max_values = sum_cum_prob_score.drop(columns=['Letter', 'Sequence']).max(axis=1)
    max_cum_prob_score['Letter'] = sum_cum_prob_score['Letter']
    max_cum_prob_score['Sequence'] = sum_cum_prob_score['Sequence']
    max_cum_prob_score['max_flash'] = max_columns
    max_cum_prob_score['max_value'] = max_values

    return max_cum_prob_score, sum_cum_prob_score

def acc_score(max_cum_prob_score, true_flashes, unique_letters, unique_sequences):
    acc_score = []
    for i in unique_sequences:
        prediction_score = 0
        for j in unique_letters:
            true_flash = true_flashes[true_flashes['Letter'] == j]['Flash_Combination'].values[0]
            max_flash = max_cum_prob_score[(max_cum_prob_score['Letter'] == j) & (max_cum_prob_score['Sequence'] == i)][
                'max_flash'].values[0]
            if true_flash == max_flash:
                prediction_score += 1
            else:
                prediction_score += 0
        acc_score.append(prediction_score / len(unique_letters))
    return acc_score

def predict(rdf, Mu_1_mean, Mu_0_mean, C_e_mean, C_t_mean, T, E):
    unique_letters = rdf['Letter'].unique()
    unique_sequences = rdf['Sequence'].unique()
    X = rdf.iloc[:, :T].to_numpy()
    info_data = rdf.iloc[:, T:]
    data = info_data.drop(columns=['Channel'])
    odf = data.drop_duplicates().reset_index(drop=True, inplace=False)

    odf = prob_score(odf, X, Mu_1_mean, Mu_0_mean, C_e_mean, C_t_mean, E)
    true_flashes = true_f(odf, unique_letters)
    cum_prob_score_df = cum_prob_score(odf, unique_letters, unique_sequences)
    max_cum_prob_score_df, sum_cum_prob_score_df = max_cum_prob_score(cum_prob_score_df)
    acc_scores = acc_score(max_cum_prob_score_df, true_flashes, unique_letters, unique_sequences)

    return acc_scores, sum_cum_prob_score_df

def predict_swLDA(rdf, b_inmodel, mean_1, mean_0, std, T, E):
    unique_letters = rdf['Letter'].unique()
    unique_sequences = rdf['Sequence'].unique()
    X = rdf.iloc[:, :T]
    info_data = rdf.iloc[:, T:]
    data = info_data.drop(columns=['Channel'])
    odf = data.drop_duplicates().reset_index(drop=True, inplace=False)

    odf = prob_score_swLDA(odf, X, b_inmodel, mean_1, mean_0, std, E)
    true_flashes = true_f(odf, unique_letters)
    cum_prob_score_df = cum_prob_score(odf, unique_letters, unique_sequences)
    max_cum_prob_score_df, sum_cum_prob_score_df = max_cum_prob_score(cum_prob_score_df)
    acc_scores = acc_score(max_cum_prob_score_df, true_flashes, unique_letters, unique_sequences)

    return acc_scores, sum_cum_prob_score_df
