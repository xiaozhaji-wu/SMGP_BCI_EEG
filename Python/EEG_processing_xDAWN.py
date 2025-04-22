import numpy as np
from pyriemann.spatialfilters import Xdawn
import scipy.io as sio
import matplotlib

matplotlib.use('Agg')

def pre_process_xdawn_train(train_signal, train_label, test_signal, signal_length,
                            E_sub=16, n_filters=2, reshape_3d_bool=True):
    xdawn_min = min(E_sub, n_filters)
    train_signal_size = train_signal.shape[0]
    test_signal_size = test_signal.shape[0]
    xdawn_obj = Xdawn(nfilter=n_filters, estimator='scm')
    xdawn_obj.fit(train_signal, train_label)

    filters = xdawn_obj.filters_
    filters = filters[0:n_filters, :]
    patterns = xdawn_obj.patterns_
    patterns = patterns[0:n_filters, :]
    # combine filters and patterns into a dictionary
    xdawn_filter_obj = {'filters': filters, 'patterns': patterns}

    train_signal_xdawn = xdawn_obj.transform(train_signal)[:, :xdawn_min, :]
    train_signal_xdawn = np.reshape(train_signal_xdawn, [train_signal_size, xdawn_min * signal_length])
    if reshape_3d_bool:
        train_signal_xdawn = np.reshape(train_signal_xdawn, [train_signal_size, xdawn_min, signal_length])

    test_signal_xdawn = xdawn_obj.transform(test_signal)[:, :xdawn_min, :]
    test_signal_xdawn = np.reshape(test_signal_xdawn, [test_signal_size, xdawn_min * signal_length])
    if reshape_3d_bool:
        test_signal_xdawn = np.reshape(test_signal_xdawn, [test_signal_size, xdawn_min, signal_length])

    return  train_signal_xdawn, test_signal_xdawn, xdawn_filter_obj

persons = ["K106", "K107", "K113", "K114",
           "K145", "K151", "K154", "K159",
           "K160", "K172", "K178", "K183",
           "K184", "K185", "K190", "K191"]

for person in persons:

    train_data = sio.loadmat(f'data/EEG_multi/{person}_TRN.mat')
    train_X = train_data['X']
    train_X = train_X.reshape(train_X.shape[0], 16, 25)
    train_Y = train_data['Y']
    train_Y = train_Y.flatten()
    train_Letter = train_data['Letter']
    train_Sequence = train_data['Sequence']
    train_Flash = train_data['Flash']
    train_Letter_text = train_data['Letter_text']

    test_data = sio.loadmat(f'data/EEG_multi/{person}_FRT.mat')
    test_X = test_data['X']
    test_X = test_X.reshape(test_X.shape[0], 16, 25)
    test_Y = test_data['Y']
    test_Y = test_Y.flatten()
    test_Letter = test_data['Letter']
    test_Sequence = test_data['Sequence']
    test_Flash = test_data['Flash']
    test_Letter_text = test_data['Letter_text']

    train_signal, test_signal, xdawn_filter_obj = pre_process_xdawn_train(train_X, train_Y, test_X, 25, E_sub=16, n_filters=2, reshape_3d_bool=False)

    # save the xDAWN filter object dictionary to a .mat file
    sio.savemat(f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/{person}_TRN_xDAWN_filter.mat", xdawn_filter_obj)

    sio.savemat(f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/{person}_TRN_xDAWN.mat", {
        'X': train_signal,
        'Y': train_Y.reshape(-1, 1),
        'Letter': train_Letter,
        'Sequence': train_Sequence,
        'Flash': train_Flash,
        'Letter_text': train_Letter_text
    })

    sio.savemat(f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/{person}_FRT_xDAWN.mat", {
        'X': test_signal,
        'Y': test_Y.reshape(-1, 1),
        'Letter': test_Letter,
        'Sequence': test_Sequence,
        'Flash': test_Flash,
        'Letter_text': test_Letter_text
    })

    sio.savemat(f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi_ref/{person}/{person}_TRN_xDAWN.mat", {
        'X': train_signal,
        'Y': train_Y.reshape(-1, 1),
        'Letter': train_Letter,
        'Sequence': train_Sequence,
        'Flash': train_Flash,
        'Letter_text': train_Letter_text
    })

    sio.savemat(f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi_ref/{person}/{person}_FRT_xDAWN.mat", {
        'X': test_signal,
        'Y': test_Y.reshape(-1, 1),
        'Letter': test_Letter,
        'Sequence': test_Sequence,
        'Flash': test_Flash,
        'Letter_text': test_Letter_text
    })


