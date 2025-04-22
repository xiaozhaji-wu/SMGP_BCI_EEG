import numpy as np
import scipy.io as sio
from mne.viz import plot_topomap, plot_brain_colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use("bmh")

channel_name_short = [
    'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz'
]

position_x = np.array([-48, 0, 48, -87, -63, 0, 63, 87,
                       -59, 59, -48, 0, 48, -51, 51, 0])
position_y = np.array([59, 63, 59, 0, 0, 0, 0, 0,
                       -31, -31, -59, -63, -59, -71, -71, -87])
position_2d = np.stack([position_x, position_y], axis=1)

bottom_caption_color = 'black'


persons = ["K106", "K107", "K113", "K114",
           "K145", "K151", "K154", "K159",
           "K160", "K172", "K178", "K183",
           "K184", "K185", "K190", "K191"]

for person in persons:
    xdawn_filter_obj = sio.loadmat(
        f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/{person}_TRN_xDAWN_filter.mat")
    xdawn_spatial = xdawn_filter_obj['patterns']
    xdawn_filter = xdawn_filter_obj['filters']
    xdawn_spatial_min = np.round(np.min(xdawn_spatial, axis=0), decimals=1) - 0.1
    xdawn_spatial_max = np.round(np.max(xdawn_spatial, axis=0), decimals=1) + 0.1
    xdawn_spatial_mean = (xdawn_spatial_min + xdawn_spatial_max) / 2

    xdawn_filter_min = np.round(np.min(xdawn_filter, axis=0), decimals=1) - 0.1
    xdawn_filter_max = np.round(np.max(xdawn_filter, axis=0), decimals=1) + 0.1
    xdawn_filter_mean = (xdawn_filter_min + xdawn_filter_max) / 2

    cmap_option = 'Wistia'
    # z_vec_threshold_bool = np.sum((z_vec_mean >= z_threshold))
    fig0, ax0 = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    for e in range(2):
        plot_topomap(data=xdawn_spatial[e, :], pos=position_2d, ch_type='eeg', cmap=cmap_option,
                     names=channel_name_short, size=4, show=False, axes=ax0[e, 0])
        divider_e_1 = make_axes_locatable(ax0[e, 0])
        cax_e_1 = divider_e_1.append_axes('right', size='5%', pad=0.5)
        cbar_e_1 = plot_brain_colorbar(cax_e_1,
                                       clim=dict(kind='value',
                                                 lims=[xdawn_spatial_min[e], xdawn_spatial_mean[e],
                                                       xdawn_spatial_max[e]]),
                                       orientation='vertical',
                                       colormap=cmap_option, label='')
        ax0[e, 0].set_title('')

        plot_topomap(data=xdawn_filter[e, :], pos=position_2d, ch_type='eeg', cmap=cmap_option,
                     names=channel_name_short, size=4, show=False, axes=ax0[e, 1])
        divider_e_2 = make_axes_locatable(ax0[e, 1])
        cax_e_2 = divider_e_2.append_axes('right', size='5%', pad=0.5)
        cbar_e_2 = plot_brain_colorbar(cax_e_2,
                                       clim=dict(kind='value',
                                                 lims=[xdawn_filter_min[e], xdawn_filter_mean[e],
                                                       xdawn_filter_max[e]]),
                                       orientation='vertical',
                                       colormap=cmap_option, label='')
        ax0[e, 1].set_title('')

    # ax0[0, 0].set_title('BSM-Mixture', fontsize=11)
    fig0.text(0.28, 0.01, 'Spatial Pattern', ha='center', size=12, color=bottom_caption_color)
    fig0.text(0.72, 0.01, 'Spatial Filter', ha='center', size=12, color=bottom_caption_color)
    # plt.show()

    fig0.savefig(f"E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/R_plots/xDWAN_skull_figure.png",
                 bbox_inches='tight', dpi=300)

    del fig0

