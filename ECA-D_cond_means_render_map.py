"""
created on June 13, 2014

@author: Nikola Jajcay
"""

import cPickle
#import hickle as hkl
from datetime import datetime, date
from matplotlib import rc
rc('ps',usedistiller='xpdf')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d

def moving_average_2d(data, window):
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a numpy array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)

    # The output array has the same dimensions as the input data 
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    return convolve2d(data, window, mode='same', boundary='symm')


def render_differences_map(diffs, lats, lons, subtit = '', fname = None):
    fig = plt.figure(figsize=(20,16))
    lat_ndx = np.argsort(lats)
    lats = lats[lat_ndx]
    m = Basemap(projection = 'merc',
                llcrnrlat = lats[0], urcrnrlat = lats[-1],
                llcrnrlon = lons[0], urcrnrlon = lons[-1],
                resolution = 'i')
                
    m.fillcontinents(color = "#ECF0F3", lake_color = "#A9E5FF", zorder = 0)
    m.drawmapboundary(fill_color = "#A9E5FF")
    m.drawcoastlines(linewidth = 2, color = "#333333")
    m.drawcountries(linewidth = 1.5, color = "#333333")
    m.drawparallels(np.arange(20, 80, 10), linewidth = 1.2, labels = [1,0,0,0], color = "#222222", size = 30)
    m.drawmeridians(np.arange(-40, 80, 10), linewidth = 1.2, labels = [0,0,0,1], color = "#222222", size = 30)
    x, y = m(*np.meshgrid(lons, lats))
    # if not MEANS:
    #     levs = np.arange(0.,1.,0.05) # 0.5 - 6 / 0.25
    # else:
    #     levs = np.arange(0.1,30.1,0.1) # 0 - 4 / 0.2
    if ECA:
        cs = m.contourf(x, y, diffs, 40, cmap = plt.get_cmap('CMRmap'))
    else:
        cs = m.contourf(x, y, diffs[::-1, :], levels = levs, cmap = plt.get_cmap('CMRmap'))
    # cbar = m.colorbar(cs, location = 'right', size = "5%", pad = "10%")
    cbar = plt.colorbar(cs, pad = 0.07, shrink = 0.8, fraction = 0.05)
    cbar.ax.tick_params(labelsize = 30)
    if MEANS:
        cbar.set_label("DIFFERENCE/SD", size = 38, labelpad = 30)
    else:
        cbar.set_label("differecnce in standard deviation [$^{\circ}$C]", size = 23)
    if SIGN:
        if MEANS:
            title = ("%s reanalysis - differences in cond. mean SATA DJF \n %d %s surrogates" % ('ECA&D' if ECA else 'ERA-40', num_surr, SURR_TYPE))
        else:
            title = ("%s reanalysis - differences of conditional standard deviation \n %d %s surrogates" % ('ECA & D' if ECA else 'ERA-40', num_surr, SURR_TYPE))        
    else:
        if MEANS:
            # title = ("%s reanalysis - differences in cond. mean SATA \n DATA" % ('ECA&D' if ECA else 'ERA-40'))
            # title = ("%s reanalysis - scaled mean of bins \n SAT amplitude - DATA" % ('ECA & D' if ECA else 'ERA-40'))
            title = "DIFFERENCE div SD SATA"
        else:
            title = ("%s reanalysis - differences of conditional standard deviation \n MF SURROGATE STD" % ('ECA & D' if ECA else 'ERA-40'))
    title += subtit
    plt.title(title, size = 35)
    
    if fname != None:
        plt.savefig(fname)
    else:
        plt.show()


ECA = True
SURR_TYPE = 'FT' # MF, FT or AR
START_DATE = date(1958,1,1)
MEANS = True
ANOMALISE = True
PICKLE = True # whether to use pickled file or hickled
SIGN = False # wheter to check significance or just plot results
SIGMAS_ABOVE = 2
PERCENTIL = 95
SAME_BINS = False
CONDITION = False
NUM_FILES = 5
LATS = [35, 65]
LONS = [-12.5, 40]
FALSE_POS = False



# load data 
print("[%s] Loading data..." % (str(datetime.now())))
# if ECA:
#     fname = ('result/new_grl/ECA-D_%s_cond_mean_var_data_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', 
#                                                               str(START_DATE)))
# else:
#     fname = ('result/ERA_%s_cond_mean_var_data_from_%s_16k_OLD' % ('SATA' if ANOMALISE else 'SAT', 
#                                                               str(START_DATE)))
if PICKLE:
    fname = "result/grl-resub/GRL-RESUBMISSION-sata-w-sd"
    with open(fname + '.bin', 'rb') as f:
        data = cPickle.load(f)
else:
    data = hkl.load(fname + '.hkl')
print data.keys()
bins_data = data['bins_data']
bins_data_var = data['bins_data_var']
lats = data['lats']
lons = data['lons']
sd_data = data['std']
del data

# load surrogates
# bins_surrogates_list = []
# bins_surrogates_var_list = []
# print("[%s] Data loaded. Now loading surrogates..." % (str(datetime.now())))
# if ECA:
#     fname = ('result/new_grl/ECA-D_%s_cond_mean_var_%ssurrogates_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', 
#                  SURR_TYPE, str(START_DATE)))
# else:
#     fname = ('result/ERA_%s_cond_mean_var_%ssurrogates_from_%s_16k_OLD' % ('SATA' if ANOMALISE else 'SAT', 
#                  SURR_TYPE, str(START_DATE)))
# if PICKLE:
#     for i in range(NUM_FILES):
#         with open(fname + '_%d' % (i) + '.bin', 'rb') as f:
#             data = cPickle.load(f)
#         bins_surrogates_list.append(data['bins_surrogates'])
#         bins_surrogates_var_list.append(data['bins_surrogates_var'])
#     with open("result/ECA-D_SATamplitude_SATA_cond_mean_var_data_from_1958-01-01_16kOLD.bin", 'rb') as f:
#         data = cPickle.load(f)
#         lat_surrs_large = data['lats']
#         lons_surrs_large = data['lons']

# else:
#     data = hkl.load(fname + '.hkl')
# del data
# print("[%s] Surrogates loaded." % (str(datetime.now())))
# bins_surrogates = np.zeros(([NUM_FILES * bins_surrogates_list[0].shape[1]] + list(bins_surrogates_list[0].shape[2:])))
# pointer = 0
# for i in range(NUM_FILES):
#     lat_ndx = np.nonzero(np.logical_and(lat_surrs_large >= LATS[0], lat_surrs_large <= LATS[1]))[0]
#     lon_ndx = np.nonzero(np.logical_and(lons_surrs_large >= LONS[0], lons_surrs_large <= LONS[1]))[0]
#     if bins_surrogates_list[i].shape[2] > 120:
#         a = bins_surrogates_list[i][0, ...]
#         e = a[:, lat_ndx, :, :]
#         bins_surrogates[pointer:pointer+100, ...] = e[:, :, lon_ndx, :]
#     else:
#         bins_surrogates[pointer:pointer+100, ...] = bins_surrogates_list[i][0, ...]
#     pointer += bins_surrogates_list[0].shape[1]
# del bins_surrogates_list
# bins_surrogates_var = np.zeros_like(bins_surrogates)
# pointer = 0
# # for i in range(NUM_FILES):
# #     bins_surrogates_var[pointer:pointer+100, ...] = bins_surrogates_var_list[i][0, ...]
# #     pointer += bins_surrogates_var_list[0].shape[1]
# del bins_surrogates_var_list
# print("[%s] Data prepared to test and plot..." % (str(datetime.now())))

# lat_ndx = np.nonzero(np.logical_and(lats >= LATS[0], lats <= LATS[1]))[0]
# lon_ndx = np.nonzero(np.logical_and(lons >= LONS[0], lons <= LONS[1]))[0]
# d = bins_data[lat_ndx, ...]
# bins_data = d[:, lon_ndx, ...]
# e = bins_surrogates[:, lat_ndx, ...]
# bins_surrogates = e[:, :, lon_ndx, :]
# lats = lats[lat_ndx]
# lons = lons[lon_ndx]


# print bins_data.shape
# print bins_surrogates.shape



if SIGN:
    # SU = 2 # 0 - MF, 1 - FT, 2 - AR
    # compute significance
    result_sigma = np.zeros((bins_data.shape[0], bins_data.shape[1]))
    result_percentil = np.zeros_like(result_sigma)
    num_surr = bins_surrogates.shape[0]
    if FALSE_POS:
        sur_no = np.random.randint(0, bins_surrogates.shape[0])
        print sur_no
    for lat in range(lats.shape[0]):
        for lon in range(lons.shape[0]):
            if MEANS:
                if np.any(np.isnan(bins_data[lat, lon, ...])): # if on lat x lon is NaN
                    result_sigma[lat, lon] = np.nan
                    result_percentil[lat, lon] = np.nan
                else:
                    # sigma-based significance
                    if not FALSE_POS:
                        if SAME_BINS:
                            ma = bins_data[lat, lon, :].argmax()
                            mi = bins_data[lat, lon, :].argmin()
                            diff_data = bins_data[lat, lon, ma] - bins_data[lat, lon, mi]
                        else:
                            diff_data = bins_data[lat, lon, :].max() - bins_data[lat, lon, :].min()
                    else:
                        if SAME_BINS:
                            ma = bins_surrogates[sur_no, lat, lon, :].argmax()
                            mi = bins_surrogates[sur_no, lat, lon, :].argmin()
                            diff_data = bins_surrogates[sur_no, lat, lon, ma] - bins_surrogates[sur_no, lat, lon, mi]
                        else:
                            diff_data = bins_surrogates[sur_no, lat, lon, :].max() - bins_surrogates[sur_no, lat, lon, :].min()
                    # scaled
                    # diff_data /= np.mean(bins_data[lat, lon, :])
                    diff_surrs = np.zeros((num_surr))
                    if CONDITION:
                        cnt = 0
                    for i in range(num_surr):
                        if SAME_BINS:
                            diff_surrs[i] = bins_surrogates[i, lat, lon, ma] - bins_surrogates[i, lat, lon, mi]
                        elif CONDITION:
                            ma_surr = bins_surrogates[i, lat, lon, :].argmax()
                            mi_surr = bins_surrogates[i, lat, lon, :].argmin()
                            if (np.abs(ma_surr - mi_surr) > 2) and (np.abs(ma_surr - mi_surr) < 6):
                                diff_surrs[cnt] = bins_surrogates[i, lat, lon, ma_surr] - bins_surrogates[i, lat, lon, mi_surr]
                                cnt += 1
                        else:
                            diff_surrs[i] = bins_surrogates[i, lat, lon, :].max() - bins_surrogates[i, lat, lon, :].min()
                        # scaled
                        # diff_surrs[i] /= np.mean(bins_surrogates[i, lat, lon, :])
                    if CONDITION:
                        diff_surrs = np.delete(diff_surrs, np.s_[cnt:])
                    sigma = np.std(diff_surrs, axis = 0, ddof = 1)
                    mean = np.mean(diff_surrs, axis = 0)
                    if diff_data >= mean + SIGMAS_ABOVE*sigma:
                        result_sigma[lat, lon] = diff_data
                    else:
                        result_sigma[lat, lon] = np.nan # or np.nan
                        
                    # percentil-based significance
                    greater_mat = np.greater(diff_data, diff_surrs)
                    if not CONDITION:
                        cnt = num_surr
                    if np.sum(greater_mat) > PERCENTIL/100. * cnt:
                        result_percentil[lat, lon] = diff_data
                    else:
                        result_percentil[lat, lon] = np.nan # or np.nan
            # var
            else:
                if np.any(np.isnan(bins_data_var[lat, lon])): # if on lat x lon is NaN
                    result_sigma[lat, lon] = np.nan
                    result_percentil[lat, lon] = np.nan
                else:
                    # sigma-based significance
                    if SAME_BINS:
                        ma = bins_data_var[lat, lon, :].argmax()
                        mi = bins_data_var[lat, lon, :].argmin()
                        diff_data = bins_data_var[lat, lon, ma] - bins_data_var[lat, lon, mi]
                    else:
                        diff_data = bins_data_var[lat, lon, :].max() - bins_data_var[lat, lon, :].min()
                    # scaled
                    diff_data /= np.mean(bins_data[lat, lon, :])
                    diff_surrs = np.zeros((num_surr))
                    if CONDITION:
                        cnt = 0
                    for i in range(num_surr):
                        if SAME_BINS:
                            diff_surrs[i] = bins_surrogates_var[i, lat, lon, ma] - bins_surrogates_var[i, lat, lon, mi]
                        elif CONDITION:
                            ma_surr = bins_surrogates_var[i, lat, lon, :].argmax()
                            mi_surr = bins_surrogates_var[i, lat, lon, :].argmin()
                            if (np.abs(ma_surr - mi_surr) > 2) and (np.abs(ma_surr - mi_surr) < 6):
                                diff_surrs[cnt] = bins_surrogates_var[i, lat, lon, ma_surr] - bins_surrogates_var[i, lat, lon, mi_surr]
                                cnt += 1
                        else:
                            diff_surrs[i] = bins_surrogates_var[i, lat, lon, :].max() - bins_surrogates_var[i, lat, lon, :].min()
                        # scaled
                        diff_surrs[i] /= np.mean(bins_surrogates[i, lat, lon, :])
                    if CONDITION:
                        diff_surrs = np.delete(diff_surrs, np.s_[cnt:])
                    sigma = np.std(diff_surrs, axis = 0, ddof = 1)
                    mean = np.mean(diff_surrs, axis = 0)
                    if diff_data >= mean + SIGMAS_ABOVE*sigma:
                        result_sigma[lat, lon] = diff_data
                    else:
                        result_sigma[lat, lon] = np.nan # or np.nan
                        
                    # percentil-based significance
                    greater_mat = np.greater(diff_data, diff_surrs)
                    if np.sum(greater_mat) > PERCENTIL/100. * num_surr:
                        result_percentil[lat, lon] = diff_data
                    else:
                        result_percentil[lat, lon] = np.nan # or np.nan              
    # if not ECA:
    #     if SU == 0:
    #         SURR_TYPE = 'MF'
    #     elif SU == 1:
    #         SURR_TYPE = 'FT'
    #     elif SU == 2:
    #         SURR_TYPE = 'AR'
    # print('total grid points: %d -- not significant grid points: %d' % (np.prod(result_sigma.shape), np.sum(np.isnan(result_sigma))))
    # fname = ('debug/%s_SATamplitude_%s_scaled_%s_bins_%ssurrogates_from_%s_16k_above_%.1fsigma%s%s.png' % ('ECA-D' if ECA else 'ERA', 'SATA' if ANOMALISE else 'SAT', 
    #              'means' if MEANS else 'std', SURR_TYPE, str(START_DATE), SIGMAS_ABOVE, '_same_bins' if SAME_BINS else '', 
    #              '_condition' if CONDITION else ''))
    # render_differences_map(result_sigma, lats, lons, subtit = (' - above %.2f $\sigma$ (STDs) %s' % 
    #                         (SIGMAS_ABOVE, '- SAME BINS' if SAME_BINS else '- CONDITION' if CONDITION else '')), fname = fname)
    
    # fname = ('debug/%s_SATamplitude_%s_scaled_%s_bins_%ssurrogates_from_%s_16k_above_%dpercentil%s%s.png' % ('ECA-D' if ECA else 'ERA', 'SATA' if ANOMALISE else 'SAT', 
    #              'means' if MEANS else 'std', SURR_TYPE, str(START_DATE), PERCENTIL, '_same_bins' if SAME_BINS else '', 
    #              '_condition' if CONDITION else ''))
    fname = ('debug/ECA-D_SATA_DJF_500FTsurrogates_above_%dpercentil_%s.eps' % (PERCENTIL, 'false_pos' if FALSE_POS else 'TEST'))
    # NaNs to 0
    mask = np.isnan(result_percentil)
    result_percentil[mask] = 0.
    # # smooth
    # result_percentil = scipy.ndimage.zoom(result_percentil, 15)
    # lats = scipy.ndimage.zoom(lats, 15)
    # lons = scipy.ndimage.zoom(lons, 15)
    # mask = scipy.ndimage.zoom(mask, 3)
    # result_percentil[mask] = np.nan
    # result_percentil = scipy.ndimage.gaussian_filter(result_percentil, sigma = 2.0, order = 0)
    # win = np.ones((1, 2)) 
    # result_percentil = moving_average_2d(result_percentil, win)
    # result_percentil[mask] = np.nan
    to_txt = np.zeros((lats.shape[0] * lons.shape[0], 3))
    for lat in range(lats.shape[0]):
        for lon in range(lons.shape[0]):
            to_txt[lat*lons.shape[0] + lon, 0] = lats[lat]
            to_txt[lat*lons.shape[0] + lon, 1] = lons[lon]
            to_txt[lat*lons.shape[0] + lon, 2] = result_percentil[lat, lon]
    np.savetxt('debug/ECA-D_SATA_DJF_FT_%dpercentil.txt' % PERCENTIL, to_txt, fmt = '%.3f')
    render_differences_map(result_percentil, lats, lons, subtit = (' - %d percentil %s' % 
                            (PERCENTIL, '- SAME BINS' if SAME_BINS else '- CONDITION' if CONDITION else '')), fname = fname)
    
else:
    if ECA:
        # fname = ('debug/ECA-D_SATamplitude_%s_cond_%s_data_from_%s.png' % ('SATA' if ANOMALISE else 'SAT', 'means' if MEANS else 'std', 
        #                                                                str(START_DATE)))
        fname = ('debug/ECA-D_%s_scaled_%s_bins_data_from_%s.png' % ('SATA' if ANOMALISE else 'SAT', 'means' if MEANS else 'std', 
                                                                       str(START_DATE)))
    else:
        fname = ('debug/ERA_%s_cond_%s_MFsurrogate_std_from_%s.png' % ('SATA' if ANOMALISE else 'SAT', 'means' if MEANS else 'std', 
                                                                       str(START_DATE)))
    if MEANS:
        result = np.zeros((bins_data.shape[0], bins_data.shape[1]))
        result_data = np.zeros_like(result)
        amp = np.zeros_like(result)
        for lat in range(lats.shape[0]):
            for lon in range(lons.shape[0]):
                result[lat, lon] = bins_data[lat, lon, :].max() - bins_data[lat, lon].min()
                # result = scipy.ndimage.zoom(result, 3)
                # result[lat, lon] = np.mean(bins_data[lat, lon, :])
                # result[lat, lon] /= np.mean(bins_data[lat, lon, :])
                # result[lat, lon] = np.mean([np.mean(bins_surrogates[i, lat, lon, :]) for i in range(bins_surrogates.shape[0])])
                amp[lat, lon] = sd_data[lat, lon]
        result /= amp
        render_differences_map(result, lats, lons, subtit = (' - no significance test'), 
                                fname = fname)
    # else:
    #     render_differences_map(np.std(difference_surrogates_var[0, ...], axis = 0, ddof = 1), lats, lons, subtit = (' - no significance test'), 
    #                             fname = fname)

