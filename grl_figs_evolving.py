import cPickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def render(diffs, meanvars, stds = None, subtit = '', percentil = None, phase = None, fname = None):
    fig, ax1 = plt.subplots(figsize=(14,11), dpi = 600)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(color = "#686A69")
    cnt = diffs[0].shape[0]
    if stds is not None:
        ax1.fill_between(np.arange(0,diffs[1].shape[0],1), diffs[1] + stds[0], diffs[1] - stds[0],
                         facecolor = "#CA6D57", edgecolor = "#CA6D57", alpha = 0.5)
    p2, = ax1.plot(diffs[1], color = '#BF3919', linewidth = 1.5, figure = fig)
    p1, = ax1.plot(diffs[0], color = '#76C06E', linewidth = 4, figure = fig) #19BF86
    if percentil != None:
        for pos in np.where(percentil[:, 0] == True)[0]:
            ax1.plot(pos, diffs[0][pos], 'o', markersize = 14, color = '#76C06E')
    ax1.axis([0, cnt-1, diff_ax[0], diff_ax[1]])
    ax1.set_xlabel('TIME [YEAR]', size = 40)
    ax1.set_ylabel('DIFF SATA COND MEAN [$^{\circ}$C]', size = 40)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 28)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    # plt.xticks(np.arange(0, cnt+8, 12), np.arange(first_mid_year, last_mid_year+8, 12), rotation = 30)
    plt.xticks(np.arange(0, cnt+7, 7), np.arange(first_mid_year, last_mid_year+7, 7), rotation = 30)
    # ax1.set_rasterized(True)
    # ax2 = ax1.twinx()
    # if stds is not None:
    #     ax2.fill_between(np.arange(0,diffs[1].shape[0],1), meanvars[1] + stds[1], meanvars[1] - stds[1],
    #                      facecolor = "#B4FFAC", edgecolor = "#B4FFAC", alpha = 0.5)
    # p4, = ax2.plot(meanvars[1], color = '#76C06E', linewidth = 1.5, figure = fig)
    # p3, = ax2.plot(meanvars[0], color = '#C06EA2', linewidth = 2, figure = fig)
    # if percentil != None:
    #     for pos in np.where(percentil[:, 1] == True)[0]:
    #         ax2.plot(pos, meanvars[0][pos], 'o', markersize = 8, color = '#C06EA2')
    # ax2.set_ylabel('mean of cond. means in temperature [$^{\circ}$C]', size = 17)
    # ax2.axis([0, cnt-1, mean_ax[0], mean_ax[1]])
    # ax2.tick_params(axis = 'both', which = 'major', labelsize = 16)
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('#76C06E')
    # # ax2.set_rasterized(True)
    # plt.legend([p1, p2, p3, p4], ["difference DATA", "difference SURROGATE mean", "mean DATA", "mean SURROGATE mean"], loc = 2)
    # tit = ('Praha-Klementinum, Czech Republic -- SAT amplitude \n %s' % (''.join([mons[m-1] for m in SEASON]) if SEASON != None else ''))
    # tit += subtit
    # plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 23, transform = ax1.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()



diff_ax = (0, 6)
mean_ax = (-1, 1.5)
WINDOW_LENGTH = 13462 # 13462, 16384
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
seas = [[12, 1, 2], [6, 7, 8]]
mons = {0: 'J', 1: 'F', 2: 'M', 3: 'A', 4: 'M', 5: 'J', 6: 'J', 7: 'A', 8: 'S', 9: 'O', 10: 'N', 11: 'D'}
first_mid_year = 1915 # for short time series
# first_mid_year = 1856 # for long time series
last_mid_year = 1992


fn = 'LONG--POTS1000FTevolution'
fn_out = 'POTS1000FTevolvingSeasons'

# with open('data_temp/%s.bin' % fn, 'rb') as f:
#     data = cPickle.load(f)

# for k, v in data.iteritems():
#     locals()[k] = v

# fn = ("debug/%s.eps" % fn_out)  
# SEASON = None  
# render([difference_data, difference_surr], [meanvar_data, meanvar_surr], [difference_surr_std, meanvar_surr_std],
#             subtit = ("95 percentil: difference - %d/%d" % (difference_95perc[difference_95perc == True].shape[0], cnt)),
#             percentil = where_percentil, fname = fn)


# to_txt = np.zeros((cnt, 5))
# # first col - continuous year
# to_txt[:, 0] = np.arange(first_mid_year, last_mid_year, 1)
# # second col - data
# to_txt[:, 1] = difference_data
# # third col - mean surr
# to_txt[:, 2] = difference_surr
# # fourth col - std surr
# to_txt[:, 3] = difference_surr_std
# # fifth col - percentil
# to_txt[:, 4] = -50.
# for pos in np.where(where_percentil[:, 0] == True)[0]:
#     to_txt[pos, 4] = difference_data[pos]

# np.savetxt('debug/%s.txt' % fn_out, to_txt, fmt = '%.3f')



for se in seas:
    SEASON = ''.join([mons[m-1] for m in se])
    with open('data_temp/%s%s.bin' % (fn, SEASON), 'rb') as f:
        data = cPickle.load(f)
        for k, v in data.iteritems():
            locals()[k + SEASON] = v


fig = plt.figure(figsize = (14,22), frameon = False, dpi = 600)
gs = gridspec.GridSpec(2, 1)
gs.update(left = 0.1, right = 0.95, top = 0.95, bottom = 0.07, wspace = 0.4, hspace = 0.3)

for i in range(2):
    if i == 0:
        spec = gs[0, 0]
        diffs = [difference_dataDJF, difference_surrDJF]
        meanvars = [meanvar_dataDJF, meanvar_surrDJF]
        stds = [difference_surr_stdDJF, meanvar_surr_stdDJF]
        # subtit = ("95 percentil: difference - %d/%d and mean %d/%d" % (difference_95percDJF[difference_95percDJF == True].shape[0], cnt, mean_95percDJF[mean_95percDJF == True].shape[0], cnt))
        percentil = where_percentilDJF
        cnt = cntDJF
        axtit = 'DJF'
    elif i == 1:
        spec = gs[1, 0]
        diffs = [difference_dataJJA, difference_surrJJA]
        meanvars = [meanvar_dataJJA, meanvar_surrJJA]
        stds = [difference_surr_stdJJA, meanvar_surr_stdJJA]
        # subtit = ("95 percentil: difference - %d/%d and mean %d/%d" % (difference_95perc[difference_95perc == True].shape[0], cnt, mean_95perc[mean_95perc == True].shape[0], cnt))
        percentil = where_percentilJJA
        cnt = cntJJA
        axtit = 'JJA'
    ax = plt.Subplot(fig, spec)
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#686A69')
    ax.set_title(axtit, size = 38)
    if stds is not None:
        ax.fill_between(np.arange(0,diffs[1].shape[0],1), diffs[1] + stds[0], diffs[1] - stds[0],
                         facecolor = "#CA6D57", edgecolor = "#CA6D57", alpha = 0.5)
    p2, = ax.plot(diffs[1], color = '#BF3919', linewidth = 1.5, figure = fig)
    p1, = ax.plot(diffs[0], color = '#76C06E', linewidth = 4, figure = fig)
    if percentil != None:
        for pos in np.where(percentil[:, 0] == True)[0]:
            ax.plot(pos, diffs[0][pos], 'o', markersize = 14, color = '#76C06E')
    ax.axis([0, cnt-1, diff_ax[0], diff_ax[1]])
    ax.set_xlabel('TIME [YEAR]', size = 40)
    ax.set_ylabel('DIFF SATA COND MEAN [$^{\circ}$C]', size = 40)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 28)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    # plt.xticks(np.arange(0, cnt+8, 12), np.arange(first_mid_year, last_mid_year+8, 12), rotation = 30)
    plt.xticks(np.arange(0, cnt+7, 7), np.arange(first_mid_year, last_mid_year+7, 7), rotation = 30)
    # ax2 = ax.twinx()
    # if stds is not None:
    #     ax2.fill_between(np.arange(0,diffs[1].shape[0],1), meanvars[1] + stds[1], meanvars[1] - stds[1],
    #                      facecolor = "#B4FFAC", edgecolor = "#B4FFAC", alpha = 0.5)
    # p4, = ax2.plot(meanvars[1], color = '#76C06E', linewidth = 1.5, figure = fig)
    # p3, = ax2.plot(meanvars[0], color = '#C06EA2', linewidth = 2, figure = fig)
    # if percentil != None:
    #     for pos in np.where(percentil[:, 1] == True)[0]:
    #         ax2.plot(pos, meanvars[0][pos], 'o', markersize = 8, color = '#C06EA2')
    # ax2.set_ylabel('mean of cond. means in temperature [$^{\circ}$C]', size = 17)
    # ax2.axis([0, cnt-1, mean_ax[0], mean_ax[1]])
    # ax2.tick_params(axis = 'both', which = 'major', labelsize = 16)
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('#76C06E')
    # plt.legend([p1, p2, p3, p4], ["difference DATA", "difference SURROGATE mean", "mean DATA", "mean SURROGATE mean"], loc = 2)

# tit = ('Praha-Klementinum, Czech Republic -- SATA seasons \n')
# tit += subtit
# plt.suptitle(tit, size = 34)
fn = ("debug/%s.eps" % fn_out)  
plt.savefig(fn, bbox_inches='tight')