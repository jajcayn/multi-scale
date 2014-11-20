import cPickle
import matplotlib.pyplot as plt
import numpy as np


def render(diffs, meanvars, stds = None, subtit = '', percentil = None, phase = None, fname = None):
    fig, ax1 = plt.subplots(figsize=(13,8))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    if len(diffs) > 3:
        ax1.plot(diffs, color = '#403A37', linewidth = 2, figure = fig)
    else:
        p2, = ax1.plot(diffs[1], color = '#899591', linewidth = 1.5, figure = fig)
        if stds is not None:
            ax1.plot(diffs[1] + stds[0], color = '#899591', linewidth = 0.7, figure = fig)
            ax1.plot(diffs[1] - stds[0], color = '#899591', linewidth = 0.7, figure = fig)
            ax1.fill_between(np.arange(0,diffs[1].shape[0],1), diffs[1] + stds[0], diffs[1] - stds[0],
                             facecolor = "#899591", alpha = 0.5)
        p1, = ax1.plot(diffs[0], color = '#403A37', linewidth = 2, figure = fig)
        if percentil != None:
            for pos in np.where(percentil[:, 0] == True)[0]:
                ax1.plot(pos, diffs[0][pos], 'o', markersize = 8, color = '#403A37')
    ax1.axis([0, cnt-1, diff_ax[0], diff_ax[1]])
    ax1.set_xlabel('middle year of %.2f-year wide window' % (WINDOW_LENGTH / 365.25), size = 14)
    ax1.set_ylabel('difference in cond. means in temperature [$^{\circ}$C]', size = 14)
    plt.xticks(np.arange(0, cnt+8, 8), np.arange(first_mid_year, last_mid_year+8, 8), rotation = 30)
    ax2 = ax1.twinx()
    if len(meanvars) > 3:
        ax2.plot(meanvars, color = '#CA4F17', linewidth = 2, figure = fig) # color = '#CA4F17'
    else:
        p4, = ax2.plot(meanvars[1], color = '#64C4A0', linewidth = 1.5, figure = fig)
        if stds is not None:
            ax2.plot(meanvars[1] + stds[1], color = '#64C4A0', linewidth = 0.7, figure = fig)
            ax2.plot(meanvars[1] - stds[1], color = '#64C4A0', linewidth = 0.7, figure = fig)
            ax2.fill_between(np.arange(0,diffs[1].shape[0],1), meanvars[1] + stds[1], meanvars[1] - stds[1],
                             facecolor = "#64C4A0", alpha = 0.5)
        p3, = ax2.plot(meanvars[0], color = '#CA4F17', linewidth = 2, figure = fig)
        if percentil != None:
            for pos in np.where(percentil[:, 1] == True)[0]:
                ax2.plot(pos, meanvars[0][pos], 'o', markersize = 8, color = '#CA4F17')
        ax2.set_ylabel('mean of cond. means in temperature [$^{\circ}$C]', size = 14)
        ax2.axis([0, cnt-1, mean_ax[0], mean_ax[1]])
        for tl in ax2.get_yticklabels():
            tl.set_color('#CA4F17')
        if len(diffs) < 3:
            plt.legend([p1, p2, p3, p4], ["difference DATA", "difference SURROGATE mean", "mean DATA", "mean SURROGATE mean"], loc = 2)
    tit = 'SURR: Evolution of difference in cond'
    tit += (' mean in temp, ')
    tit += 'SATA, '
    if np.int(WINDOW_LENGTH) == WINDOW_LENGTH:
        tit += ('%d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    else:
        tit += ('%.2f-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    #plt.title(tit)
    tit = ('Evolution of difference in cond. means temp SATA -- Praha-Klementinum, Czech Republic \n %s' % (''.join([mons[m-1] for m in SEASON]) if SEASON != None else ''))
    tit += subtit
    plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()



diff_ax = (0, 5)
mean_ax = (-1, 1.5)
WINDOW_LENGTH = 13462 # 13462, 16384
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
seas = [[12, 1, 2], [6, 7, 8]]
mons = {0: 'J', 1: 'F', 2: 'M', 3: 'A', 4: 'M', 5: 'J', 6: 'J', 7: 'A', 8: 'S', 9: 'O', 10: 'N', 11: 'D'}
first_mid_year = 1856
last_mid_year = 1991


with open('data_temp/PRGlong_1000MFevolution.bin', 'rb') as f:
    data = cPickle.load(f)

for k, v in data.iteritems():
    locals()[k] = v

fn = ("debug/PRGlong1000MFevolving.png")  
SEASON = None  
render([difference_data, difference_surr], [meanvar_data, meanvar_surr], [difference_surr_std, meanvar_surr_std],
            subtit = ("95 percentil: difference - %d/%d and mean %d/%d" % (difference_95perc[difference_95perc == True].shape[0], cnt, mean_95perc[mean_95perc == True].shape[0], cnt)),
            percentil = where_percentil, fname = fn)

for se in seas:
    SEASON = ''.join([mons[m-1] for m in se])
    with open('data_temp/PRGlong_1000MFevolution%s.bin' % SEASON, 'rb') as f:
        locals()['data' + SEASON] = cPickle.load(f)



