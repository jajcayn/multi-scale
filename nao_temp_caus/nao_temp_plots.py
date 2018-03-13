import cPickle
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ipython')


def get_zscore(a, sur):
    assert a.shape[0] == sur.shape[1]
    return (a - np.mean(sur, axis=0))/np.std(sur, axis=0, ddof=1)

def get_pval(a, sur):
    assert a.shape[0] == sur.shape[1]
    return 1 - np.sum(np.greater_equal(a, sur), axis=0) / sur.shape[0]

with open("PRG_NAO-temp_caus_100FT.bin", "rb") as f:
    raw = cPickle.load(f)
data = raw['data']
nsurrs = raw['NAOsurrs']
tsurrs = raw['TGsurrs']
tau = raw['taus']

print data.shape, nsurrs.shape

i = 1 # 0 or 1
i_tit = ['3dim', '4dim (pp incl.)']
j = 0 # 0, 1, 2, 3, 4
j_tit = ['GCM', 'EQQ-8', 'EQQ-16', 'knn 16', 'knn 64']

plt.plot(tau, data[i, :, j])
plt.show()

# surs = 0
# su = tsurrs if surs == 0 else nsurrs

# # for j in range(5):
# #     plt.plot(tau, get_zscore(data[i, :, j], su[i, :, :, j]), label="%s" % j_tit[j])
# # # plt.plot(tau, get_zscore(data[i, :, j], tsurrs[i, :, :, j]), label="from TG")
# # plt.ylabel("z-score")
# # plt.axhline(2.6, 0, 1, linestyle="--", color='k', linewidth=1)
# # # plt.ylabel("p-value")
# # # plt.gca().invert_yaxis()
# # plt.xlabel(r"delay $\tau$")
# # plt.title("PRAGUE: %s surrs from %s" % (i_tit[i], 'TG' if surs==0 else 'NAO'))
# # plt.legend()
# # plt.xticks(np.arange(min(tau), max(tau)+1, 2))
# # plt.show()

# wh = 3
# plt.figure(figsize=(12,9))
# for j in range(5):
#     plt.subplot(2,3,j+1)
#     plt.hist(su[i, :, wh, j], bins=20)
#     plt.axvline(data[i, wh, j], 0, 1, color='k', linewidth=4)
#     plt.title(j_tit[j])
# plt.suptitle("PRAGUE: %s -- delay %d" % (i_tit[i], tau[wh]))
# plt.savefig("nao_temp_caus/PRG_delay%d.png" % (tau[wh]), bbox_inches='tight')