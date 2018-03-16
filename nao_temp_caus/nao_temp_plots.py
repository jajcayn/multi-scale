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

with open("nao_temp_caus/PRG_NAO-temp_caus_100FT-cc.bin", "rb") as f:
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

# plt.plot(tau, data[i, :, j])
# plt.show()

surs = 0
su = tsurrs if surs == 0 else nsurrs

for j in range(1):
    # plt.plot(tau, get_zscore(data[i, :, j], su[i, :, :, j]), label="%s" % j_tit[j])
    plt.plot(tau, data[i, :, j], label="%s" % j_tit[j], color='k')
    me = np.mean(su[i, :, :, j], axis=0)
    sd = np.std(su[i, :, :, j], axis=0, ddof=1)
    plt.plot(tau, me, label="%s mean surrs" % j_tit[j], color='#777777')
    plt.fill_between(tau, me+2.6*sd, me-2.6*sd, label="%s $\pm$ 2.6 SD surrs" % j_tit[j], facecolor="#999999", alpha=0.7)
# plt.plot(tau, get_zscore(data[i, :, j], tsurrs[i, :, :, j]), label="from TG")
# plt.ylabel("z-score")
plt.ylabel("CMI [nats]")
# plt.axhline(2.6, 0, 1, linestyle="--", color='k', linewidth=1)
# plt.ylabel("p-value")
# plt.gca().invert_yaxis()
plt.xlabel(r"delay $\tau$")
plt.title("PRAGUE: %s surrs from %s" % (i_tit[i], 'TG' if surs==0 else 'NAO'))
plt.legend()
plt.grid()
plt.xticks(np.arange(min(tau), max(tau)+1, 2))
plt.show()

# wh = 3
# plt.figure(figsize=(12,9))
# for j in range(5):
#     plt.subplot(2,3,j+1)
#     plt.hist(su[i, :, wh, j], bins=20)
#     plt.axvline(data[i, wh, j], 0, 1, color='k', linewidth=4)
#     plt.title(j_tit[j])
# plt.suptitle("PRAGUE: %s -- delay %d" % (i_tit[i], tau[wh]))
# plt.savefig("nao_temp_caus/PRG_delay%d.png" % (tau[wh]), bbox_inches='tight')