import pyclits as clt
from pyclits.surrogates import SurrogateField
import pyclits.mutual_inf as MI
import numpy as np
import cPickle
from datetime import date
from multiprocessing import Queue, Process
from pathos.multiprocessing import Pool
# from time import sleep

def _process_CMI_data(jobq, resq):
    while True:
        a = jobq.get()
        if a is None:
            break
        else:
            tau, la, lo = a
            if not np.any(np.isnan(tg.data[:, la, lo])):
                # 3d
                x, y, z = MI.get_time_series_condition([nao.data, tg.data[:, la, lo]], tau=tau, dim_of_condition=3, eta=3,
                    close_condition=True)
                cmi_3d =  MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
                
                # with pressure = 4dim cond.
                if not np.any(np.isnan(pp.data[:, la, lo])):
                    x, y, z = MI.get_time_series_condition([nao.data, tg.data[:, la, lo]], tau=tau, dim_of_condition=3, eta=3, 
                        add_cond=pp.data[:, la, lo], close_condition=True)
                    cmi_4d = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
                else:
                    cmi_4d = np.nan

                resq.put([tau, la, lo, cmi_3d, cmi_4d])
            else:
                resq.put([tau, la, lo, np.nan, np.nan])


def _process_CMI_surrs(jobq, resq):
    while True:
        a = jobq.get()
        if a is None:
            break
        else:
            tau, la, lo = a
            if not np.any(np.isnan(sg.data[:, la, lo])):
                # 3d
                x, y, z = MI.get_time_series_condition([nao.data, sg.data[:, la, lo]], tau=tau, dim_of_condition=3, eta=3,
                    close_condition=True)
                cmi_3d =  MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
                
                # with pressure = 4dim cond.
                if not np.any(np.isnan(pp.data[:, la, lo])):
                    x, y, z = MI.get_time_series_condition([nao.data, sg.data[:, la, lo]], tau=tau, dim_of_condition=3, eta=3, 
                        add_cond=pp.data[:, la, lo], close_condition=True)
                    cmi_4d = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
                else:
                    cmi_4d = np.nan

                resq.put([tau, la, lo, cmi_3d, cmi_4d])
            else:
                resq.put([tau, la, lo, np.nan, np.nan])



# load NAO
# path = "/Users/nikola/work-ui/data/"
path = "/home/nikola/Work/phd/thesis/"
print("Loading NAO...")
nao_raw = np.loadtxt("NAO.daily.1950-2017.txt")
nao = clt.geofield.DataField()
nao.data = nao_raw[:, 3].copy()
nao.create_time_array(date_from=date(1950,1,1), sampling='d')
nao.select_date(date(1950,1,1), date(2017,1,1))

print("Loading ECAD temperatures...")
tg = clt.data_loaders.load_ECA_D_data_daily(path + "ECAD.tg.daily.0.50deg.nc", 'tg', date(1950,1,1), 
            date(2017, 1, 1), lats=[35, 70], lons=[-12.5, 60], anom=False)
print("Loading ECAD SLPs...")
pp = clt.data_loaders.load_ECA_D_data_daily(path + "ECAD.pp.daily.0.50deg.nc", 'pp', date(1950,1,1), 
            date(2017, 1, 1), lats=[35, 70], lons=[-12.5, 60], anom=False)

assert tg.shape() == pp.shape()

print("Creating surrogate fields...")
# surrogates
mean, var, _ = tg.get_seasonality(detrend=True)
sg = SurrogateField()
sg.copy_field(tg)
tg.return_seasonality(mean, var, None)

NUM_SURRS = 100
WORKERS = 20
TAUS = np.arange(1,41,1)
data_caus = np.zeros([2, TAUS.shape[0]] + tg.get_spatial_dims()) # pp yes or no x delays
surrs_caus = np.zeros([NUM_SURRS, 2, TAUS.shape[0]] + tg.get_spatial_dims())

jobq = Queue()
resq = Queue()
to_compute = 0
THEORY_to_compute = TAUS.shape[0] * tg.lats.shape[0] * tg.lons.shape[0]
print("starting computation on data using %d workers" % (WORKERS))
workers = [Process(target=_process_CMI_data, args=(jobq, resq)) for _ in range(WORKERS)]
for w in workers:
    w.start()

for tau in TAUS:
    for la in range(tg.lats.shape[0]):
        for lo in range(tg.lons.shape[0]):
            jobq.put([tau, la, lo])
            to_compute += 1
    print("...filling up the queue - %d / %d done..." % (to_compute, THEORY_to_compute))        

for _ in range(WORKERS):
    jobq.put(None)

assert to_compute == THEORY_to_compute
cnt = 0
while cnt < to_compute:
    tau, la, lo, cmi1, cmi2 = resq.get()
    data_caus[:, tau-1, la, lo] = [cmi1, cmi2]
    cnt += 1
    if cnt%50000==0:
        print("...getting results - %d / %d done..." % (cnt, to_compute))

for w in workers:
    w.join()
print("Data done!")

print("Starting %d surrogates..." % (NUM_SURRS))

for ns in range(NUM_SURRS):
    pool = Pool(WORKERS)
    sg.construct_fourier_surrogates(algorithm='FT', pool=pool)
    pool.close()
    pool.join()
    sg.add_seasonality(mean, var, None)
    jobq = Queue()
    resq = Queue()
    to_compute = 0
    THEORY_to_compute = TAUS.shape[0] * tg.lats.shape[0] * tg.lons.shape[0]
    workers = [Process(target=_process_CMI_surrs, args=(jobq, resq)) for _ in range(WORKERS)]
    for w in workers:
        w.start()
    for tau in TAUS:
        for la in range(sg.lats.shape[0]):
            for lo in range(sg.lons.shape[0]):
                jobq.put([tau, la, lo])
                to_compute += 1
        print("%d/%d SURR  ...filling up the queue - %d / %d done..." % (ns, NUM_SURRS, to_compute, THEORY_to_compute)) 
    for _ in range(WORKERS):
        jobq.put(None)
    assert to_compute == THEORY_to_compute
    cnt = 0
    while cnt < to_compute:
        tau, la, lo, cmi1, cmi2 = resq.get()
        surrs_caus[ns, :, tau-1, la, lo] = [cmi1, cmi2]
        cnt += 1
        if cnt%500000==0:
            print("%d/%d SURR  ...getting results - %d / %d done..." % (ns, NUM_SURRS, cnt, to_compute))

    for w in workers:
        w.join()
print("surrogates done! Saving...")

with open("NAO_temp_delay_ECAD.bin", "wb") as f:
    cPickle.dump({'data' : data_caus, 'surrs' : surrs_caus, 'tau' : TAUS}, 
        f, protocol=cPickle.HIGHEST_PROTOCOL)