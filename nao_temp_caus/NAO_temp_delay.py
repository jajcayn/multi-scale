import pyclits as clt
from pyclits.surrogates import SurrogateField
import pyclits.mutual_inf as MI
import numpy as np
import cPickle
from datetime import date
from multiprocessing import Queue, Process

cities = ['PRG', 'FRANKFURT', 'WARSAW']

for location in cities:

    # load NAO
    nao_raw = np.loadtxt("NAO.daily.1950-2017.txt")
    nao = clt.geofield.DataField()
    nao.data = nao_raw[:, 3].copy()
    nao.create_time_array(date_from=date(1950,1,1), sampling='d')
    nao.select_date(date(1950,1,1), date(2017,1,1))

    nao_sg = SurrogateField()
    nao_sg.copy_field(nao)

    # load station-like TG and PP time series
    prg_raw = np.loadtxt("%s_from_ECAD_tg_pp.txt" % (location))
    prg_tg = clt.geofield.DataField()
    prg_tg.data = prg_raw[:, 3].copy()
    prg_tg.create_time_array(date_from=date(1950,1,1), sampling='d')
    prg_tg.select_date(date(1950,1,1), date(2017,1,1))

    prg_sg = SurrogateField()
    mean, var, trend = prg_tg.get_seasonality(True)
    prg_sg.copy_field(prg_tg)
    prg_tg.return_seasonality(mean, var, trend)

    prg_pp = clt.geofield.DataField()
    prg_pp.data = prg_raw[:, 4].copy()
    prg_pp.create_time_array(date_from=date(1950,1,1), sampling='d')
    prg_pp.select_date(date(1950,1,1), date(2017,1,1))

    def _process_NAOsurrs(jobq, resq, nao_sg, tg, pp, tau):
        while jobq.get() is not None:
            nao_sg.construct_fourier_surrogates(algorithm='FT')

            caus_surrs_temp = np.zeros((2,5))
            # no pressure = 3dim cond.
            x, y, z = MI.get_time_series_condition([nao_sg.data, tg], tau=tau, dim_of_condition=3, eta=3)
            # MIGAU
            caus_surrs_temp[0, 0] = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
            # MIEQQ8
            caus_surrs_temp[0, 1] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=8, log2=False)
            # MIEQQ16
            caus_surrs_temp[0, 2] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=16, log2=False)
            # MIKNN16
            caus_surrs_temp[0, 3] = MI.knn_cond_mutual_information(x, y, z, k=16, standardize=True, dualtree=True)
            # MIKNN64
            caus_surrs_temp[0, 4] = MI.knn_cond_mutual_information(x, y, z, k=64, standardize=True, dualtree=True)
            # with pressure = 4dim cond.
            x, y, z = MI.get_time_series_condition([nao_sg.data, tg], tau=tau, dim_of_condition=3, eta=3, 
                add_cond = pp)
            # MIGAU
            caus_surrs_temp[1, 0] = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
            # MIEQQ8
            caus_surrs_temp[1, 1] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=8, log2=False)
            # MIEQQ16
            caus_surrs_temp[1, 2] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=16, log2=False)
            # MIKNN16
            caus_surrs_temp[1, 3] = MI.knn_cond_mutual_information(x, y, z, k=16, standardize=True, dualtree=True)
            # MIKNN64
            caus_surrs_temp[1, 4] = MI.knn_cond_mutual_information(x, y, z, k=64, standardize=True, dualtree=True)

            resq.put(caus_surrs_temp)


    def _process_TGsurrs(jobq, resq, nao, sg, pp, tau, mean, var, trend):
        while jobq.get() is not None:
            sg.construct_fourier_surrogates(algorithm='FT')
            sg.add_seasonality(mean, var, trend)

            caus_surrs_temp = np.zeros((2,5))
            # no pressure = 3dim cond.
            x, y, z = MI.get_time_series_condition([nao, sg.data], tau=tau, dim_of_condition=3, eta=3)
            # MIGAU
            caus_surrs_temp[0, 0] = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
            # MIEQQ8
            caus_surrs_temp[0, 1] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=8, log2=False)
            # MIEQQ16
            caus_surrs_temp[0, 2] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=16, log2=False)
            # MIKNN16
            caus_surrs_temp[0, 3] = MI.knn_cond_mutual_information(x, y, z, k=16, standardize=True, dualtree=True)
            # MIKNN64
            caus_surrs_temp[0, 4] = MI.knn_cond_mutual_information(x, y, z, k=64, standardize=True, dualtree=True)
            # with pressure = 4dim cond.
            x, y, z = MI.get_time_series_condition([nao, sg.data], tau=tau, dim_of_condition=3, eta=3, 
                add_cond = pp)
            # MIGAU
            caus_surrs_temp[1, 0] = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
            # MIEQQ8
            caus_surrs_temp[1, 1] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=8, log2=False)
            # MIEQQ16
            caus_surrs_temp[1, 2] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=16, log2=False)
            # MIKNN16
            caus_surrs_temp[1, 3] = MI.knn_cond_mutual_information(x, y, z, k=16, standardize=True, dualtree=True)
            # MIKNN64
            caus_surrs_temp[1, 4] = MI.knn_cond_mutual_information(x, y, z, k=64, standardize=True, dualtree=True)

            resq.put(caus_surrs_temp)


    # compute causality
    NUM_SURRS = 100
    WORKERS = 20
    TAUS = np.arange(1,41,1)
    data_caus = np.zeros((2, TAUS.shape[0], 5)) # pp yes or no x delays x (MIGAU, MIEQQ8, MIEQQ16, knn16, knn64)
    surrs_NAOcaus = np.zeros((2, NUM_SURRS, TAUS.shape[0], 5))
    surrs_TGcaus = np.zeros((2, NUM_SURRS, TAUS.shape[0], 5))

    for tau, ii in zip(TAUS, range(TAUS.shape[0])):
        print("computing for lag %d..." % (tau))
        # no pressure = 3dim cond.
        x, y, z = MI.get_time_series_condition([nao.data, prg_tg.data], tau=tau, dim_of_condition=3, eta=3)
        # MIGAU
        data_caus[0, ii, 0] = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
        # MIEQQ8
        data_caus[0, ii, 1] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=8, log2=False)
        # MIEQQ16
        data_caus[0, ii, 2] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=16, log2=False)
        # MIKNN16
        data_caus[0, ii, 3] = MI.knn_cond_mutual_information(x, y, z, k=16, standardize=True, dualtree=True)
        # MIKNN64
        data_caus[0, ii, 4] = MI.knn_cond_mutual_information(x, y, z, k=64, standardize=True, dualtree=True)

        # with pressure = 4dim cond.
        x, y, z = MI.get_time_series_condition([nao.data, prg_tg.data], tau=tau, dim_of_condition=3, eta=3, 
            add_cond = prg_pp.data)
        # MIGAU
        data_caus[1, ii, 0] = MI.cond_mutual_information(x, y, z, algorithm='GCM', log2=False)
        # MIEQQ8
        data_caus[1, ii, 1] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=8, log2=False)
        # MIEQQ16
        data_caus[1, ii, 2] = MI.cond_mutual_information(x, y, z, algorithm='EQQ2', bins=16, log2=False)
        # MIKNN16
        data_caus[1, ii, 3] = MI.knn_cond_mutual_information(x, y, z, k=16, standardize=True, dualtree=True)
        # MIKNN64
        data_caus[1, ii, 4] = MI.knn_cond_mutual_information(x, y, z, k=64, standardize=True, dualtree=True)
        
        print("  data done. starting surrogates from NAO...")
        # surrs from NAO
        jobq = Queue()
        resq = Queue()
        for _ in range(NUM_SURRS):
            jobq.put(1)
        for _ in range(WORKERS):
            jobq.put(None)
        surrs_completed = 0
        workers = [Process(target=_process_NAOsurrs, args=(jobq, resq, nao_sg, prg_tg.data, prg_pp.data, tau))]
        for w in workers:
            w.start()

        while surrs_completed < NUM_SURRS:
            temp_caus = resq.get()
            surrs_NAOcaus[:, surrs_completed, ii, :] = temp_caus
            surrs_completed += 1
            if surrs_completed%20 == 0:
                print("  ...%d/%d NAO surrs done..." % (surrs_completed, NUM_SURRS))

        for w in workers:
            w.join()
        print("  NAO surrs done. starting surrogates from TG...")
        # surrs from TG
        jobq = Queue()
        resq = Queue()
        for _ in range(NUM_SURRS):
            jobq.put(1)
        for _ in range(WORKERS):
            jobq.put(None)
        surrs_completed = 0
        workers = [Process(target=_process_TGsurrs, args=(jobq, resq, nao.data, prg_sg, prg_pp.data, tau, mean, var, trend))]
        for w in workers:
            w.start()

        while surrs_completed < NUM_SURRS:
            temp_caus = resq.get()
            surrs_TGcaus[:, surrs_completed, ii, :] = temp_caus
            surrs_completed += 1
            if surrs_completed%20 == 0:
                print("  ...%d/%d TG surrs done..." % (surrs_completed, NUM_SURRS))

        for w in workers:
            w.join()
        print("  done.")

    print("all done. saving...")
    with open("%s_NAO-temp_caus_100FT.bin" % (location), "wb") as f:
        cPickle.dump({'data' : data_caus, 'NAOsurrs' : surrs_NAOcaus, 
            'TGsurrs' : surrs_TGcaus, 'taus' : TAUS}, f, protocol=cPickle.HIGHEST_PROTOCOL)