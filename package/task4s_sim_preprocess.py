####
import numpy as np
import os
import multiprocessing as mul
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads, vectorize, guvectorize, cuda
from tqdm import tqdm

os.system("cp -r ./o_data/icecube_10year_ps/events/ ./data/icecube_10year_ps/events/")

from core.signal_bag import *
from core.stacking_analysis import *
from core.req_arrays import *
print(icwidths)
import pandas as pd

import argparse as ap

parser = ap.ArgumentParser()

parser.add_argument('-nb', '--numbins', type=int, help='Number of bins', required=False)
parser.add_argument('-np', '--numpulsars', type=int, help='Number of pulsars for neutrinos to be generated at', required=False)
arrg = parser.parse_args()


season = 0
if arrg.numbins:
    nbins = arrg.numbins
else:
    nbins = 100

n_psrs = 50
if arrg.numpulsars:
    n_psrs = arrg.numpulsars

enus = np.logspace(11.001, 18.999, int(nbins))
enus_bin_indices = np.zeros(len(enus), dtype=np.int64)

for i in prange(len(enus)):
    enus_bin_indices[i] = np.digitize(enus[i], e_nu_wall) - 1
msdec_bin_indices = np.zeros(p, dtype=np.int64)
for i in prange(p):
    msdec_bin_indices[i] = np.digitize(msdec[i], dec_nu) - 1

for i in prange(len(enus)):
    enus_bin_indices[i] = np.digitize(enus[i], e_nu_wall) - 1
gamma_arr = [-2, -2.2, -2.53, -3]
# phio = np.logspace(-38, -26, 1000) #CHANGING TO LINEAR BINS RESULTS IN STRAIGHT LINES

####
syn_nu_choice = np.random.randint(0, p, n_psrs) #Choose 50 random pulsars from the 3389 pulsars
syn_nudec_bin = msdec_bin_indices[syn_nu_choice] #Find the declination bin of the chosen pulsars to be allocated for the synthetic neutrinos
syn_nu_ra = msra[syn_nu_choice] #Find the right ascension of the chosen pulsars to be allocated for the synthetic neutrinos
syn_nu_dec = msdec[syn_nu_choice] #Find the declination of the chosen pulsars to be allocated for the synthetic neutrinos
phio_const = 4.98 * (10**(-27)) #GeV-1 to ev-1 conversion factor
# phio_const *= 1e-5
filenames = ["IC40_exp.csv", "IC59_exp.csv","IC79_exp.csv", "IC86_I_exp.csv", "IC86_II_exp.csv",
        "IC86_III_exp.csv", "IC86_IV_exp.csv", "IC86_V_exp.csv", "IC86_VI_exp.csv", "IC86_VII_exp.csv"]

# syn_nu_dec = []
syn_N_nu = 0   # No.of neutrinos generated per season per energy bin per declination bin
o_lengths = []
for season_i in range(10):
    icdata_k = pd.read_csv("./o_data/icecube_10year_ps/events/" + filenames[season_i], sep="\s+", comment="#", names="MJD[days]	log10(E/GeV)	AngErr[deg]	RA[deg]	Dec[deg]	Azimuth[deg]	Zenith[deg]".split("\t"), dtype=float)
    o_lengths.append(len(icdata_k.index))
    min_ang_err_k = min(icdata_k['AngErr[deg]'])
    max_ang_err_k = max(icdata_k['AngErr[deg]'])
    syn_N_nu_sing_season = 0
    c= 0
    for i in range(len(enus)):    
    
        # apt = 0
        for j in range(len(syn_nudec_bin)):
            n_nu_temp = t_upt[season_i] * earea[ea_season(season_i)][syn_nudec_bin[j] * 40 + enus_bin_indices[i]] * enus[i] * phio_const * ((enus[i] / 10**14) ** gamma_arr[2])
            #n_nu_temp neutrinos are generated in this season, in this energy bin, in this declination bin
            n_nu_i = 0
            for n_nu_i in range(int(np.round(n_nu_temp, 0))):
                
                
                
                tempp = pd.DataFrame([-1, -1, np.random.uniform(min_ang_err_k, max_ang_err_k), msra[syn_nu_choice[j]], msdec[syn_nu_choice[j]], -1, -1]).T
                tempp.columns = "MJD[days]	log10(E/GeV)	AngErr[deg]	RA[deg]	Dec[deg]	Azimuth[deg]	Zenith[deg]".split("\t")
                icdata_k = pd.concat([icdata_k, tempp], ignore_index=True )
                c+=1
            # apt+=(n_nu_temp)
            syn_N_nu_sing_season+=n_nu_i
            
            
            
            
        # syn_N_nu_sing_season+=apt
    icdata_k.to_csv("./data/icecube_10year_ps/events/" + filenames[season_i], sep="\t", index=False)
    print(o_lengths[season_i], len(icdata_k.index), c)
    # print(len(icdata_k.index), icwidths[season_i + 1], c, len(icdata_k.index) - icwidths[season_i + 1])    
    syn_N_nu += syn_N_nu_sing_season


