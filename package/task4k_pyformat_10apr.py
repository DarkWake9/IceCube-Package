from core import download_IC
from core import download_ATNF
from core import readfiles
import numpy as np
import pandas as pd
import os
import multiprocessing as mul
import matplotlib.pyplot as plt
import scipy.stats as sct
from numba import jit, njit, prange, set_num_threads, vectorize
from tqdm import tqdm
from core.signal_bag import *
from core.stacking_analysis import *
from core.req_arrays import *
set_num_threads(int(mul.cpu_count()*0.9))
all_enu = e_nu_wall


enus = np.logspace(11.001, 18.999, 2000)
np.diff(all_enu).shape
gamma_arr = [-1, -2.2, -2.5, -3]
phio = np.logspace(-38, -26, 1000) #CHANGING TO LINEAR BINS RESULTS IN STRAIGHT LINES
#Uncomment to compute the background PDF for all neutrinos as per eqn 9 of 2205.15963

#and store them at /data/all_Bi.dat

#then recomment it


# all_Bi = signals(4).compute_background()
# all_Bi.tofile(os.getcwd() + '/data/all_Bi.dat', sep=',')

all_Bi = np.loadtxt(os.getcwd() + '/data/all_Bi.dat', delimiter=',')
weight_obj =  [weights.weights(gamma).all_weights for gamma in gamma_arr]
sum_weights = [weights.weights(gamma).sum_weights for gamma in gamma_arr]

#Calculating weight/sum(weights) to avoid wrong ns = 0 entries 
normalized_wt = []
for gamma in range(4):
    temp = []
    for season in range(len(weight_obj[gamma])):
        temp.append(weight_obj[gamma][ea_season(season)]/sum_weights[gamma][ea_season(season)])
    normalized_wt.append(temp)
#Compute the signal PDF for all neutrinos as per eqns 6, 7 and weights as per eqn 8 of 2205.15963
all_Si = np.asfarray([signals(gamma_arr[i]).compute_signal() for i in tqdm(prange(4))])
#generates ns^ for a single pulsar a single season and a single energy


print("\nCAULCULATED SIGNAL PDFS FOR ALL GAMMAS AND BACKGROUND PDFS")

# @njit()
@vectorize(['float64(float64, float64, float64, float64, int64)'], target='parallel')
def ns_singleseason_sing_psr_HAT(dec,enu, gamma, phi0 = 1, season=0):
    '''
    This function returns the number of signal events for a single pulsar as in EQN3 of 2205.15963
    -------------------

    Parameters
    ----------
    dec : float
        The declination of the pulsar in radians
    
    enu : float
        The neutrino energy in eV

    gamma : float
        The spectral index of the neutrino flux

    phi0 : float (optional)
        The normalization constant of the neutrino flux

    season : int (optional)
        The IceCube season number
    
    Returns
    -------
    float
        The number of signal events for the given parameters
    '''


    tt_upt = t_upt[season]
    if enu <= 1e11 or enu >= 1e19:
        return 0.0
    else:
        k=0
        l=0
        for i in range(0, len(e_nu_wall)):
            if e_nu_wall[i] <= enu and e_nu_wall[i+1] > enu:
                k=i
                break

        for j in range(0, len(dec_nu)):
            if dec_nu[j] <= dec and dec_nu[j+1] > dec:
                l=j
                break

            
        temp_ea = np.asarray(earea[ea_season(season)])[l*40 + k]
        return tt_upt * temp_ea * phi0 * ((enu/(10**14))**gamma)     #in s cm2 eV


def ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma, e_nus=enus, phi0=1):
    ns_hat = 0
    ns_hat_wt = 0
    for season in tqdm(prange(10)):
        for psr in prange(p):
            wt = normalized_wt[gamma_arr.index(gamma)][ea_season(season)][psr]
            nsop = ns_singleseason_sing_psr_HAT(msdec[psr], e_nus, gamma, phi0, season)
            ns_hat_wt += wt * np.trapz(nsop, x=e_nus)
            ns_hat += np.trapz(nsop, x=e_nus)
    return np.array([ns_hat, ns_hat_wt], dtype=np.float64)

print('CALCULATING ns_HAT WITH AND WITHOUT WEIGHTS FOR ALL PSRS FOR ALL GAMMAS')

pool = mul.Pool(int(mul.cpu_count()*0.9))
op_async = pool.map_async(ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights, gamma_arr)
arr= op_async.get()
pool.close()
op_async = []
pool = []



print('\nCALCULATED ns_HAT WITH AND WITHOUT WEIGHTS FOR ALL PSRS FOR ALL GAMMAS')

print('CALCULATING TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS')
all_TSS = []
for wt in range(2):
    tmp = []
    for gamma in tqdm(range(4)):
        @njit
        def TS_for_all_psrs2(nsa):  
            return Ts_arr2(nsa, all_Si[gamma], all_Bi, Ns)      #No units

        pool = mul.Pool(int(mul.cpu_count()*0.9), maxtasksperchild=200)
        op_arr = pool.map_async(TS_for_all_psrs2, arr[gamma][wt]*phio) #No units
        temp = op_arr.get()
        pool = []
        op_arr = []
        tmp.append(temp)

    all_TSS.append(tmp)

print('\nCALCULATED TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS')


np.shape(all_TSS)
for w in range(2):
    for g in range(4):
        print(min(all_TSS[w][g]), max(all_TSS[w][g]))
    print('wt\n')


all_TSS = np.asarray(all_TSS)
gamma_arr = np.asarray(gamma_arr)
e2dfde = []

for gamma in prange(4):
    temp = []
    for phi in range(len(phio)):
        temp.append( 1e28 * dfde(1e14, gamma_arr[gamma], phio[phi]))        #in eV
    e2dfde.append(temp)

e2dfde = np.asarray(e2dfde)
mark = ['^', 'o', 's', 'd']
plt.figure(figsize=(12,8))
for gamma in [0, 1, 2, 3]:#range(4):
    plt.plot(e2dfde[gamma]/1e9, all_TSS[0][gamma], label='$\Gamma$ = ' + str(gamma_arr[gamma]))    #in GeV
    plt.plot(e2dfde[gamma]/1e9, all_TSS[1][gamma], ls='--',label='$\Gamma$ = ' + str(gamma_arr[gamma]) + ' with wt')    #in GeV

plt.legend()
plt.xscale('log')
plt.xlabel('$E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}$ (GeV)')
plt.ylabel('TS')
plt.title('TS vs $E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}$')
# plt.xlim(1e-30, 1e-22)
plt.ylim(-20, 5)
plt.savefig('TS_vs_e2dfde.png', facecolor='w', edgecolor='w')
# plt.show()
print('\nPLOT SAVED TO TS_vs_e2dfde.png\nDONE')