import os
os.system("cp -r ./o_data/icecube_10year_ps/events/ ./data/icecube_10year_ps/")
import numpy as np
import multiprocessing as mul
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads, vectorize
from tqdm import tqdm
import pickle
import scipy.stats as st
import scipy.interpolate as interp
import argparse as ap
import time


################################################################################################################################
parser = ap.ArgumentParser()
parser.add_argument('-c', '--cone', type=float, help='Cone angle in degrees', required=False)
parser.add_argument('-nc', '--numthreads', type=int, help='Number of threads', required=False)
parser.add_argument('-nb', '--numbins', type=int, help='Number of bins', required=False)
parser.add_argument('-np', '--numpulsars', type=int, help='Number of pulsars for neutrinos to be generated at', required=False)
parser.add_argument('-lp', '--logphiomul', type=float, help='log10(Phi_0 multiplication factor)', required=False)

arrg = parser.parse_args()
num_threads = int(mul.cpu_count())
if arrg.numthreads:
    num_threads = arrg.numthreads
else:
    pass


set_num_threads(num_threads)



cone_deg = 5.0
if arrg.cone:
    cone_deg = arrg.cone
else:
    pass
    
cone = np.deg2rad(cone_deg)
cut = cone

print('#'*50)
print("Cone angle: ", cone_deg)


nbins = 1e2
if arrg.numbins:
    nbins = arrg.numbins
else:
    pass

n_psrs = 30
if arrg.numpulsars:
    n_psrs = arrg.numpulsars

phi_mul = 0
if arrg.logphiomul:
    phi_mul = arrg.logphiomul
else:
    pass

################################################################################################################################

print('#'*50)
print(f'Generating synthetic neutrinos for {n_psrs} pulsars and {nbins} energy bins')

# os.system('python3 task4s_sim_preprocess.py -nb ' + str(int(nbins)) + ' -np ' + str(n_psrs))
os.system('python3 task4w_syn_nu_smear.py -nb ' + str(int(nbins)) + ' -np ' + str(n_psrs) + ' -lp ' + str(phi_mul))

print('\nGenerated synthetic neutrinos')
print('#'*50)
time.sleep(5)
################################################################################################################################

from core.signal_bag import *
from core.stacking_analysis import *
from core.req_arrays import *
print(icwidths)


enus = np.logspace(11.001, 18.999, int(nbins))
enus_bin_indices = np.zeros(len(enus), dtype=np.int64)

for i in prange(len(enus)):
    enus_bin_indices[i] = np.digitize(enus[i], e_nu_wall) - 1
msdec_bin_indices = np.zeros(p, dtype=np.int64)
for i in prange(p):
    msdec_bin_indices[i] = np.digitize(msdec[i], dec_nu) - 1

for i in prange(len(enus)):
    enus_bin_indices[i] = np.digitize(enus[i], e_nu_wall) - 1
gamma_arr = [-2.2, -2, -2.53, -3]
phio = np.logspace(-48, -26, 1000) #CHANGING TO LINEAR BINS RESULTS IN STRAIGHT LINES

# print("\nNumber of threads: ", num_threads)
print("\nNumber of energy bins: ", len(enus))
print("\nNumber of phi bins: ", len(phio))
print("\nCalculating weights...\n\n")
eareaa = [i.astype(np.float64) for i in earea]
eareaa = np.asfarray(eareaa, dtype=np.float64)

earea2 = np.asfortranarray(earea)
earea2 = earea2[0].astype(np.float64)


altier_path = [os.getcwd() + '/pickle/', os.getcwd() + '/../pickle/']
@vectorize(['float64(int64, float64, int64)'], nopython=True, target='parallel')
def psr_wt_sing_gamma(psrno,gamma, season):

    tt_upt = t_upt[season]
    l = msdec_bin_indices[psrno]
    wt_ac_temp = np.zeros(len(enus), dtype=np.float64)
    for i in prange(len(enus)):
        wt_ac_temp[i] = np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * (enus[i]**gamma))


    return np.trapz(wt_ac_temp, enus)


print("Calculating wt_acc for all pulsars and seasons and gamma")
wt_acc = []
for gamma in prange(len(gamma_arr)):
    wt_allpsr = []
    for season in tqdm(prange(10)):


        wt_allpsr.append(np.array(psr_wt_sing_gamma(prange(p), gamma_arr[gamma], season), dtype=np.float64))
        # tmp = []
    wt_acc.append(wt_allpsr)
    wt_allpsr = []
    
wt_acc = np.asfarray(wt_acc, dtype=np.float64)


print("Calculated wt_acc for all pulsars and seasons and gamma")
# season_walls = np.asarray([0, 36900, 143911, 237044, 373288, 486146, 608687, 735732, 865043, 988700, 1134450])
season_walls = [0]
season_walls.extend(icwidths)
season_walls = np.asarray(season_walls)
season_widts= np.diff(season_walls)


#Compute the signal PDF for all neutrinos as per eqns 6, 7 and weights as per eqn 8 of 2205.15963

w_models = np.array([np.ones(p)])

@njit(nogil=True, parallel=True)
def S_ijk(nu): 

    '''
    Calculates S_ij as in EQN 7 of 2205.15963
    ----------

    Parameters
    ----------
    nu : int
        Index of the neutrino in the sample
        
    
    Returns
    -------
        Returns the signal PDF for the {psrno}th pulsar and nuind_inp neutrino
    '''
    
    
    
   
    
    #SUGGESTION 0: SIGNAL PDF USES A CUT = CONE OF BGND PDF
    ang = hvovec(msra, msdec, icra[nu], icdec[nu], rad=True)
    
    sg = np.deg2rad(icang[nu]) ** 2                                     #rad**2
    psr_angs =  np.divide(np.exp(-1 * np.divide(ang ** 2, 2*sg)), (2 * np.pi * sg))      #1/rad**2
    for i in prange(p):
        if ang[i] > cone:
            psr_angs[i] = 0
            
    return psr_angs
    


@njit(nogil=True)
def S_ik(nu, weight, w_models, gamma_index, ws):

    '''
    
    Calculates S_i as in EQN 8 of 2205.15963
    ----------

    Parameters
    ----------
    nu : int
        Index of the neutrino in the sample

    normalized_wt : array
        Normalized weights of the pulsars


    gamma_index : int
        Index of the gamma value in the gamma array

    ws : int
        Index of the weight model

    Returns
    -------
        Returns the signal PDF for the {psrno}th pulsar and nuind_inp neutrino

    '''

    sij = S_ijk(nu)
    season = 0
    for i in range(10):
        if season_walls[i] <= nu and nu < season_walls[i+1]:
            season = i
            break

    return np.sum(np.multiply(sij, np.multiply(w_models[ws], weight[gamma_index][season])/np.sum(np.multiply(w_models[ws], weight[gamma_index][season]))))      #1/rad**2

@njit(parallel=True, nogil=True)
def Sik_sing_s_g(gamma_index, ws):#, wt_acc=wt_acc, w_models=w_models):
    '''
    Calculates S_i as in EQN 8 of 2205.15963
    ----------

    Parameters
    ----------
    weight : array
         weights of the pulsars

    season : int
        Season of the neutrino

    gamma_index : int
        Index of the gamma value in the gamma array

    ws : int
        Index of the weight model

    Returns
    -------
        Returns the signal PDF for the {psrno}th pulsar and nuind_inp neutrino
    '''



    tmp = []
    if ws == -1: #No weights
        for nu in prange(len(icra)):
            tmp.append(np.sum(S_ijk(nu)))
        return np.array(tmp, dtype=np.float64)

    for nu in prange(len(icra)):
        tmp.append(S_ik(nu, wt_acc, w_models, gamma_index, ws))
    return np.array(tmp, dtype=np.float64)


@vectorize(['float64(int64, int64)'], nopython=True,target='parallel')
def Bi_stacked_compute(nu, cone=cone):

    '''
    Calculates B_i as in EQN 9 of 2205.15963
    ----------

    Parameters
    ----------
    nu : int
        Index of the neutrino from IceCube sample
    cone : float
        Cone angle in degrees.
    

    Returns
    -------
    float
        Returns the background PDF for the {nu}th neutrino
    '''

    # count = np.sum(np.abs(np.subtract(icdec, icdec[nu])) <= cone)
    count=0
    for i in prange(len(icdec)):
        if abs(icdec[i] - icdec[nu]) <= cone:
            count+=1
    binwidth = (np.sin(np.deg2rad(icdec[nu] + cone)) - np.sin(np.deg2rad(icdec[nu] - cone)))*2*np.pi
    return count/(binwidth * lnu)           #No units or sr**-1


print("\nCalculating Bi for all neutrinos\n")
all_Bi = Bi_stacked_compute(np.arange(lnu), int(cone_deg))


print("\nCalculated Bi for all neutrinos")

        

@vectorize(['float64(int64, float64, float64, int64)'], nopython=True, target='parallel')
def ns_singleseason_sing_psr_HAT(psrno,gamma, phi0, season):
   

    tt_upt = t_upt[season]

        
    l = msdec_bin_indices[psrno]
     
        
    ns_temp = np.zeros(len(enus), dtype=np.float64)
    for i in prange(len(enus)):
        ns_temp[i] += np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * phi0 * (enus[i]/(10**14))**gamma)
    # temp_ea = np.asarray(earea[ea_season(season)])[l*40 + k]
    # return tt_upt * temp_ea * phi0 * ((enu/(10**14))**gamma)     #in s cm2 eV

    return np.trapz(ns_temp, enus) 


wt_acc.shape



@jit(nopython=True)
def Pr(x, Ns, S, B):
    nsN = x/Ns
    return np.add(np.multiply(nsN , S), np.multiply(np.subtract(1, nsN), B))



# @njit(nogil=True)
def TS_st_vec(x, S, B, Ns):
    nsN = x/Ns
    pr = np.add(np.multiply(nsN , S), np.multiply(np.subtract(1, nsN), B))
    return np.sum(np.asfarray(2*np.log(pr/B)))

lnu = len(icra)
Ns = lnu


print("\nCalculating S_i for all neutrinos and gammas and weighting schemes...\n")

all_Si_ws_g_s = []

tmp_wt_acc = []



for gamma_index in tqdm(prange(4)):
    
    
    tmp_wt_acc.append(Sik_sing_s_g(gamma_index, 0))
    
    



all_Si_ws_g_s.append([tmp_wt_acc])

tmp_wt_acc = []

all_Si_ws_g_s = np.asfarray(all_Si_ws_g_s[0])

print("Calculated S_i for all neutrinos and gammas and weighting schemes")




def ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma, e_nus=enus, phi0=1):
    ns_hat = 0
    ns_hat_wt = 0

    for season in tqdm(prange(10)):

        ns_hat = ns_singleseason_sing_psr_HAT(prange(p), gamma, phi0, season)
        ns_hat_wt += ns_hat

    return np.array([np.sum(ns_hat_wt)], dtype=np.float64)


arr = []


print("\nCalculating ns_HAT for all gamma and weighting schemes...\n")

arr=[]
for gamma in prange(len(gamma_arr)):
    tmp = ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma_arr[gamma])
    np.savetxt('outputs/ns_hat_wt_wt_gamma_{}.txt'.format(gamma_arr[gamma]), tmp)
    arr.append(tmp)
    tmp = []

arr = np.array(arr, dtype=np.float64)


print("\nCalculationed ns_HAT for all gamma and weighting schemes")



print('\nCALCULATING TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS\n')
all_TSS_wmod1 = []
for gamma in prange(len(gamma_arr)):
    print("gamma = {}".format(gamma))
    
    
    
    t2mp = np.asfarray(all_Si_ws_g_s[0][gamma])#.reshape(len(all_Si_ws_g_s[0][gamma]), 1)
    
    temp = []
    for phi in tqdm(prange(len(phio))):
        # temp.append(TS_for_all_psrs2(arr[gamma]*phio[phi]))
        temp.append(TS_st_vec(arr[gamma]*phio[phi], t2mp, all_Bi, Ns))
    all_TSS_wmod1.append(temp)
    temp = []

print('\nCALCULATED TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS')



all_TSS_wmod1 = np.array(all_TSS_wmod1, dtype=np.float64)



all_TSS_wmod1 = np.asarray(all_TSS_wmod1)
gamma_arr = np.asarray(gamma_arr)

pickle.dump(all_TSS_wmod1, open(f'./pickle/all_TSS_wmod1_{nbins}_bins_{n_psrs}_psrs_{phi_mul}_phi_fac_smeared.pkl', 'wb'))

all_e_UL = []
e_decade = [1e13, 1e14, 1e15, 1e16, 1e17]
for e_UL in e_decade:
    e2dfde = []

    for gamma in prange(len(gamma_arr)):
        temp = []
        for phi in range(len(phio)):
            temp.append( e_UL**2 * dfde(e_UL, gamma_arr[gamma], phio[phi]))        #in eV
        e2dfde.append(temp)
    e2dfde = np.asarray(e2dfde)

    all_e_UL.append(e2dfde)
mark = ['^', 'o', 's', 'd']


for g in range(3):
    print(phio[np.nanargmax(all_TSS_wmod1[g])]) 
    print(all_TSS_wmod1[g][np.nanargmax(all_TSS_wmod1[g])])

f = open(f'outputs/TS_wmod1_{nbins}_bins_{n_psrs}_psrs_{phi_mul}_phi_fac.txt', 'w')
f.write(str(all_TSS_wmod1[2][np.nanargmax(all_TSS_wmod1[2])]) + '\n')
f.write(str(100 * (np.sum(icwidths) - 1134431) / 1134431) + '\n')
f.write(str(phio[np.nanargmax(all_TSS_wmod1[2])]) + '\n')
f.close()

e2dfde = np.asarray([1e28 * dfde(1e14, g, 1) * phio for g in gamma_arr])
# plt.style.use('default')
font = {'family': 'serif',
        'weight': 'bold',
        'size': 22,
        'color':  'black',
        }
smallerfont = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 15,
        }

axesfont = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 17,
        }

fig, axs = plt.subplots(1,1, figsize=(7, 6))

for gamma in [ 1, 2, 3]:#range(4):
    
    axs.plot(1e28 * dfde(1e14, gamma_arr[gamma], 1) *phio /1e9, all_TSS_wmod1[gamma], label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2)# + ' with wt')    #in GeV
    # axs.vlines(1e19*phio[np.nanargmax(all_TSS_wmod1[gamma])], -1e7, 90,lw=2.2, label='95 % UPPER LIMIT $TS = -3.84$')
    # print(1e19*phio[np.nanargmax(all_TSS_wmod1[gamma])])
    

axs.set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = 1}}$', fontdict=smallerfont)


# for i in range(3):
    
axs.legend(prop={'size':14}, framealpha=0, loc='upper left')
# axs.hlines(-3.84, 1e-20, 1e-5, linestyles='dashed', lw=2.2, ls='-.', label='95 % UPPER LIMIT $TS = -3.84$', color='lightcoral')
axs.set_xscale('log')
axs.set_xlabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ at 100 TeV ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$)', fontdict=axesfont)
axs.set_ylabel('$\mathbf{TS_{max}}$', fontdict=axesfont)
axs.xaxis.set_tick_params(labelsize=15)
axs.yaxis.set_tick_params(labelsize=15)

# axs.set_ylim(-20, np.ravel(all_TSS_wmod1)[np.nanargmax[np.ravel(all_TSS_wmod1)]])
tempt = [all_TSS_wmod1[i][np.nanargmax(all_TSS_wmod1[i])] for i in range(len(gamma_arr))]
axs.set_ylim(-20, max(tempt))
axs.set_xlim(0.95e-19, 1e-6)
axs.vlines(x = 4.98e-9, ymin=-21, ymax=max(tempt) + 10, label='$\phi_0 = 4.98 x 10^{-19}$',lw=2.2, ls='--', color='black')


# plt.suptitle('TS vs Total Neutrino Flux', fontweight='bold', fontsize=20, fontfamily='serif')

plt.tight_layout()
plt.savefig(f'outputs/TS_vs_E2dfde_w_model=1_bins={len(enus)}_{n_psrs}_psrs_{phi_mul}_phi_fac_smeared.pdf')
# plt.show()
print(f'\nTS_vs_E2dfde_w_model=1_bins={len(enus)}_{n_psrs}_psrs_{phi_mul}_phi_fac_smeared.pdf\nDONE')