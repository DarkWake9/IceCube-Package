import numpy as np
import os
import multiprocessing as mul
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads, vectorize
from tqdm import tqdm
import pickle
import scipy.stats as st
import scipy.interpolate as interp
import argparse as ap


parser = ap.ArgumentParser()

parser.add_argument('-c', '--cone', type=float, help='Cone angle in degrees', required=False)
parser.add_argument('-nc', '--numthreads', type=int, help='Number of threads', required=False)
parser.add_argument('-nb', '--numbins', type=int, help='Number of bins', required=False)

arrg = parser.parse_args()
num_threads = int(mul.cpu_count())
if arrg.numthreads:
    num_threads = arrg.numthreads
else:
    pass


set_num_threads(num_threads)

from core.signal_bag import *
from core.stacking_analysis import *
from core.req_arrays import *

# UNCOMMENT FOR LINEAR BINS
# all_enu = np.linspace(10**11.001, 10**18.999, 1000)
all_enu = e_nu_wall

cone_deg = 5.0
if arrg.cone:
    cone_deg = arrg.cone
else:
    pass
    
cone = np.deg2rad(cone_deg)
cut = cone

print('#'*50)
print("Cone angle: ", cone_deg)
# enus = 0.5*(all_enu[1:]+all_enu[:-1])
# UNCOMMENT FOR DENSER LOGARITHMIC BINS, optimal nbins is 1e6
nbins = 1e7
if arrg.numbins:
    nbins = arrg.numbins
else:
    pass


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
phio = np.logspace(-38, -26, 1000)

# print("\nNumber of threads: ", num_threads)
print("\nNumber of energy bins: ", len(enus))
print("\nNumber of phi bins: ", len(phio))
print("\nCalculating weights...\n\n")
eareaa = [i.astype(np.float64) for i in earea]
eareaa = np.asfarray(eareaa, dtype=np.float64)
eareaa[0][0]
earea2 = np.asfortranarray(earea)
earea2 = earea2[0].astype(np.float64)


msra = np.array(mspdata['RAJD'], dtype=np.float64)
msdec = np.array(mspdata['DECJD'], dtype=np.float64)
p = len(mspdata)

w_models = np.column_stack([np.ones(p), np.ones(p), np.ones(p)]).T.astype(np.float64)
sum_wt_model = [p, p, p]

altier_path = [os.getcwd() + '/pickle/', os.getcwd() + '/../pickle/']
@vectorize(['float64(int64, float64, int64)'], nopython=True, target='parallel')
def psr_wt_sing_gamma(psrno,gamma, season):

    tt_upt = t_upt[season]
    l = msdec_bin_indices[psrno]
    wt_ac_temp = np.zeros(len(enus), dtype=np.float64)
    for i in prange(len(enus)):
        wt_ac_temp[i] = np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * (enus[i]**gamma))


    return np.trapz(wt_ac_temp, enus)


if f'wt_acc_{len(enus)}_bins_C_wt.pkl' in os.listdir(altier_path[0]):# or f'wt_acc.pkl_{len(enus)}' in os.listdir(altier_path[1]):
    print("Loading wt_acc from pickle")
    
    with open(altier_path[0] + f'wt_acc_{len(enus)}_bins_C_wt.pkl', 'rb') as f:
        wt_acc = pickle.load(f)
    
    
    print("Loaded wt_acc from pickle with nbins= ", len(enus))

else:
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
    with open(altier_path[0] + f'wt_acc_{len(enus)}_bins_C_wt.pkl', 'wb') as f:
        pickle.dump(wt_acc, f)
    print("Calculated wt_acc for all pulsars and seasons and gamma")
    

season_walls = np.array(icparts.copy(), dtype=np.float64)

#Compute the signal PDF for all neutrinos as per eqns 6, 7 and weights as per eqn 8 of 2205.15963

@njit(nogil=True, parallel=True)
def S_ijk(nu): 
    
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


    sij = S_ijk(nu)
    season = 0
    for i in range(10):
        if season_walls[i] <= nu and nu < season_walls[i+1]:
            season = i
            break

    return np.sum(np.multiply(sij, np.multiply(w_models[ws], weight[gamma_index][season])/np.sum(np.multiply(w_models[ws], weight[gamma_index][season]))))      #1/rad**2

@njit(parallel=True, nogil=True)
def Sik_sing_s_g(gamma_index, ws):#, wt_acc=wt_acc, w_models=w_models):
   
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
    
    # count = np.sum(np.abs(np.subtract(icdec, icdec[nu])) <= cone)
    count=0
    for i in prange(len(icdec)):
        if abs(icdec[i] - icdec[nu]) <= cone:
            count+=1
    binwidth = (np.sin(np.deg2rad(icdec[nu] + cone)) - np.sin(np.deg2rad(icdec[nu] - cone)))*2*np.pi
    return count/(binwidth * lnu)           #No units or sr**-1


#Pickle
if os.path.isfile(altier_path[0] + f'all_Bi_C_cone_{cone_deg}.pkl'):
    print("Loading all_Bi from pickle...")
    with open(altier_path[0] + f'all_Bi_C_cone_{cone_deg}.pkl', 'rb') as f:
        all_Bi = pickle.load(f)
    print("Loaded all_Bi from pickle")
else:
    print("\nCalculating Bi for all neutrinos\n")
    all_Bi = Bi_stacked_compute(np.arange(lnu), 5)
    # all_Bi+=1e-90
    print("\nCalculated Bi for all neutrinos")
    #Save to pickle
    with open(altier_path[0] + f'all_Bi_C_cone_{cone_deg}.pkl', 'wb') as f:
        pickle.dump(all_Bi, f)
        

@vectorize(['float64(int64, float64, float64, int64)'], nopython=True, target='parallel')
def ns_singleseason_sing_psr_HAT(psrno,gamma, phi0, season):
   

    tt_upt = t_upt[season]

        
    l = msdec_bin_indices[psrno]
     
        
    ns_temp = np.zeros(len(enus), dtype=np.float64)
    for i in prange(len(enus)):
        ns_temp[i] += np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * phi0 * (enus[i]/(10**14))**gamma)  

    return np.trapz(ns_temp, enus)  #in s cm2 eV

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
Ns = lnu#np.count_nonzero(nuind+1)

def ns_HAT_all_season_all_psr_sing_gamma_wmodel_1(gamma, e_nus=enus, phi0=1):
    ns_hat = 0
    ns_hat_wt = 0

    for season in tqdm(prange(10)):

        ns_hat = ns_singleseason_sing_psr_HAT(prange(p), gamma, phi0, season)
        ns_hat_wt += np.dot(w_models[0]/ np.sum(w_models[0]) , ns_hat)

    return np.array([np.sum(ns_hat_wt)], dtype=np.float64)

############################################################################################################################################################################

# WMODEL = 1 with all pulsars and seasons



if os.path.isfile(altier_path[0] + f'all_Si_wmodel_1_{len(enus)}_b_all_psr_wmod1_cone_{cone_deg}.pkl'):
    print("Loading all_Si_wmodel_1 from pickle")
    with open(altier_path[0] + f'all_Si_wmodel_1_{len(enus)}_b_all_psr_wmod1_cone_{cone_deg}.pkl', 'rb') as f:
        all_Si_wmodel_1 = pickle.load(f)
    print("Loaded all_Si_wmodel_1 from pickle with nbins =", len(enus))
else:


    print("\nCalculating S_i for all neutrinos and gammas wmodel=1 all pulsars...\n")

    all_Si_wmodel_1 = []
    tmp_wt_acc = []

    for gamma_index in tqdm(prange(4)):
        tmp_wt_acc.append(Sik_sing_s_g(gamma_index, 0))

    all_Si_wmodel_1.append([tmp_wt_acc])
    tmp_wt_acc = []
    all_Si_wmodel_1 = np.asfarray(all_Si_wmodel_1[0])

    print("Calculated S_i for all neutrinos and gammas and weighting schemes")
    #Save to pickle
    with open(altier_path[0] + f'all_Si_wmodel_1_{len(enus)}_b_all_psr_wmod1_cone_{cone_deg}.pkl', 'wb') as f:
        pickle.dump(all_Si_wmodel_1, f)


arr = []
if os.path.isfile(altier_path[0] + f'ns_all_ws_{len(enus)}_b_all_psr_wmod1_cone_{cone_deg}.pkl'):
    print("Loading ns_hat from pickle...")
    with open(altier_path[0] + f'ns_all_ws_{len(enus)}_b_all_psr_wmod1_cone_{cone_deg}.pkl', 'rb') as f:
        arr = pickle.load(f)
    print("Loaded ns_hat from pickle with nbins =", len(enus))
else:
    print("\nCalculating ns_HAT for all gamma and wmodel=1...\n")

    arr=[]
    for gamma in prange(len(gamma_arr)):
        tmp = ns_HAT_all_season_all_psr_sing_gamma_wmodel_1(gamma_arr[gamma])
        np.savetxt('outputs/ns_hat_wt_wt_gamma_{}.txt'.format(gamma_arr[gamma]), tmp)
        arr.append(tmp)
        tmp = []

    arr = np.array(arr, dtype=np.float64)
    with open(altier_path[0] + f'ns_all_ws_{len(enus)}_b_all_psr_wmod1_cone_{cone_deg}.pkl', 'wb') as f:
        pickle.dump(arr, f)
    print("\nCalculationed ns_HAT for all gamma and wmodel=1")



print('\nCALCULATING TS FOR ALL PSRS FOR ALL GAMMAS FOR wmodel=1\n')
all_TSS_wmod1 = []
for gamma in prange(len(gamma_arr)):
    print("gamma = {}".format(gamma))
    # tmp = np.zeros(len(phio))
    # for season in tqdm(range(10)):
    t2mp = np.asfarray(all_Si_wmodel_1[0][gamma])
    @njit(nogil=True)
    def TS_for_all_psrs2(nsa):  
        return TS_st_vec(nsa, t2mp, all_Bi, Ns)      #No units
    temp = []
    for phi in tqdm(prange(len(phio))):
        try:
            temp.append(TS_for_all_psrs2(arr[gamma]*phio[phi]))
            # print('\a')
        except:
            temp_dd = TS_st_vec(np.round(np.float64(arr[gamma]*phio[phi]), 6), t2mp, all_Bi, Ns)
            temp.append(temp_dd)
            # print(np.shape(arr[gamma]*phio[phi]))
            # print(np.shape(t2mp))
            # exit()
        
    all_TSS_wmod1.append(temp)
    temp = []

print('\nCALCULATED TS FOR ALL PSRS FOR ALL GAMMAS FOR wmodel=1')

all_TSS_wmod1 = np.array(all_TSS_wmod1, dtype=np.float64)

all_TSS_wmod1 = np.asarray(all_TSS_wmod1)
gamma_arr = np.asarray(gamma_arr)

all_e_UL_wmod1 = []
e_decade = [1e12, 1e13, 1e14, 1e15, 1e16]
for e_UL in e_decade:
    e2dfde = []

    for gamma in prange(len(gamma_arr)):
        temp = []
        for phi in range(len(phio)):
            temp.append( e_UL**2 * dfde(e_UL, gamma_arr[gamma], phio[phi]))        #in eV
        e2dfde.append(temp)
    e2dfde = np.asarray(e2dfde)

    all_e_UL_wmod1.append(e2dfde)
mark = ['^', 'o', 's', 'd']


for g in range(3):
    print(phio[np.argmax(all_TSS_wmod1[g])]) 


all_UL_wmod1 = []

for gamma in prange(len(gamma_arr)):
    temp = []
    for i in all_e_UL_wmod1:
        dist_g = interp.interp1d(all_TSS_wmod1[gamma], i[gamma]/1e9)
        temp.append(dist_g(-3.84))

        
    all_UL_wmod1.append(temp)
e2dfde_wmod1 = np.asarray([1e28 * dfde(1e14, g, 1) * phio for g in gamma_arr])




############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
# # ALL W_MODELS with valid dDM and s1400

mspdata = mspdata[mspdata['DIST_DM'] != '*']
mspdata = mspdata[mspdata['S1400'] != '*']
msdist = np.array(mspdata['DIST_DM'], dtype=np.float64)
mss1400 = np.array(mspdata['S1400'], dtype=np.float64)
msra = np.array(mspdata['RAJD'], dtype=np.float64)
msdec = np.array(mspdata['DECJD'], dtype=np.float64)
p = len(mspdata)

w_models = np.column_stack([np.ones(p), np.divide(1, msdist**2), mss1400]).T.astype(np.float64)
sum_wt_model = np.array([p, np.sum(np.divide(1, msdist**2)), np.sum(mss1400)], dtype=np.float64)

if f'wt_acc_{len(enus)}_b_valid_dDM_s14.pkl' in os.listdir(altier_path[0]):
    print("Loading wt_acc from pickle")
    
    with open(altier_path[0] + f'wt_acc_{len(enus)}_b_valid_dDM_s14.pkl', 'rb') as f:
        wt_acc = pickle.load(f)
    
    
    print("Loaded wt_acc from pickle with nbins= ", len(enus))

else:
    print("Calculating wt_acc for all pulsars and seasons and gamma")
    wt_acc = []
    for gamma in prange(len(gamma_arr)):
        wt_allpsr = []
        for season in tqdm(prange(10)):
            wt_allpsr.append(np.array(psr_wt_sing_gamma(prange(p), gamma_arr[gamma], season), dtype=np.float64))
            
        wt_acc.append(wt_allpsr)
        wt_allpsr = []
        
    wt_acc = np.asfarray(wt_acc, dtype=np.float64)
    with open(altier_path[0] + f'wt_acc_{len(enus)}_b_valid_dDM_s14.pkl', 'wb') as f:
        pickle.dump(wt_acc, f)
    print("Calculated wt_acc for all pulsars and seasons and gamma")


if os.path.isfile(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_b_valid_dDM_s14.pkl'):
    print("Loading all_Si_ws_g_s from pickle")
    with open(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_b_valid_dDM_s14.pkl', 'rb') as f:
        all_Si_valid_psr_wmodel_all = pickle.load(f)
    print("Loaded all_Si_ws_g_s from pickle with nbins =", len(enus))
else:


    print("\nCalculating S_i for all neutrinos and gammas and weighting schemes...\n")

    all_Si_valid_psr_wmodel_all = []
    # tmp = []
    tmp_wt_acc = []
    tmp_wt_acc_w_dist = []
    tmp_wt_acc_w_s1400 = []

    for gamma_index in tqdm(prange(4)):
        # for season in tqdm(prange(10)):
        # tmp.append(Sik_sing_s_g(gamma_index, -1))
        tmp_wt_acc.append(Sik_sing_s_g(gamma_index, 0))
        tmp_wt_acc_w_dist.append(Sik_sing_s_g(gamma_index, 1))
        tmp_wt_acc_w_s1400.append(Sik_sing_s_g(gamma_index, 2))


    all_Si_valid_psr_wmodel_all.append([tmp_wt_acc, tmp_wt_acc_w_dist, tmp_wt_acc_w_s1400])
    # tmp = []
    tmp_wt_acc = []
    tmp_wt_acc_w_dist = []
    tmp_wt_acc_w_s1400 = []
    all_Si_valid_psr_wmodel_all = np.asfarray(all_Si_valid_psr_wmodel_all[0])

    print("Calculated S_i for all neutrinos and gammas and weighting schemes")
    #Save to pickle
    with open(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_b_valid_dDM_s14.pkl', 'wb') as f:
        pickle.dump(all_Si_valid_psr_wmodel_all, f)


# ### ns


def ns_HAT_all_season_all_psr_sing_gamma_wt_d2_s1400_wtht_weights(gamma, e_nus=enus, phi0=1):
    ns_hat = 0
    ns_hat_wt = 0
    ns_hat_wt_dist = 0
    ns_hat_wt_s1400 = 0
    for season in tqdm(prange(10)):

        ns_hat = ns_singleseason_sing_psr_HAT(prange(p), gamma, phi0, season)
        ns_hat_wt += np.dot(w_models[0]/ np.sum(w_models[0]) , ns_hat)
        ns_hat_wt_dist += np.dot(w_models[1]/ np.sum(w_models[1]) , ns_hat)
        ns_hat_wt_s1400 += np.dot(w_models[2]/ np.sum(w_models[2]), ns_hat)
    return np.array([ns_hat_wt, ns_hat_wt_dist, ns_hat_wt_s1400], dtype=np.float64)

#Pickle
arr = []
if os.path.isfile(altier_path[0] + f'ns_all_ws_{len(enus)}_b_valid_dDM_s14.pkl'):
    print("Loading ns_hat from pickle...")
    with open(altier_path[0] + f'ns_all_ws_{len(enus)}_b_valid_dDM_s14.pkl', 'rb') as f:
        arr = pickle.load(f)
    print("Loaded ns_hat from pickle with nbins =", len(enus))
else:
    print("\nCalculating ns_HAT for all gamma and weighting schemes...\n")

    arr=[]
    for gamma in prange(len(gamma_arr)):
        tmp = ns_HAT_all_season_all_psr_sing_gamma_wt_d2_s1400_wtht_weights(gamma_arr[gamma])
        np.savetxt('outputs/ns_hat_wt_wt_gamma_{}.txt'.format(gamma_arr[gamma]), tmp)
        arr.append(tmp)
        tmp = []

    arr = np.array(arr, dtype=np.float64)
    with open(altier_path[0] + f'ns_all_ws_{len(enus)}_b_valid_dDM_s14.pkl', 'wb') as f:
        pickle.dump(arr, f)
    print("\nCalculationed ns_HAT for all gamma and weighting schemes")
    
    
    
    
print('\nCALCULATING TS FOR VALID PSRS FOR ALL GAMMAS FOR ALL WEIGHTS\n')
print(np.shape(arr))
all_TSS_wt_d2_wt_s = []
for ws in prange(len(all_Si_valid_psr_wmodel_all)):
    tmpp = []
    print("ws = {}".format(ws))
    for gamma in prange(len(gamma_arr)):
        print("gamma = {}".format(gamma))
        # tmp = np.zeros(len(phio))
        # for season in tqdm(range(10)):
        t2mp = np.asfarray(all_Si_valid_psr_wmodel_all[ws][gamma])
        @njit(nogil=True)
        def TS_for_all_psrs2(nsa):  
            return TS_st_vec(nsa, t2mp, all_Bi, Ns)      #No units
        temp = []
        for phi in tqdm(prange(len(phio))):
            try:
                temp.append(TS_for_all_psrs2(arr[gamma][ws]*phio[phi]))
            
            except:
                temp_dd = TS_st_vec(np.float64(arr[gamma][ws]*phio[phi]), t2mp, all_Bi, Ns)
                temp.append(temp_dd)    
        
        tmpp.append(temp)
        temp = []
    all_TSS_wt_d2_wt_s.append(tmpp)
    tmpp = []

print('\nCALCULATED TS FOR VALID PSRS FOR ALL GAMMAS FOR ALL WEIGHTS')



all_TSS_wt_d2_wt_s = np.array(all_TSS_wt_d2_wt_s, dtype=np.float64)

# PLOTS FOR VALID PULSARS

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

fig, axs = plt.subplots(1,3, figsize=(18, 6))

for i in range(3):

        for gamma in [ 1, 2, 3]:#range(4):
        
        
                axs[i].plot( 1e28 * dfde(1e14, gamma_arr[gamma], 1) *phio /1e9, all_TSS_wt_d2_wt_s[i][gamma], label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2)# + ' with wt')    #in GeV

        

        axs[i].set_title(['Weighting scheme:  $\mathsf{\mathbf{w_{model} = \dfrac{1}{d_{DM}^2}}}$', 'Weighting scheme:  $\mathsf{\mathbf{w_{model} = s_{1400}}}$'][i-1] , fontdict=smallerfont)


                
        
        axs[i].hlines(-3.84, 1e-20, 1e-5, linestyles='dashed', lw=2.2, ls='-.', label='$TS = -3.84$', color='lightcoral')
        axs[i].set_xscale('log')
        axs[i].set_xlabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ at 100 TeV ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$)', fontdict=axesfont)
        axs[i].set_ylabel('TS', fontdict=axesfont, fontsize=20)
        axs[i].xaxis.set_tick_params(labelsize=15)
        axs[i].yaxis.set_tick_params(labelsize=15)
        axs[i].legend(prop={'size':15}, framealpha=0, loc='lower left')
        axs[i].set_ylim(-20, 5)
        axs[i].set_xlim(0.95e-14, 1e-6)

if cone_deg == 5:
    plt.suptitle('TS vs Total Neutrino Flux at 100 TeV', fontweight='bold', fontsize=20, fontfamily='serif')
else:
    plt.suptitle('TS vs Total Neutrino Flux at 100 TeV (' + str(cone_deg) + '$^{\circ}$)', fontweight='bold', fontsize=20, fontfamily='serif')

plt.tight_layout()
plt.savefig(f'outputs/TS_vs_E2dfde_bins={len(enus)}_wmodel_all_{cone_deg}_valid_dDM_s14.pdf')

all_e_UL = []
e_decade = [1e12, 1e13, 1e14, 1e15, 1e16]
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

all_UL_w_all = []
for ws in range(3):
    print(ws, '#'*20)
    ul_all_gamma = []
    for gamma in prange(len(gamma_arr)):
        temp = []
        for i in all_e_UL:
            dist_g = interp.interp1d(all_TSS_wt_d2_wt_s[ws][gamma], i[gamma]/1e9)
            temp.append(dist_g(-3.84))
        
        print(max(temp), gamma_arr[gamma])    
        ul_all_gamma.append(temp)
    all_UL_w_all.append(ul_all_gamma)
e2dfde = all_e_UL[1]

f = open('outputs/UL_ALL_w.txt', 'w')

f.write(str(3 * all_UL_w_all[0][2][1]))
f.write(str(3 * all_UL_w_all[1][2][1]))
f.write(str(3 * all_UL_w_all[2][2][1]))
f.close()

# pickle.dump(all_UL_wmodel1, open('outputs/UL_ALL_wmodel1.txt', 'wb'))
pickle.dump(all_UL_w_all, open('outputs/UL_ALL_wmodel_all.txt', 'wb'))
# plt.style.use('default')

#PLOTS FOR 95% UPPER LIMIT 
fig, axs = plt.subplots(1,3, figsize=(18, 6))


for i in range(3):
    for gamma in range(1, len(gamma_arr)):

        axs[i].plot(np.divide(e_decade, 1e9), np.multiply(all_UL_w_all[i][gamma], 3), label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2, ls='-')# + ' with wt')    #in GeV
        # axs[i].scatter(np.divide(e_decade, 1e9), np.multiply(all_UL_wd_ws[i-1][gamma], 3))
    
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlabel('E$_{\u03BD}$ (GeV)', fontdict=axesfont)
    axs[i].set_ylabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$ )', fontdict=axesfont)
    axs[i].xaxis.set_tick_params(labelsize=15)
    axs[i].yaxis.set_tick_params(labelsize=15)
    
    axs[i].legend(prop={'size':16}, framealpha=0, loc='lower left')


axs[0].set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = 1}}$', fontdict=smallerfont)
axs[1].set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = \dfrac{1}{d_{DM}^2}}}$' , fontdict=smallerfont)
axs[2].set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = s_{1400}}}$', fontdict=smallerfont)

if cone_deg == 5:
    plt.suptitle('95% UL of Total Energy Flux vs Neutrino Energy', fontweight='bold', fontsize=20, fontfamily='serif')
else: 
    plt.suptitle('95% UL of Total Energy Flux vs Neutrino Energy (' + str(cone_deg) + '$^{\circ}$)', fontweight='bold', fontsize=20, fontfamily='serif')
    
plt.tight_layout()
plt.savefig(f'outputs/UL_bins={len(enus)}_wmodel_all_{cone_deg}_valid_dDM_s14.pdf')
# plt.show()
print('done')


############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################

# PLOTS FOR
# wmodel = 1 with all pulsars and seasons
# wmodel = 1/dDM^2 and s1400 with valid dDM and s1400



fig, axs = plt.subplots(1,3, figsize=(18, 6))

for gamma in [1, 2, 3]:

    axs[0].plot( 1e28 * dfde(1e14, 1, 1) *phio /1e9, all_TSS_wmod1[gamma], label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2)# + ' with wt')    #in GeV

for i in range(1,3):

        for gamma in [ 1, 2, 3]:#range(4):
        
        
                axs[i].plot( 1e28 * dfde(1e14, gamma_arr[gamma], 1) *phio /1e9, all_TSS_wt_d2_wt_s[i][gamma], label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2)# + ' with wt')    #in GeV

        

        axs[i].set_title(['Weighting scheme:  $\mathsf{\mathbf{w_{model} = \dfrac{1}{d_{DM}^2}}}$', 'Weighting scheme:  $\mathsf{\mathbf{w_{model} = s_{1400}}}$'][i-1] , fontdict=smallerfont)


                
for i in range(3):        
        axs[i].hlines(-3.84, 1e-20, 1e-5, linestyles='dashed', lw=2.2, ls='-.', label='$TS = -3.84$', color='lightcoral')
        axs[i].set_xscale('log')
        axs[i].set_xlabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ at 100 TeV ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$)', fontdict=axesfont)
        axs[i].set_ylabel('TS', fontdict=axesfont, fontsize=20)
        axs[i].xaxis.set_tick_params(labelsize=15)
        axs[i].yaxis.set_tick_params(labelsize=15)
        axs[i].legend(prop={'size':15}, framealpha=0, loc='lower left')
        axs[i].set_ylim(-20, 5)
        axs[i].set_xlim(0.95e-14, 1e-6)

if cone_deg == 5:
    plt.suptitle('TS vs Total Neutrino Flux at 100 TeV', fontweight='bold', fontsize=20, fontfamily='serif')
else:
    plt.suptitle('TS vs Total Neutrino Flux at 100 TeV (' + str(cone_deg) + '$^{\circ}$)', fontweight='bold', fontsize=20, fontfamily='serif')

plt.tight_layout()
plt.savefig(f'outputs/TS_vs_E2dfde_bins={len(enus)}_wmodel_all_{cone_deg}_mixed.pdf')

all_e_UL = []
e_decade = [1e12, 1e13, 1e14, 1e15, 1e16]
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

all_UL_w_all = []
for ws in range(3):
    print(ws, '#'*20)
    ul_all_gamma = []
    for gamma in prange(len(gamma_arr)):
        temp = []
        for i in all_e_UL:
            dist_g = interp.interp1d(all_TSS_wt_d2_wt_s[ws][gamma], i[gamma]/1e9)
            temp.append(dist_g(-3.84))
        
        print(max(temp), gamma_arr[gamma])    
        ul_all_gamma.append(temp)
    all_UL_w_all.append(ul_all_gamma)
e2dfde = all_e_UL[1]

#PLOTS FOR 95% UPPER LIMIT 
fig, axs = plt.subplots(1,3, figsize=(18, 6))


for gamma in range(1, len(gamma_arr)):
    axs[0].plot(np.divide(e_decade, 1e9), np.multiply(all_UL_wmod1[gamma], 3), label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2, ls='-')# + ' with wt')    #in GeV
    # axs[0].scatter(np.divide(e_decade, 1e9), np.multiply(all_UL_wd_ws[0][gamma], 3))

for i in range(1,3):
    for gamma in range(1, len(gamma_arr)):

        axs[i].plot(np.divide(e_decade, 1e9), np.multiply(all_UL_w_all[i][gamma], 3), label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2, ls='-')# + ' with wt')    #in GeV
        # axs[i].scatter(np.divide(e_decade, 1e9), np.multiply(all_UL_wd_ws[i-1][gamma], 3))

for i in range(3):    
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlabel('E$_{\u03BD}$ (GeV)', fontdict=axesfont)
    axs[i].set_ylabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$ )', fontdict=axesfont)
    axs[i].xaxis.set_tick_params(labelsize=15)
    axs[i].yaxis.set_tick_params(labelsize=15)
    
    axs[i].legend(prop={'size':16}, framealpha=0, loc='lower left')


axs[0].set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = 1}}$', fontdict=smallerfont)
axs[1].set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = \dfrac{1}{d_{DM}^2}}}$' , fontdict=smallerfont)
axs[2].set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = s_{1400}}}$', fontdict=smallerfont)

if cone_deg == 5:
    plt.suptitle('95% UL of Total Energy Flux vs Neutrino Energy', fontweight='bold', fontsize=20, fontfamily='serif')
else: 
    plt.suptitle('95% UL of Total Energy Flux vs Neutrino Energy (' + str(cone_deg) + '$^{\circ}$)', fontweight='bold', fontsize=20, fontfamily='serif')
    
plt.tight_layout()
plt.savefig(f'outputs/UL_bins_{len(enus)}_wmodel_all_{cone_deg}_mixed.pdf')
# plt.show()
print('done')