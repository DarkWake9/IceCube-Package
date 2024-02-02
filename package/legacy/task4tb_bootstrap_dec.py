import numpy as np
import os
import multiprocessing as mul
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads, vectorize, guvectorize, cuda
from tqdm import tqdm
from core.signal_bag import *
from core.stacking_analysis import *
from core.req_arrays import *
import pickle
import scipy.stats as st
import scipy.interpolate as interp




num_threads = 56
set_num_threads(num_threads)
# UNCOMMENT FOR LINEAR BINS
# all_enu = np.linspace(10**11.001, 10**18.999, 1000)
all_enu = e_nu_wall

# enus = 0.5*(all_enu[1:]+all_enu[:-1])
# UNCOMMENT FOR DENSER LOGARITHMIC BINS, optimal nbins is 1e6
enus = np.logspace(11.001, 18.999, int(1e5))
enus_bin_indices = np.zeros(len(enus), dtype=np.int64)

for i in prange(len(enus)):
    enus_bin_indices[i] = np.digitize(enus[i], e_nu_wall) - 1
    
gamma_arr = [-2, -2.2, -2.53, -3]
phio = np.logspace(-38, -26, 1000)
eareaa = [i.astype(np.float64) for i in earea]
eareaa = np.asfarray(eareaa, dtype=np.float64)
eareaa[0][0]
earea2 = np.asfortranarray(earea)
earea2 = earea2[0].astype(np.float64)
print("\nNumber of energy bins: ", len(enus))
print("\nNumber of phi bins: ", len(phio))


dec_offset = 2
n_cyc = 10
DEFAULT_PERM = np.arange(0, len(msdec), dtype=np.int64)

### Bootstrap 3:

#- Uniform generation of DEC
for cyc in prange(n_cyc):
    prm = np.random.permutation(DEFAULT_PERM)
    msdec = msdec[prm] + 2*np.random.uniform(size=len(msdec))
    # msdec = np.random.uniform(-85, 85, len(msdec))
    msdec_bin_indices = np.zeros(p, dtype=np.int64)
    for i in prange(p):
        msdec_bin_indices[i] = np.digitize(msdec[i], dec_nu) - 1

    #CHANGING TO LINEAR BINS RESULTS IN STRAIGHT LINES

    # print("\nNumber of threads: ", num_threads)
    
    print(f"\nCalculating weights for {cyc} time...\n\n")
    


    altier_path = [os.getcwd() + '/pickle/', os.getcwd() + '/../pickle/']
    @vectorize(['float64(int64, float64, int64)'], nopython=True, target='parallel')
    def psr_wt_sing_gamma(psrno,gamma, season):

        tt_upt = t_upt[season]
        l = msdec_bin_indices[psrno]
        wt_ac_temp = np.zeros(len(enus), dtype=np.float64)
        for i in prange(len(enus)):
            wt_ac_temp[i] = np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * (enus[i]**gamma))


        return np.trapz(wt_ac_temp, enus)

    # if f'wt_acc_{len(enus)}_bins_C_wt.pkl' in os.listdir(altier_path[0]):# or f'wt_acc.pkl_{len(enus)}' in os.listdir(altier_path[1]):
    #     print("Loading wt_acc from pickle")
        
    #     with open(altier_path[0] + f'wt_acc_{len(enus)}_bins_C_wt.pkl', 'rb') as f:
    #         wt_acc = pickle.load(f)
        
        
    #     print("Loaded wt_acc from pickle with nbins= ", len(enus))

    
    print(f"Calculating wt_acc for all pulsars and seasons and gamma for {cyc} time\n\n")
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
    season_walls = np.asarray([0, 36900, 143911, 237044, 373288, 486146, 608687, 735732, 865043, 988700, 1134450])
    season_widts= np.diff(season_walls)
    #Compute the signal PDF for all neutrinos as per eqns 6, 7 and weights as per eqn 8 of 2205.15963

    w_models = np.array([np.ones(p)])

    @njit(nogil=True)
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
        ang2 = hvovec(msra, msdec, icra[nu], icdec[nu], rad=True) ** 2      #rad**2
        sg = np.deg2rad(icang[nu]) ** 2                                     #rad**2
        return np.divide(np.exp(-1 * np.divide(ang2, 2*sg)), (2 * np.pi * sg))      #1/rad**2


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
    def Bi_stacked_compute(nu, cone=5):

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
        return count/(binwidth * N_ic)           #No units or sr**-1
    #Pickle
    if os.path.isfile(altier_path[0] + f'all_Bi_C.pkl'):
        print("Loading all_Bi from pickle...")
        with open(altier_path[0] + f'all_Bi_C.pkl', 'rb') as f:
            all_Bi = pickle.load(f)
        print("Loaded all_Bi from pickle")
    else:
        print("\nCalculating Bi for all neutrinos\n")
        all_Bi = Bi_stacked_compute(np.arange(lnu), 5)
        # all_Bi+=1e-90
        print("\nCalculated Bi for all neutrinos")
        #Save to pickle
        with open(altier_path[0] + f'all_Bi_C.pkl', 'wb') as f:
            pickle.dump(all_Bi, f)
            

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


    len(msra)


    # @njit(nogil=True)
    # def TS_for_all_psrs2(nsa):  
    #     return Ts_arr2(nsa, t2mp, all_Bi, Ns) 


    @jit(nopython=True)
    def Pr(x, Ns, S, B):
        nsN = x/Ns
        return np.add(np.multiply(nsN , S), np.multiply(np.subtract(1, nsN), B))



    @njit(nogil=True)
    def TS_st_vec(x, S, B, Ns):
        nsN = x/Ns
        pr = np.add(np.multiply(nsN , S), np.multiply(np.subtract(1, nsN), B))
        return np.sum(np.asfarray(2*np.log(pr/B)))

    lnu = 1134450
    Ns = lnu#np.count_nonzero(nuind+1)


    phio = np.logspace(-38, -20, 1000) 

    
    # # W_MODEL=1 ONLY


    # if os.path.isfile(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl'):
    #     print("Loading all_Si_ws_g_s from pickle")
    #     with open(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl', 'rb') as f:
    #         all_Si_ws_g_s = pickle.load(f)
    #     print("Loaded all_Si_ws_g_s from pickle with nbins =", len(enus))
    # else:


    print("\nCalculating S_i for all neutrinos and gammas and weighting schemes...\n")

    all_Si_ws_g_s = []
    # tmp = []
    tmp_wt_acc = []
    # tmp_wt_acc_w_dist = []
    # tmp_wt_acc_w_s1400 = []

    for gamma_index in tqdm(prange(4)):
        tmp_wt_acc.append(Sik_sing_s_g(gamma_index, 0))
        
        
    all_Si_ws_g_s.append([tmp_wt_acc])
    
    tmp_wt_acc = []
    all_Si_ws_g_s = np.asfarray(all_Si_ws_g_s[0])

    print("Calculated S_i for all neutrinos and gammas and weighting schemes")
        #Save to pickle
    with open(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl', 'wb') as f:
        pickle.dump(all_Si_ws_g_s, f)

    
    # ### ns


    def ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma, e_nus=enus, phi0=1):
        ns_hat = 0
        ns_hat_wt = 0

        for season in tqdm(prange(10)):

            ns_hat = ns_singleseason_sing_psr_HAT(prange(p), gamma, phi0, season)
            ns_hat_wt += ns_hat

        return np.array([np.sum(ns_hat_wt)], dtype=np.float64)
    #Pickle
    arr = []
    # if os.path.isfile(altier_path[0] + f'ns_all_ws_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl'):
    #     print("Loading ns_hat from pickle...")
    #     with open(altier_path[0] + f'ns_all_ws_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl', 'rb') as f:
    #         arr = pickle.load(f)
    #     print("Loaded ns_hat from pickle with nbins =", len(enus))
    # else:
    print("\nCalculating ns_HAT for all gamma and weighting schemes...\n")

    arr=[]
    for gamma in prange(len(gamma_arr)):
        tmp = ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma_arr[gamma])
        np.savetxt('outputs/ns_hat_wt_wt_gamma_{}.txt'.format(gamma_arr[gamma]), tmp)
        arr.append(tmp)
        tmp = []

    arr = np.array(arr, dtype=np.float64)
    with open(altier_path[0] + f'ns_all_ws_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl', 'wb') as f:
        pickle.dump(arr, f)
    print("\nCalculationed ns_HAT for all gamma and weighting schemes")



    print('\nCALCULATING TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS\n')

    all_TSS_wmod1 = []
    for gamma in prange(len(gamma_arr)):
        print("gamma = {}".format(gamma))
        # tmp = np.zeros(len(phio))
        # for season in tqdm(range(10)):
        t2mp = np.asfarray(all_Si_ws_g_s[0][gamma])
        @njit(nogil=True)
        def TS_for_all_psrs2(nsa):  
            return TS_st_vec(nsa, t2mp, all_Bi, Ns)      #No units
        temp = []
        for phi in tqdm(prange(len(phio))):
            temp.append(TS_for_all_psrs2(arr[gamma]*phio[phi]))
        all_TSS_wmod1.append(temp)
        temp = []

    print('\nCALCULATED TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS')



    all_TSS_wmod1 = np.array(all_TSS_wmod1, dtype=np.float64)


    for g in range(len(gamma_arr)):
        print(min(all_TSS_wmod1[g]), max(all_TSS_wmod1[g]))

        print('wt\n')

    with open(altier_path[0] + f'all_TSS_{len(enus)}_{len(enus)}_bins_C_wt_all_psr_wmod1_btsp4_d_off_{dec_offset}_cyc_{cyc}.pkl', 'wb') as f:
        pickle.dump(all_TSS_wmod1, f)


    all_TSS_wmod1 = np.asarray(all_TSS_wmod1)
    gamma_arr = np.asarray(gamma_arr)

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
        print(phio[np.argmax(all_TSS_wmod1[g])]) 


    all_UL = []

    for gamma in prange(len(gamma_arr)):
        temp = []
        for i in all_e_UL:
            dist_g = interp.interp1d(all_TSS_wmod1[gamma], i[gamma]/1e9)
            temp.append(dist_g(-3.84))

            
        all_UL.append(temp)
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

        

    axs.set_title('Weighting scheme:  $\mathsf{\mathbf{w_{model} = 1}}$', fontdict=smallerfont)


    # for i in range(3):
        
    axs.legend(prop={'size':14}, framealpha=0, loc='lower left')
    axs.hlines(-3.84, 1e-20, 1e-5, linestyles='dashed', lw=2.2, ls='-.', label='95 % UPPER LIMIT $TS = -3.84$', color='lightcoral')
    axs.set_xscale('log')
    axs.set_xlabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ at 100 TeV ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$ )', fontdict=axesfont)
    axs.set_ylabel('TS', fontdict=axesfont, fontsize=20)
    axs.xaxis.set_tick_params(labelsize=15)
    axs.yaxis.set_tick_params(labelsize=15)

    axs.set_ylim(-220, 90)
    axs.set_xlim(0.95e-19, 1e-6)

    plt.suptitle('TS vs Total Neutrino Flux at 100 TeV', fontweight='bold', fontsize=20, fontfamily='serif')

    plt.tight_layout()
    plt.savefig(f'outputs/TS_vs_E2dfde_all_w_model_bins={len(enus)}_C_wt_all_psr_wmod1_btsp3_cyc_{cyc}.pdf')
    # plt.show()
    print(f'\nTS_vs_E2dfde_all_w_model_bins={len(enus)}_C_wt_all_psr_wmod1_cyc_{cyc}.pdf\nDONE')