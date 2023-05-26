
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
# import torch
# from core import weights


num_threads = int(14)
set_num_threads(num_threads)

def counted_season(removed_season):
        season = list(np.arange(0, 10, 1))
        season.remove(removed_season)
        return np.asfortranarray(season)

for REMOVED_SEASON in prange(10):
    print("#"*50)
    print("Removed season: ", REMOVED_SEASON)
    print("#"*50)
    

    # seasons = counted_season(REMOVED_SEASON)
    seasons = np.array([REMOVED_SEASON])


    # UNCOMMENT FOR LINEAR BINS
    # all_enu = np.linspace(10**11.001, 10**18.999, 1000)
    all_enu = e_nu_wall

    # UNCOMMENT FOR DENSER LOGARITHMIC BINS, optimal nbins is 1e6
    enus = np.logspace(11.001, 18.999, int(1e3))


    enus_bin_indices = np.zeros(len(enus), dtype=np.int64)

    for i in prange(len(enus)):
        enus_bin_indices[i] = np.digitize(enus[i], e_nu_wall) - 1


    msdec_bin_indices = np.zeros(p, dtype=np.int64)
    for i in prange(p):
        msdec_bin_indices[i] = np.digitize(msdec[i], dec_nu) - 1



    gamma_arr = [-2, -2.2, -2.53, -3]
    phio = np.logspace(-38, -26, 1000) #CHANGING TO LINEAR BINS RESULTS IN STRAIGHT LINES

    # print("\nNumber of threads: ", num_threads)
    print("\nNumber of energy bins: ", len(enus))
    print("\nNumber of phi bins: ", len(phio))
    print("\nCalculating weights...\n\n")


    eareaa = [i.astype(np.float64) for i in earea]
    eareaa = np.asfarray(eareaa, dtype=np.float64)
    eareaa[0][0]
    earea2 = np.asfortranarray(earea)
    earea2 = earea2[0].astype(np.float64)




    @vectorize(['float64(int64, float64, int64)'], nopython=True, target='parallel')
    def psr_wt_sing_gamma(psrno,gamma, season):

        tt_upt = t_upt[season]
        l = msdec_bin_indices[psrno]
        wt_ac_temp = np.zeros(len(enus), dtype=np.float64)
        for i in prange(len(enus)):
            wt_ac_temp[i] = np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * enus[i]**gamma)


        return np.trapz(wt_ac_temp, enus)



    w_models = np.column_stack([np.ones(p), 1/(msdist**2), mss1400]).T.astype(np.float64)
    sum_wt_model = [1, np.sum(1/(msdist**2)), np.sum(mss1400)]


    altier_path = [os.getcwd() + '/pickle/', os.getcwd() + '/../pickle/']
    if f'wt_acc_{len(enus)}_bins_rem_s_{REMOVED_SEASON}.pkl' in os.listdir(altier_path[0]):# or f'wt_acc.pkl_{len(enus)}' in os.listdir(altier_path[1]):
        print("Loading wt_acc from pickle")
        # try:
        #     wt_acc = pickle.load(altier_path[1] + f'wt_acc.pkl_{len(enus)//2}_bins')
        # except:
        with open(altier_path[0] + f'wt_acc_{len(enus)}_bins_rem_s_{REMOVED_SEASON}.pkl', 'rb') as f:
            wt_acc = pickle.load(f)
        
        
        print("Loaded wt_acc from pickle with nbins= ", len(enus))

    else:
        print("Calculating wt_acc for all pulsars and seasons and gamma with removed season = ", REMOVED_SEASON)
        wt_acc = []
        for gamma in prange(len(gamma_arr)):
            wt_allpsr = []
            for season in tqdm(prange(9)):
                wt_allpsr.append(np.array(psr_wt_sing_gamma(prange(p), gamma_arr[gamma], seasons[season]), dtype=np.float64))
                # tmp = []
            wt_acc.append(wt_allpsr)
            wt_allpsr = []
            
        wt_acc = np.asfarray(wt_acc, dtype=np.float64)
        with open(altier_path[0] + f'wt_acc_{len(enus)}_bins_rem_s_{REMOVED_SEASON}.pkl', 'wb') as f:
            pickle.dump(wt_acc, f)
        print("Calculated wt_acc for all pulsars and seasons and gamma")


    season_walls = np.asarray([0, 36900, 143911, 237044, 373288, 486146, 608687, 735732, 865043, 988700, 1134450])
    season_widts= np.diff(season_walls)


    #Compute the signal PDF for all neutrinos as per eqns 6, 7 and weights as per eqn 8 of 2205.15963

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
    def Sik_sing_s_g(gamma_index, ws, icra_rm_s):#, wt_acc=wt_acc, w_models=w_models):
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
            for nu in prange(len(icra_rm_s)):
                tmp.append(np.sum(S_ijk(icra_rm_s[nu])))
            return np.array(tmp, dtype=np.float64)

        for nu in prange(len(icra_rm_s)):
            tmp.append(S_ik(icra_rm_s[nu], wt_acc, w_models, gamma_index, ws))
        return np.array(tmp, dtype=np.float64)

    icra_rm_s = np.delete(np.arange(0, 1134450), np.arange(season_walls[REMOVED_SEASON], season_walls[REMOVED_SEASON+1]))


    #Pickle
    if os.path.isfile(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_bins_only_s{REMOVED_SEASON}.pkl'):
        print("Loading all_Si_ws_g_s from pickle")
        with open(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_bins_only_s{REMOVED_SEASON}.pkl', 'rb') as f:
            all_Si_ws_g_s = pickle.load(f)
        print("Loaded all_Si_ws_g_s from pickle with nbins =", len(enus))
    else:


        print("\nCalculating S_i for all neutrinos and gammas and weighting schemes...\n")

        all_Si_ws_g_s = []
        tmp = []
        tmp_wt_acc = []
        tmp_wt_acc_w_dist = []
        tmp_wt_acc_w_s1400 = []

        for gamma_index in tqdm(prange(4)):
            # for season in tqdm(prange(10)):
            tmp.append(Sik_sing_s_g(gamma_index, -1, icra_rm_s))
            tmp_wt_acc.append(Sik_sing_s_g(gamma_index, 0, icra_rm_s))
            tmp_wt_acc_w_dist.append(Sik_sing_s_g(gamma_index, 1, icra_rm_s))
            tmp_wt_acc_w_s1400.append(Sik_sing_s_g(gamma_index, 2, icra_rm_s))


        all_Si_ws_g_s.append([tmp, tmp_wt_acc, tmp_wt_acc_w_dist, tmp_wt_acc_w_s1400])
        tmp = []
        tmp_wt_acc = []
        tmp_wt_acc_w_dist = []
        tmp_wt_acc_w_s1400 = []
        all_Si_ws_g_s = np.asfarray(all_Si_ws_g_s[0])

        print("Calculated S_i for all neutrinos and gammas and weighting schemes")
        #Save to pickle
        with open(altier_path[0] + f'all_Si_ws_g_s_{len(enus)}_bins_only_s{REMOVED_SEASON}.pkl', 'wb') as f:
            pickle.dump(all_Si_ws_g_s, f)




    arrr = icdec[icra_rm_s]

    # @jit(nopython=True)
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
        for i in prange(len(arrr)):
            if abs(arrr[i] - icdec[nu]) <= cone:
                count+=1
        binwidth = (np.sin(np.deg2rad(icdec[nu] + cone)) - np.sin(np.deg2rad(icdec[nu] - cone)))*2*np.pi
        return count/(binwidth * len(arrr))           #No units or sr**-1



    #Pickle
    if os.path.isfile(altier_path[0] + f'all_Bi_only_s{REMOVED_SEASON}.pkl'):
        print("Loading all_Bi from pickle...")
        with open(altier_path[0] + f'all_Bi_only_s{REMOVED_SEASON}.pkl', 'rb') as f:
            all_Bi = pickle.load(f)
        print("Loaded all_Bi from pickle")
    else:
        print("\nCalculating Bi for all neutrinos\n")
        all_Bi = Bi_stacked_compute(icra_rm_s, 5)
        # all_Bi+=1e-90
        print("\nCalculated Bi for all neutrinos")
        #Save to pickle
        with open(altier_path[0] + f'all_Bi_only_s{REMOVED_SEASON}.pkl', 'wb') as f:
            pickle.dump(all_Bi, f)


    arr = np.zeros(10, dtype=np.float64)


    #generates ns^ for a single pulsar a single season and a single energy


    @vectorize(['float64(int64, float64, float64, int64)'], nopython=True, target='parallel')
    def ns_singleseason_sing_psr_HAT(psrno,gamma, phi0, season):
    

        tt_upt = t_upt[season]

        l = msdec_bin_indices[psrno]

        ns_temp = np.zeros(len(enus), dtype=np.float64)
        for i in prange(len(enus)):
            ns_temp[i] += np.float64(tt_upt * earea[ea_season(season)][l*40 + enus_bin_indices[i]] * phi0 * (enus[i]/(10**14))**gamma)
        # temp_ea = np.asarray(earea[ea_season(season)])[l*40 + k]
        # return tt_upt * temp_ea * phi0 * ((enu/(10**14))**gamma)     #in s cm2 eV

        return np.trapz(ns_temp, enus)     #in s cm2 eV





    def ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma, e_nus=enus, phi0=1):
        ns_hat = 0
        ns_hat_wt = 0
        ns_hat_wt_dist = 0
        ns_hat_wt_s1400 = 0
        for season in tqdm(prange(len(seasons))):
            # for psr in prange(p):
                # wt = all_weights[0][gamma_arr.index(gamma)][psr][ea_season(season)]
            wt_sum = np.sum(wt_acc[gamma_arr.index(gamma)][season])

            wt = wt_acc[gamma_arr.index(gamma)][season]
            ns_hat = ns_singleseason_sing_psr_HAT(prange(p), gamma, phi0, seasons[season])
            
            
            ns_hat_wt += np.dot((wt/wt_sum),  ns_hat)
            ns_hat_wt_dist += np.dot(wt * w_models[1]/ np.sum(np.multiply(wt_acc[gamma_arr.index(gamma)][season], w_models[1])) , ns_hat)
            ns_hat_wt_s1400 += np.dot(wt * w_models[2]/ np.sum(np.multiply(wt_acc[gamma_arr.index(gamma)][season], w_models[2])), ns_hat)
        return np.array([np.sum(ns_hat), ns_hat_wt, ns_hat_wt_dist, ns_hat_wt_s1400], dtype=np.float64)


    #Pickle
    arr = []
    if os.path.isfile(altier_path[0] + f'ns_all_ws_{len(enus)}_bins.pkl'):
        print("Loading ns_hat from pickle...")
        with open(altier_path[0] + f'ns_all_ws_{len(enus)}_bins.pkl', 'rb') as f:
            arr = pickle.load(f)
        print("Loaded ns_hat from pickle with nbins =", len(enus))
    else:
        print("\nCalculating ns_HAT for all gamma and weighting schemes...\n")

        arr=[]
        for gamma in prange(len(gamma_arr)):
            tmp = ns_HAT_all_season_all_psr_sing_gamma_wt_wtht_weights(gamma_arr[gamma])
            np.savetxt('outputs/ns_hat_wt_wt_gamma_{}_rm_s.txt'.format(gamma_arr[gamma], REMOVED_SEASON), tmp)
            arr.append(tmp)
            tmp = []

        arr = np.array(arr, dtype=np.float64)
        with open(altier_path[0] + f'ns_all_ws_{len(enus)}_bins.pkl', 'wb') as f:
            pickle.dump(arr, f)
        print("\nCalculationed ns_HAT for all gamma and weighting schemes")


    arr.shape


    @njit(parallel=True, nogil=True)
    def season_nu(nu):
        for i in prange(10):
            if nu >= season_walls[i] and nu < season_walls[i+1]:
                return i


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


    #### [markdown]
    # TS_st_vec(0.5e6, all_Si_ws_g_s[0][0], all_Bi, Ns)


    phio = np.logspace(-38, -20, 1000)


    print('\nCALCULATING TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS\n')

    all_TSS = []
    for ws in prange(4):
        tmpp = []
        print("ws = {}".format(ws))
        for gamma in prange(len(gamma_arr)):
            print("gamma = {}".format(gamma))
            
            
            t2mp = np.asfarray(all_Si_ws_g_s[ws][gamma])
            @njit(nogil=True)
            def TS_for_all_psrs2(nsa):  
                return TS_st_vec(nsa, t2mp, all_Bi, Ns=len(icra_rm_s))      #No units
            temp = []
            for phi in tqdm(prange(len(phio))):
                temp.append(TS_for_all_psrs2(arr[gamma][ws]*phio[phi]))
            tmpp.append(temp)
            temp = []
        all_TSS.append(tmpp)
        tmpp = []

    print('\nCALCULATED TS FOR ALL PSRS FOR ALL GAMMAS FOR ALL WEIGHTS')

    #### [markdown]
    # ### $N \approx 10^6$


    all_TSS = np.array(all_TSS, dtype=np.float64)
    all_TSS.shape


    # for i in range(1, len(all_TSS)):
    plt.figure()
    histdetails = plt.hist(np.ravel(all_TSS[1:]), bins=10, density=True, histtype='step')
    # plt.ylim(0, 1000)
    plt.yscale('log')
    plt.xlabel('TS')
    plt.ylabel('Counts')
    plt.title('TS distribution')
    plt.show()

    #### [markdown]
    # histdetails = np.histogram(np.ravel(all_TSS[1:]), bins=10)


    histdetails


    len(np.ravel(all_TSS[1:]))

    #### [markdown]
    # np.count_nonzero(np.ravel(all_TSS[1:])> )


    all_TSS = np.array(all_TSS, dtype=np.float64)


    all_TSS.shape


    for w in range(1,4):
        for g in range(len(gamma_arr)):
            print(min(all_TSS[w][g]), max(all_TSS[w][g]))
            # np.savetxt(f'TS_w{w}_g{g}.txt', all_TSS[w][g])
            
        print('wt\n')

    with open(altier_path[0] + f'all_TSS_{len(enus)}_bins_only_s{REMOVED_SEASON}.pkl', 'wb') as f:
        pickle.dump(all_TSS, f)


    #Plotting

    all_TSS = np.asarray(all_TSS)
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


    all_TSS.shape


    # e2dfde = all_e_UL[1][2]
    # dist_g25 = interp.interp1d(all_TSS[1][2], e2dfde/1e9)

    # print(dist_g25(-3.84))

    all_UL = []
    for ws in range(1,4):
        ul_all_gamma = []
        for gamma in prange(len(gamma_arr)):
            temp = []
            for i in all_e_UL:
                dist_g = interp.interp1d(all_TSS[ws][gamma], i[gamma]/1e9)
                temp.append(dist_g(-3.84))

            ul_all_gamma.append(temp)
        all_UL.append(ul_all_gamma)


    e2dfde = all_e_UL[1]
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

    fig, axs = plt.subplots(1,3, figsize=(18, 6))

    for gamma in [ 1, 2, 3]:#range(4):
        
        for i in range(3):
            axs[i].plot(e2dfde[gamma]/1e9, all_TSS[i+1][gamma], label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2)# + ' with wt')    #in GeV

        

    axs[0].set_title('Weighting scheme: $\mathsf{\mathbf{w_{acc}}}$; $\mathsf{\mathbf{w_{model} = 1}}$', fontdict=smallerfont)
    axs[1].set_title('Weighting scheme: $\mathsf{\mathbf{w_{acc}}}$; $\mathsf{\mathbf{w_{model} = \dfrac{1}{d_{DM}^2}}}$' , fontdict=smallerfont)
    axs[2].set_title('Weighting scheme: $\mathsf{\mathbf{w_{acc}}}$; $\mathsf{\mathbf{w_{model} = s_{1400}}}$', fontdict=smallerfont)


    for i in range(3):
        
        axs[i].legend(prop={'size':14}, framealpha=0, loc='lower left')
        axs[i].hlines(-3.84, 1e-20, 1e-5, linestyles='dashed', lw=2.2, ls='-.', label='95 % UPPER LIMIT $TS = -3.84$', color='lightcoral')
        axs[i].set_xscale('log')
        axs[i].set_xlabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ at 100 TeV ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$ $\mathsf{\mathbf{sr^{-1}}}$)', fontdict=axesfont)
        axs[i].set_ylabel('TS', fontdict=axesfont, fontsize=20)
        axs[i].xaxis.set_tick_params(labelsize=15)
        axs[i].yaxis.set_tick_params(labelsize=15)
        
        axs[i].set_ylim(-220, 90)
        # axs[i].set_xlim(min(e2dfde[-1]), max(e2dfde[1]))

    plt.suptitle('TS vs Total Neutrino Flux at 100 TeV', fontweight='bold', fontsize=20, fontfamily='serif')

    plt.tight_layout()
    plt.savefig(f'outputs/TS_vs_E2dfde_all_w_model_bins={len(enus)}_only_s{REMOVED_SEASON}.png')
    # plt.show()


    #SIMILAR PLOTS FOR 95% UPPER LIMIT 
    fig, axs = plt.subplots(1,3, figsize=(18, 6))



    for i in range(3):
        for gamma in range(1, len(gamma_arr)):

            axs[i].plot(np.divide(e_decade, 1e9), np.multiply(all_UL[i][gamma], 3), label='$\Gamma$ = ' + str(gamma_arr[gamma]), lw=2.2, ls='-')# + ' with wt')    #in GeV

        
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('E$_{\u03BD}$ (GeV)', fontdict=axesfont)
        axs[i].set_ylabel('$\mathsf{\mathbf{E^2_{\u03BD} \dfrac{dF}{dE_{\u03BD}}}}$ at 100 TeV ($\mathsf{\mathbf{GeV}}$ $\mathsf{\mathbf{s^{-1}}}$ $\mathsf{\mathbf{cm^{-2}}}$ $\mathsf{\mathbf{sr^{-1}}}$)', fontdict=axesfont)
        axs[i].xaxis.set_tick_params(labelsize=15)
        axs[i].yaxis.set_tick_params(labelsize=15)
        
        axs[i].legend(prop={'size':14}, framealpha=0, loc='lower left')


    axs[0].set_title('Weighting scheme: $\mathsf{\mathbf{w_{acc}}}$; $\mathsf{\mathbf{w_{model} = 1}}$', fontdict=smallerfont)
    axs[1].set_title('Weighting scheme: $\mathsf{\mathbf{w_{acc}}}$; $\mathsf{\mathbf{w_{model} = \dfrac{1}{d_{DM}^2}}}$' , fontdict=smallerfont)
    axs[2].set_title('Weighting scheme: $\mathsf{\mathbf{w_{acc}}}$; $\mathsf{\mathbf{w_{model} = s_{1400}}}$', fontdict=smallerfont)


    plt.suptitle('95% UL of Total Energy Flux vs Neutrino Energy', fontweight='bold', fontsize=20, fontfamily='serif')
    plt.tight_layout()
    plt.savefig(f'outputs/UL_all_w_model_bins={len(enus)}_only_s{REMOVED_SEASON}.png')
    plt.show()

    print(f'###########\n\nRM_SEASON = {REMOVED_SEASON}\n\n###########\n')


