############################################################################################################################

# This file contains the function to calculate the weights of each pulsar 
# for each neutrino season/sample for a given spectral index gamma
# The function is called psr_wt_quick
# 
# IT FOLLOWS EQN 8 OF 2205.15963
# 
# The class weights is used to store the weights of all pulsars for all neutrino seasons/samples
# For a given gamma an object of class weights is created containing 
# the weights of all pulsars for all neutrino samples and their sum

############################################################################################################################



from numba import njit, prange
import numpy as np
from scipy.optimize import minimize
from core import readfiles

from core.req_arrays import *



@njit
def psr_wt_quick(nusample_wall, psrno, gamma = 1, weight_scheme = 0):

    '''
    EQN 8 OF 2205.15963
    -------------------

    Parameters
    ----------
    nusample_wall : float
        The index of catalogue of the given neutrino ===> season
    psrno : int
        The index of the pulsar in the ATNF catalogue
    gamma : float, optional
        (default) 1
        The spectral index of the power law
    weight_scheme : int, optional
        (default) 0
        The scheme of calculating the w_model
        0: w_model = 1 : Uniform weight
        1: w_model = 1 / (1 + d^2) : Weight inversely proportional to the square of the distance_DM
        2: w_model = s_1400 : Weight proportional to the flux at 1400 MHz (mJy)

    Returns
    -------
    
    Returns the weight of psrno^th pulsar for a given neutrino sample {nusample_wall}
    '''
    
    
    psr_decl = icdec[psrno]
    t_upt = upstop_ttt[nusample_wall] - upstart_ttt[nusample_wall] 
    t_upt*=86400                                                        #Convert to seconds
    d_ind = 0

    for i in range(len(dec_nu) - 1):
        if dec_nu[i] <= psr_decl and dec_nu[i+1] >= psr_decl:           #just comparing so no need for deg2rad
            d_ind = i
            break
    ea_temp = earea[nusample_wall][d_ind*40:(d_ind+1)*40]               #in cm2
    


    # THIS LINE FOLLOWS EQN 8 OF 2205.15963
    weight_kj = t_upt * np.sum(np.multiply(np.multiply(ea_temp, np.power(e_nu, gamma)), de_nu))     #in s cm2 eV^(gamma + 1)
    #sleep(1e-8)
    return weight_kj   
                       
            
class weights:
    def __init__(self, gamma):
        self.gamma = gamma
        all_weights = []
        for i in prange(len(upt_icparts)-1):
            all_weights.append([psr_wt_quick(i, psrno, gamma) for psrno in range(p)])
        self.all_weights = np.asfarray(all_weights)                             #in s cm2 GeV^(gamma + 1)
        sum_weights = [np.sum(i) for i in self.all_weights]
        self.sum_weights = np.asfarray(sum_weights)                             #in s cm2 GeV^(gamma + 1)


