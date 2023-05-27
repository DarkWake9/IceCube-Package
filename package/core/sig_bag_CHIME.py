#######################################################################################################################
# This file contains the functions to calculate the signal and background PDFs                                        #  
# for the stacked analysis as given in EQN 6, 7, 9 of 2205.15963                                                      #
#                                                                                                                     #
# The functions are:                                                                                                  #  
#                                                                                                                     #
# S_ij: Calculates S_ij as in EQN 7 of 2205.15963                                                                     #
#                                                                                                                     #
# S_i: Calculates S_i as in EQN 6 of 2205.15963                                                                       #
#                                                                                                                     #
# B_i: Calculates B_i as in EQN 9 of 2205.15963                                                                       #
#                                                                                                                     #
# A signal class is defined to drive these functions and store the PDFs                                               #
#                                                                                                                     #
#######################################################################################################################






from numba import jit, njit, prange, set_num_threads, vectorize
from numba.experimental import jitclass
import numpy as np
from scipy.optimize import minimize
# from core import weights
from core import weights as weights
from core.req_arrays_CHIME import *
from core.stacking_analysis import wall_nu
import multiprocessing as mul

set_num_threads(int(mul.cpu_count()))

@njit
def hvovec(lon1=0.0, lat1=0.0, lon2=0.0, lat2=0.0, rad=False):

    '''
    Parameters
    ----------
    lon1 : float
        Longitude of first point.
    lat1 : float
        Latitude of first point.


    lon2 : float
        Longitude of second point.
    lat2 : float
        Latitude of second point.

    rad : Boolean, optional
        (default) False -> Angle is returned in Degrees
        True -> Angle is returned in radians
    

    Returns
    -------
    float
        Returns the haversine angle between two vectors given their latitude and longitude
    

    Warning
    -------
        This function assumes the input to be in degrees and of equal length\n
        or one of the input pairs to be a single value and the other pair to be an array
    '''

    #Convert decimal degrees to Radians:
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    
    a = np.add(np.multiply(np.sin(lat1), np.sin(lat2)), np.multiply(np.multiply(np.cos(lat1), np.cos(lat2)), np.cos(dlon)))

    if rad == True:
        return np.arccos(a)         #In Radians
    else:
        return np.abs(np.arccos(a)/deg2rad_var)



#Compute the signal PDF for all neutrinos as per eqns 6, 7 and weights as per eqn 8 of 2205.15963

@njit
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


@njit
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

    # si_sing_season_g =
    # for i in prange(p):
        # sij = S_ijk(nu)
        # np.sum(np.multiply(sij, normalized_wt[i][gamma_index][season]))      #1/rad**2



    sij = S_ijk(nu)
    season = 0
    for i in range(10):
        if season_walls[i] <= nu and nu < season_walls[i+1]:
            season = i
            break

    return np.sum(np.multiply(sij, np.multiply(w_models[ws], weight[gamma_index][season])/np.sum(np.multiply(w_models[ws], weight[gamma_index][season]))))      #1/rad**2




N_ic = 1134450
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
    for i in prange(len(icdec)):
        if abs(icdec[i] - icdec[nu]) <= cone:
            count+=1
    binwidth = (np.sin(np.deg2rad(icdec[nu] + cone)) - np.sin(np.deg2rad(icdec[nu] - cone)))*2*np.pi
    return count/(binwidth * N_ic)  


# @jitclass
class signals:


    def __init__(self, gamma):
        self.gamma = gamma
        weight_obj =  weights.weights(self.gamma)
        self.all_weights = weight_obj.all_weights
        self.sum_weights = weight_obj.sum_weights


    def compute_signal(self):
        all_weights = self.all_weights
        sum_weights = self.sum_weights
        @jit(nopython=True, cache=True)
        def sigbag_nu(nu):
            '''
                Returns the signal and background PDF for the {nu}th neutrino
            '''
            wall = wall_nu(nu)  #Finding the wall/icecube season in which the nu^th neutrino lies
            return S_ik(nu,  all_weights, sum_weights, wall)#,Ns]

        pool = mul.Pool(int(mul.cpu_count()*0.9), maxtasksperchild=200)
        op_async = pool.map_async(sigbag_nu, range(lnu))
        tmp = op_async.get()
        pool.close()
        pool = []
        op_async = []
        tmp = np.asfarray(tmp)
        self.all_sig = tmp
        return self.all_sig

    def compute_background(self):
        pool = mul.Pool(int(mul.cpu_count()*0.9), maxtasksperchild=200)
        
        op_async = pool.map_async(Bi_stacked_compute, prange(lnu))
        tmp = op_async.get()
        pool.close()
        pool = []
        op_async = []
        tmp = np.asfarray(tmp)
        self.all_bag = tmp
        return self.all_bag