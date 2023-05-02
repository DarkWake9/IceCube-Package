from numba import jit, njit, prange
import numpy as np
from scipy.optimize import minimize

vec_uptparts = np.array([0,   36900,  143911,  237044,  373288, 1134450])

@jit(nopython=True)
def wall_nu(nu):
    
        '''
        Parameters
        ----------
        nu : int
            Index of the neutrino from IceCube sample
        
    
        Returns
        -------
        int
            Returns the index of the wall in which the {nu}th neutrino lies
        '''
        wall = 0
        for i in range(len(vec_uptparts)-1):
            if vec_uptparts[i] <= nu and vec_uptparts[i+1] > nu:
                wall = i
                break        
        return wall


@jit(nopython=True)
def Pr(x, Ns, S, B):

    '''
    Probability as in EQN 1 of 2205.15963
    ---------------


    Parameters
    ----------
    x : int
        Assumed no.of associated events

    Ns : int
        No.of neutrinos used for analysis

    S : float
        Signal PDF
    B : float
        Background PDF
    
    Returns
    -------
    float 
        Returns the probability of the selected set of neutrinos being associated\n
        with a given pulsar with {Ns} neutrinos, {S} signal and {B} background PDF and {x} assumed associated events
    '''

    nsN = x/Ns
    return np.add(np.multiply(nsN , S), np.multiply(np.subtract(1, nsN), B))



@jit(nopython=True)
def TS_st(x, S, B, Ns):

    ''' 
    Returns
    ----------
        Returns the Test Stastic value at
        $n_s$ = {x} for its parameters S, B, Ns
    '''
    # if x >=0:
    return np.sum(np.asfarray(2*np.log(Pr(x,  Ns, S, B)/B)))
    # else:
        # return -1

lnu = 1134450
Ns = lnu#np.count_nonzero(nuind+1)

@njit(parallel=True)
def Ts_arr2(x, S_all, B_all, Ns):
    '''
    Parameters
    ----------
    x : float
        The value of x for which the TS is to be calculated

    S_all : array
        The array of signal PDFs for all the neutrinos

    B_all : array
        The array of background PDFs for all the neutrinos
    
    Ns : int
        The number of neutrinos

    Returns
    -------
    float
        The TS value for the entire stack of neutrinos for the given value of x
    
    '''


    #Ts_arr = lambda x:
    sum = 0.0
    for i in prange(lnu):
        sum += TS_st(x, S_all[i], B_all[i], Ns)
    return sum


@njit(fastmath=True)
def dfde(e_nu, gamma, phi0 = 1e-40):
    '''
    Parameters
    ----------
    e_nu : float
        The neutrino energy in eV
    
    gamma : float
        The spectral index of the neutrino flux
    
    phi0 : float (optional)
        The normalization constant of the neutrino flux in eV^-1. The default is 1e-40.
    
    Returns
    -------
    float
        The differential flux of neutrinos
    '''

    return phi0 * ((e_nu/(10**14))**gamma)      #eV^-1


@njit(fastmath=True)
def upt_season(nu):
    for i in range(len(vec_uptparts)):
        if vec_uptparts[i] > nu:
            return ea_season(i-1)
    return -1

        
@njit(fastmath=True)
def ea_season(season):
    '''
    Parameters
    ----------
    season : int
        The season number

    Returns
    -------
    int
        The season number as per effective area file indices

    '''


    if season <= 3 and season >= 0:
        return season
    elif season >= 4 and season <= 10:
        return 4
    else:
        return -1



