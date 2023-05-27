###########################################################################################################################
###########################################################################################################################
#                                                                                                                         #
# This file contains all the required arrays and variables for the task 4 of Project IceCube                              #
#                                                                                                                         #
#Data:                                                                                                                    #
#                                                                                                                         #
# icdata: The IceCube data                                                                                                #
#                                                                                                                         #
# uptdata: The UPT data                                                                                                   #
#                                                                                                                         #
# eadata: The Effective Area data                                                                                         #
#                                                                                                                         #
# mspdata: The MSP data                                                                                                   #
#                                                                                                                         #
#Vectors:                                                                                                                 #
#                                                                                                                         #
# msdec: The declination of the FRBs in the CHIME catalogue                                                               #
#                                                                                                                         #
# msra: The right ascension of the FRBs in the CHIME   catalogue                                                          #
#                                                                                                                         #                                                                                                                        #
# icdec: The declination of the neutrinos in the IceCube data                                                             #
#                                                                                                                         #
# icra: The right ascension of the neutrinos in the IceCube data                                                          #
#                                                                                                                         #
# icang: The angular error of the neutrinos in the IceCube data                                                           #
#                                                                                                                         #
# iceng: The energy of the neutrinos in the IceCube data (in GeV)                                                         #
#                                                                                                                         #
# icwidths: The widths of the seasons in the IceCube data                                                                 #
#                                                                                                                         #
# ictimes: The times of the seasons in the IceCube data                                                                   #
#                                                                                                                         #
# icparts: The partitions of the IceCube data for each season                                                             #
#                                                                                                                         #
# upt_icparts: The partitions of the UPT data for each season                                                             #
#                                                                                                                         #
# t_upt: The detector uptime of the seasons from the uptime data                                                          #
#                                                                                                                         #
# log_e: The log10(E/GeV) values range as in all 'effectiveArea' files                                                    #
#                                                                                                                         #
# dec_nu: Set of Declination walls in all 'effectiveArea' files                                                           #
#                                                                                                                         #
# e_nu: The mid point of E_nu_min and E_nu_max as in all 'effectiveArea' files                                            #
#                                                                                                                         #
# de_nu: The difference of E_nu_min and E_nu_max as in all 'effectiveArea' files                                          #
#                                                                                                                         #
# earea: The effective area of the seasons in the effectiveArea data (in m^2)                                             #
#                                                                                                                         #
###########################################################################################################################

# The arrays are stored in the form of numpy vectors

# The arrays and variables are used in the following files:

# ../core/req_arrays_CHIME.py

# ../core/signal_bag.py

# ../core/stacking_analysis.py

# ../core/weights.py

# ../task4k_packed.ipynb

###########################################################################################################################
###########################################################################################################################

# from core import readfiles                     #Comment this line if you are using the CHIME data
from core import readfiles_CHIME as readfiles   #Comment this line if you are using the ATNF data
import numpy as np
import os

all_data = readfiles.Data(os.getcwd() + '/data/')
icdata = all_data.icdata
uptdata = all_data.uptdata
eadata = all_data.eadata
mspdata = all_data.mspdata
all_data = []

#ICECUBE DATA VECTORS
icwidths = [int(i) for i in "0 36900 107011 93133 136244 112858 122541 127045 129311 123657 145750".split(' ')]
# ictimes = [float(i) for i in icdata['MJD[days]']]
icparts = [np.sum(icwidths[:i]) for i in range(1,len(icwidths)+1)]  #paritions of icdata for each season (IC40, IC59, IC79, IC86_I, IC86_II)

upt_icparts = icparts[:5]
upt_icparts.append(icparts[-1])
upstop_ttt = np.asfarray([uptdata[i]['MJD_stop[days]'].values[-1] for i in range(len(uptdata))], dtype=np.float64)
upstart_ttt = np.asfarray([uptdata[i]['MJD_start[days]'].values[0] for i in range(len(uptdata))], dtype=np.float64)
vec_uptparts = np.asarray(upt_icparts, dtype=np.int64)
upt_icparts = np.asarray(upt_icparts)
t_upt = np.asarray([upstop_ttt[season] - upstart_ttt[season] for season in range(len(upstart_ttt))])*86400 #Convert days to seconds
#t_upt in seconds

log_e = np.round(np.arange(2, 10.2, 0.2), 2) #log10(E/GeV) values range as in all 'effectiveArea' files
earea = np.asfarray([eadata[i]['A_Eff[cm^2]'].values for i in range(len(eadata))])     #cm2
#dec_nu = Set of Declination walls in all 'effectiveArea' files
dec_nu = list(set(eadata[0]['Dec_nu_min[deg]'].values).union(set(eadata[0]['Dec_nu_max[deg]'].values)))

dec_nu.sort()
dec_nu = np.array(dec_nu)

e_nu_wall = np.asarray((10**log_e) * 1e9)

e_nu = ((10**(log_e[:-1])+ 10**(log_e[1:]))/2)*1e9          #E_nu in eV
de_nu = 1e9*(10**log_e[1:] - 10**log_e[:-1])                #dE_nu in eV

#ATNF PULSAR VECTORS

# msra = np.array([float(i) for i in mspdata['RAJD'].values], dtype=np.float64)         #RAJD in degrees
# msdec = np.asfarray([float(i) for i in mspdata['DECJD'].values], dtype=np.float64)       #DECJD in degrees
# msdist = np.array( [float(i) for i in mspdata['DIST_DM']], dtype=np.float64)
# mss1400 = np.array( [float(i) for i in mspdata['S1400']], dtype=np.float64)
msra = np.array([float(i) for i in mspdata['ra'].values], dtype=np.float64)         #RA CHIME in degrees
msdec = np.array([float(i) for i in mspdata['dec'].values], dtype=np.float64)       #DEC CHIME in degrees

icra = np.array([float(i) for i in icdata['RA[deg]']], dtype=np.float64)              #RA in degrees
icdec = np.array([float(i) for i in icdata['Dec[deg]']], dtype=np.float64)            #Dec in degrees
icang = np.array([float(i) for i in icdata['AngErr[deg]']], dtype=np.float64)         #AngErr in degrees
# iceng = np.array([float(i) for i in icdata['log10(E/GeV)']])
global p, lg, lnu
p = len(msra)
lg = 1 + len(icra) // p
lnu = len(icra)

deg2rad_var = np.pi/180
icdata = []
uptdata = []
eadata = []
mspdata = []


# All units check out:   eV, cm2, s, deg, GeV