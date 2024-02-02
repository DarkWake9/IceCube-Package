# Procedure for smearing
# ---------------------------------
# There are two things you have to account for : muon-> neutrino scattering angle and error in reconstruction of muon direction.
# 
# The first  quantity is given in the variables PSF_min[deg]  PSF_max[deg] and the second in AngErr_min[deg]  AngErr_max[deg] in the files
# 
# 1. For each neutrino energy and declination choose the average of PSF_min and PSF_max as well as average of AngErr_min[deg]  AngErr_max[deg]  and add them in quadrature. 
#         This will give you the total angular uncertainty. ($\theta_{tot}$)
# 
# - For each simulated neutrino, take average of PSF_min and PSF_max as well as average of AngErr_min[deg]  AngErr_max[deg] and calculate $\sqrt{PSF_{avg}^2 + AngErr_{avg}^2}$
# 
# 2. convert the simulated RA and DEC to theta  ($\theta_{mock}$) and phi  ($\phi_{mock}$) for the simulated MJD 
# 
# - Choose simulated MJD  to be uniformly distributed for each phase
# 
# 3. assume error in theta=error in phi =($\theta_{tot}$)/$\sqrt(2)$
# 
# then choose Gausian distributed random numbers with **mean=0** and standard deviation = error in theta as well as error in phi.
# 
# This will give you the error in theta  ($\Delta \theta_{mock}$) and error in phi  ($\Delta \phi_{mock}$) for the simulated neutrino event
# 
# New theta=$\theta_{mock}$+$\Delta \theta_{mock}$
# 
# and New phi=$\phi_{mock}$+$\Delta \phi_{mock}$
# 
# 4. Calculate new RA and DEC from new theta and new phi . This will be the smeared RA and DEC of the muon
 


from astropy.units import deg, meter
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time

import numpy as np
import os
import multiprocessing as mul
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads, vectorize, guvectorize, cuda
from tqdm import tqdm
import pandas as pd
import argparse as ap

os.system("cp -r ./o_data/icecube_10year_ps/events/ ./data/icecube_10year_ps/")

from core.signal_bag import *
from core.stacking_analysis import *
from core.req_arrays import *
import time
print("Original IceCube Data copied to data/icecube_10year_ps/events/")
print(icwidths)

###########################################################################################
parser = ap.ArgumentParser()
parser.add_argument('-nb', '--numbins', type=int, help='Number of bins', required=False)
parser.add_argument('-np', '--numpulsars', type=int, help='Number of pulsars for neutrinos to be generated at', required=False)
parser.add_argument('-lp', '--logphiomul', type=float, help='log10(Phi_0 multiplication factor)', required=False)
arrg = parser.parse_args()


nbins=100
if arrg.numbins:
    nbins = arrg.numbins
else:
    nbins = 100

n_psrs = 30
if arrg.numpulsars:
    n_psrs = arrg.numpulsars
    
    
phio_const = 4.98 * (10**(-27)) #GeV-1 to ev-1 converted

if arrg.logphiomul:
    phio_const *= 10**(arrg.logphiomul)
elif arrg.logphiomul == 'a':
    phio_const *= 1
else:
    pass
###########################################################################################

latitude = -89.99
longitude = 13.2536#-63.453056
location = EarthLocation(lat=latitude * deg, lon=longitude * deg, height = 2835*meter)

def ra_dec_to_azimuth_zenith(ra_deg, dec_deg, obs_mjd):
    
    # Convert MJD to Time object
    obs_time = Time(obs_mjd, format='mjd')

    # Create a SkyCoord object with the given RA and Dec
    coord = SkyCoord(ra=ra_deg * deg, dec=dec_deg* deg, frame='icrs')

    # Convert to Altitude and Azimuth coordinates
    altaz = coord.transform_to(AltAz(location=location ,obstime=obs_time))

    # Calculate zenith angle (90° - altitude)
    # zenith_angle_deg = 90.0 - altaz.alt.deg
    alt_angle_deg = altaz.alt.deg

    # Extract Azimuth and Zenith Angle values (in degrees)
    azimuth_deg = altaz.az.deg
    
    return (azimuth_deg, alt_angle_deg)


def altaz_to_radec(azimuth, alt, mjd):
    
    # Convert zenith angle to altitude
    # altitude = (90 - zenith) * deg
    altitude = alt * deg
    
    # Convert MJD to Time object
    observing_time = Time(mjd, format='mjd', scale='utc')
    
    # Create EarthLocation
    
    # Create AltAz coordinate
    altaz = AltAz(location=location, obstime=observing_time)
    
    # Create SkyCoord from AltAz
    altaz_coord = SkyCoord(alt=altitude, az=azimuth, frame=altaz, unit=(deg, deg))
    
    # Convert AltAz coordinates to RA and Dec
    equatorial_coord = altaz_coord.transform_to('icrs')
    
    
    ra_deg = equatorial_coord.ra.deg
    dec_deg = equatorial_coord.dec.deg
    
    return (ra_deg, dec_deg)





season = 0
icsmear_k = pd.read_csv('./data/icecube_10year_ps/irfs/IC40_smearing.csv', sep='\s+', comment='#', names='log10(E_nu/GeV)_min	log10(E_nu/GeV)_max	Dec_nu_min[deg]	Dec_nu_max[deg]	log10(E/GeV)_min	log10(E/GeV)_max	PSF_min[deg]	PSF_max[deg]	AngErr_min[deg]	AngErr_max[deg]	Fractional_Counts'.split('\t'), dtype=float)
icsmear_k_log_E = np.array(list(set(icsmear_k['log10(E_nu/GeV)_min'])))#.union(set(icsmear_k['log10(E_nu/GeV)_max']))))
log_E_width = 26400
icsmear_k_log_E.sort()

icsmear_k_dec = np.ravel(list(set(icsmear_k['Dec_nu_min[deg]'])))
icsmear_k_dec.sort()




enus = np.logspace(11.001, 18.999, int(nbins))
enus_bin_indices = np.digitize(enus, e_nu_wall) - 1
msdec_bin_indices = np.digitize(msdec, dec_nu) - 1


gamma_arr = [-2, -2.2, -2.53, -3]


syn_nu_choice = np.random.randint(0, p, n_psrs) #Choose 50 random pulsars from the 3389 pulsars
syn_nudec_bin = msdec_bin_indices[syn_nu_choice] #Find the declination bin of the chosen pulsars to be allocated for the synthetic neutrinos
syn_nu_ra = msra[syn_nu_choice] #Find the right ascension of the chosen pulsars to be allocated for the synthetic neutrinos
syn_nu_dec = msdec[syn_nu_choice] #Find the declination of the chosen pulsars to be allocated for the synthetic neutrinos



filenames = ["IC40_exp.csv", "IC59_exp.csv","IC79_exp.csv", "IC86_I_exp.csv", "IC86_II_exp.csv",
        "IC86_III_exp.csv", "IC86_IV_exp.csv", "IC86_V_exp.csv", "IC86_VI_exp.csv", "IC86_VII_exp.csv"]


print("Phio constant mult: ", arrg.logphiomul)
print("Phio constant: ", phio_const)


def get_syn_nu_dec(season_i):


    icdata_k = pd.read_csv("./o_data/icecube_10year_ps/events/" + filenames[season_i], sep="\s+", comment="#", names="MJD[days]	log10(E/GeV)	AngErr[deg]	RA[deg]	Dec[deg]	Azimuth[deg]	Zenith[deg]".split("\t"), dtype=float)


    uptdata_k = pd.read_csv("./o_data/icecube_10year_ps/uptime/" + filenames[season_i], sep="\s+", comment="#", names=["MJD_start[days]","MJD_stop[days]"], dtype=float, index_col=False)
    
    max_stop = max(uptdata_k["MJD_stop[days]"].values)
    min_start = min(uptdata_k["MJD_start[days]"].values)
    # uptdata_k = []
    icsmear_k = []
    try:
        icsmear_k = pd.read_csv("./o_data/icecube_10year_ps/irfs/" + filenames[season_i].replace('exp', 'smearing'), sep="\s+", comment="#", names='log10(E_nu/GeV)_min	log10(E_nu/GeV)_max	Dec_nu_min[deg]	Dec_nu_max[deg]	log10(E/GeV)_min	log10(E/GeV)_max	PSF_min[deg]	PSF_max[deg]	AngErr_min[deg]	AngErr_max[deg]	Fractional_Counts'.split('\t'), dtype=float)
    except:
        icsmear_k = pd.read_csv("./o_data/icecube_10year_ps/irfs/" + filenames[4].replace('exp', 'smearing'), sep="\s+", comment="#", names='log10(E_nu/GeV)_min	log10(E_nu/GeV)_max	Dec_nu_min[deg]	Dec_nu_max[deg]	log10(E/GeV)_min	log10(E/GeV)_max	PSF_min[deg]	PSF_max[deg]	AngErr_min[deg]	AngErr_max[deg]	Fractional_Counts'.split('\t'), dtype=float)
        # print(temp1)
    enus_smear_bin_min = icsmear_k_log_E[ np.digitize(np.log10(enus/1e9), icsmear_k_log_E) - 1]
    nu_smear_dec_bin_min = icsmear_k_dec[ np.digitize(syn_nu_dec, icsmear_k_dec) - 1]
    n_nu = 0
    for i in tqdm(range(len(enus))):    
        
        temp1 = icsmear_k[icsmear_k["log10(E_nu/GeV)_min"] == enus_smear_bin_min[i]]
        
        for j in range(len(syn_nu_dec)):
        
            temp2 = temp1[temp1["Dec_nu_min[deg]"] == nu_smear_dec_bin_min[j]]        
            psf_avg = (min(temp2["PSF_min[deg]"]) + max(temp2["PSF_max[deg]"])) / 2
            
       
            angerr_avg = (min(temp2["AngErr_min[deg]"]) + max(temp2["AngErr_max[deg]"])) / 2
            theta_tot = np.sqrt(psf_avg**2 + angerr_avg**2)
        
            # No.of neutrinos to be generated in this season, in this energy bin, in this declination bin
            n_nu_temp = t_upt[season_i] * earea[ea_season(season_i)][syn_nudec_bin[j] * 40 + enus_bin_indices[i]] * enus[i] * phio_const * ((enus[i] / 10**14) ** gamma_arr[2])
            
            
            n_nu_temp = int(n_nu_temp)
            # print(n_nu_temp)
            n_nu += n_nu_temp
            if n_nu_temp > 0:
                mjd = np.random.uniform(min_start, max_stop, n_nu_temp)
                temp_ra = np.ones(n_nu_temp, dtype=np.float64) * syn_nu_ra[j]
                temp_dec = np.ones(n_nu_temp, dtype=np.float64) * syn_nu_dec[j]
                
                
                
                syn_nu_az, syn_nu_alt = ra_dec_to_azimuth_zenith(temp_ra, temp_dec, mjd)
                
                errs = np.random.normal(0, theta_tot/(2**0.5), n_nu_temp)
                
                syn_nu_az += errs
                syn_nu_alt += errs
                
                for xx in prange(len(syn_nu_alt)):
                    if syn_nu_alt[xx] > 90:
                        # 91 degrees altitude should correspond to 89 degrees altitude
                        # so we take the negative of the angle and add 180 degrees
                        syn_nu_alt[xx] = 180 - syn_nu_alt[xx]
                    
                    elif syn_nu_alt[xx] < -90:
                        # -91 degrees altitude should correspond to -89 degrees altitude
                        # so we take the negative of the angle and add -180 degrees
                        syn_nu_alt[xx] = -180 - syn_nu_alt[xx]
                    else:
                        pass
                    
                
                
                syn_ra, syn_dec = [], []
                
                
                for temp_i in range(len(syn_nu_az)):
                    tra, tdec = altaz_to_radec(syn_nu_az[temp_i], syn_nu_alt[temp_i], mjd[temp_i])
                    syn_ra.append(tra)
                    syn_dec.append(tdec)
                    
                syn_ra = np.array(syn_ra)
                syn_dec = np.array(syn_dec)
                
                
                temp3 = pd.DataFrame(np.vstack([mjd, -1 * np.ones(len(mjd)), errs, syn_ra, syn_dec, syn_nu_az, syn_nu_alt]).T)
                temp3.columns = "MJD[days]	log10(E/GeV)	AngErr[deg]	RA[deg]	Dec[deg]	Azimuth[deg]	Zenith[deg]".split('\t')
                
                icdata_k = pd.concat([icdata_k, temp3], ignore_index=True, axis=0)
                
                
            else:
                continue
            
    icdata_k.to_csv("./data/icecube_10year_ps/events/" + filenames[season_i], sep="\t", index=False)
    print("Done with season", season_i)
    
    print(season_i, len(icdata_k.index))
    print("No.of neutrinos generated:")
    print(n_nu)
    time.sleep(5)
    return 0
    
pool = mul.Pool(int(mul.cpu_count()))
op_async = pool.map_async(get_syn_nu_dec, tqdm(range(10)))
drive = op_async.get()
pool.close()
pool.join()
op_async = []
# print("Generated Synthetic neutrinos and added to original IceCube Data")
    


