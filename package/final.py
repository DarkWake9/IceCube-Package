####
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####
temp = []

phimuls = [-2.0, -1.0, -0.5, -0.25, 0, 0.5, int(1)]
for phi_mul in phimuls:
    f = open(f"./outputs/TS_wmod1_500_bins_30_psrs_{phi_mul}_phi_fac.txt", 'r')
    temp.append(f.read().split('\n'))

####
tsmax = []
small_f = []
phio = []
phio_const = [(10**i) * 4.98e-27 for i in phimuls]  #[4.98e-29, 4.98e-28, 10**-0.75 * 4.98e-27, 10**-0.5 * 4.98e-27, 4.98e-27, 10**0.5 * 4.98e-27, 4.98e-26]
phio_const_norm = [i /4.98e-27 for i in phio_const]
for i in temp:
    tsmax.append(float(i[0]))
    small_f.append(float(i[1]))
    phio.append(float(i[2]))

####
np.sqrt(tsmax)

####
phio_const

####
np.sqrt(tsmax[1])

####
phio_const = np.array(phio_const)/(1e-9)

####
plt.figure(figsize=(8, 6))
axesfont = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 17,
        }
font = {'family': 'serif',
        'size': 22,
        'weight': 'bold',
        'color':  'black',
        }


plt.plot(phio_const, np.sqrt(tsmax), lw=2.2, ls='-')
# plt.hlines(5, 1e-20, 1e-18, colors='k', linestyles='dashed', lw=2.2)
plt.scatter(phio_const, np.sqrt(tsmax))
# plt.plot(small_f, tsmax)
# plt.xlim(4.98e-27)
plt.ylabel('$\mathbf{\sqrt{TS_{max}}}$', fontdict=axesfont)
plt.xlabel(' $\mathsf{\phi_0}$ $(GeV^{-1} cm^{-2} s^{-1})$', fontdict=axesfont)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xscale('log')
plt.title('$\mathbf{\sqrt{TS_{max}}}$ vs $\mathsf{\phi_0}$', fontdict=font)
plt.savefig('./outputs/TSmax_phio_const.pdf', facecolor='w')
plt.show()