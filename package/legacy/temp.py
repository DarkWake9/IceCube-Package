def wt_acc_temp(gamma):
    wt_acc = []
    # for gamma in prange(4):
    wt_allpsr = []
    for season in range(10):
        tmp = []
        for psrno in range(p):
            # w_model = w_models[psrno]
            tmp.append(np.trapz(psr_wt_sing_e_gamma(psrno, enus, gamma_arr[gamma], season), enus))
            # tmp.append(trapz_numba(psr_wt_sing_e_gamma(psrno, enus, gamma_arr[gamma], season), enus))

        wt_allpsr.append(np.array(tmp, dtype=np.float64))
        tmp = []
    # wt_acc.append(wt_allpsr)
    # wt_allpsr = []
    return np.array(wt_allpsr, dtype=np.float64)
        # print(f"Calculated wt_acc for gamma = {gamma_arr[gamma]}")