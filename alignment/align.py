import os
import sys
import numpy as np

from alignment.equations import save_and_store_results_module, solve_alignment_chi2_equation, apply_soft_mode_cut_to_chi2_equation, fix_parameters_in_chi2_equation
from alignment.constants import RESCALING, runs
from alignment.monitoring import plot_matrix


def align(output_dir, t, M, soft_mode_cut, cutoff, fix):

    t, M = apply_soft_mode_cut_to_chi2_equation(t, M, soft_mode_cut=soft_mode_cut)
    t, M = fix_parameters_in_chi2_equation(t, M, fix=fix)

    plot_matrix(M, output_dir)

    a, Minv = solve_alignment_chi2_equation(t, M, output_dir, cutoff=cutoff, plot=True)
    return a, Minv


if __name__ == "__main__":

    CUTOFF = 100
    METEOROLOGY_alpha0_module = np.zeros(144)
    METEOROLOGY_sigma0_module = np.array(8 * 3 * [1e-1, 1e-1, 1e-1, 2e-3, 2e-3, 2e-3]) / np.array(8 * 3 * [*RESCALING])
    SOFT_MODE_CUT_module = (METEOROLOGY_alpha0_module, METEOROLOGY_sigma0_module)

    
    FIX_gamma = np.array([i*6 + np.array([5]) for i in range(3*8+3)]).flatten()    
    FIX_z_alpha_beta = np.array([i*6 + np.array([2, 3, 4]) for i in range(3*8+3)]).flatten()
    #FIX = np.array([[2, 3, 4] + [2+6, 3+6, 4+6] + [2+12, 3+12, 4+12] + list(range(18, 144+18))])
    #FIX = set([*FIX_z_alpha_beta, *FIX_gamma *range(18, 18+144)])
    #FIX = set([*FIX_z_alpha_beta, *FIX_gamma *range(18)])
    FIX = set([*FIX_z_alpha_beta, *FIX_gamma])
    FIX=None
    
    print(f"Applied cutoff: {CUTOFF}")
    print(f"Applied fixes: {FIX}")

    all_t_module = []
    all_M_module = []

    for run in runs:
    
        data_dir = os.path.join("results", sys.argv[1], str(run))

        output_dir = os.path.join(data_dir, "results_module")
        os.makedirs(output_dir, exist_ok=True)
        t = np.load(os.path.join(data_dir, "t_module.npy"))
        M = np.load(os.path.join(data_dir, "M_module.npy"))
        all_t_module.append(t)
        all_M_module.append(M)
        save_and_store_results_module(*align(output_dir, t, M, soft_mode_cut=SOFT_MODE_CUT_module, cutoff=CUTOFF, fix=FIX), output_dir)


    data_dir = os.path.join("results", sys.argv[1], "all")

    output_dir = os.path.join(data_dir, "results_module")
    os.makedirs(output_dir, exist_ok=True)
    t = sum(all_t_module)
    M = sum(all_M_module)
    save_and_store_results_module(*align(output_dir, t, M, soft_mode_cut=SOFT_MODE_CUT_module, cutoff=CUTOFF, fix=FIX), output_dir)


    data_dir = os.path.join("results", sys.argv[1], "none")

    output_dir = os.path.join(data_dir, "results_module")
    os.makedirs(output_dir, exist_ok=True)
    dim = 144
    t = np.zeros(dim)
    M = np.zeros((dim, dim))
    save_and_store_results_module(*align(output_dir, t, M, soft_mode_cut=SOFT_MODE_CUT_module, cutoff=CUTOFF, fix=FIX), output_dir)
