import os
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
from copy import copy

from alignment.monitoring import plot_matrix
from alignment.constants import RESCALING, sigma_x, sigma_y, result_cols

def module_index(layer, module):
    """Calculates the offset-index of a specific module in a layer in the alignment matrix.

    Args:
        layer (int): Layer Index
        module (int): Module Index

    Returns:
        int: Offset-index inside alignment matrix.
    """
    return 18 + layer*6 + module*6*3


def layer_index(layer):
    """Calculates the offset-index of a specific layer in the alignment matrix.

    Args:
        layer (int): Layer Index
        module (int): Module Index

    Returns:
        int: Offset-index inside alignment matrix.
    """
    return layer*6


def index(layer, module):
    """Legacy Code Compatibility"""
    return layer*6 + module*6*3


def f(z, cx, mx, cy, my):
    return np.array([mx*z + cx, my*z + cy, z])


def A(z):
    return np.block([
        [np.vstack([np.ones(len(z)), z]).T, np.zeros((3, 2))],
        [np.zeros((3, 2)), np.vstack([np.ones(len(z)), z]).T],
    ])


def b(x, y):
    return np.block([
        x, y
    ])


def What(z):
    _A = A(z)
    _W = np.diag([1/sigma_x**2, 1/sigma_x**2, 1/sigma_x**2, 1/sigma_y**2, 1/sigma_y**2, 1/sigma_y**2])
    return (
        np.identity(_A.shape[0]) - _A @ np.linalg.inv(
            _A.transpose() @ _W @ _A
        ) @ _A.transpose() @ _W
    ).transpose() @ _W

_What = What(np.array([50, 100, 150]))


def dr_da(x, y, mx, my):
    spacer = 6
    
    drda_x = lambda x, y: np.array([-1, 0, mx, 0, 0, y])
    drda_y = lambda x, y: np.array([0, -1, my, 0, 0, -x])
    drda_x = lambda x, y: np.array([-1, 0, mx, mx*y, -mx*x, y]) * RESCALING
    drda_y = lambda x, y: np.array([0, -1, my, my*y, -my*x, -x]) * RESCALING
    
    return np.array([
        # --- x ---
        [*drda_x(x[0], y[0]), *np.zeros(spacer), *np.zeros(spacer)],  # L0
        [*np.zeros(spacer), *drda_x(x[1], y[1]), *np.zeros(spacer)],  # L1
        [*np.zeros(spacer), *np.zeros(spacer), *drda_x(x[2], y[2])],  # L2
        # --- y ---
        [*drda_y(x[0], y[0]), *np.zeros(spacer), *np.zeros(spacer)],  # L0
        [*np.zeros(spacer), *drda_y(x[1], y[1]), *np.zeros(spacer)],  # L1
        [*np.zeros(spacer), *np.zeros(spacer), *drda_y(x[2], y[2])],  # L2
    ])


def build_alignment_chi2_equation(df, soft_mode_cut=None):
    t = np.zeros(144)
    M = np.zeros((144, 144))
    # Initalize as np array and sum while iterating
    for _, row in df.iterrows():
        x = row.values
        module0 = x[14]
        module1 = x[15]
        module2 = x[16]
        
        _t = dr_da(x[:3], x[3:6], x[6], x[7]).transpose() @ What(x[17:20]) @ x[8:14]
        _M = dr_da(x[:3], x[3:6], x[6], x[7]).transpose() @ What(x[17:20]) @ dr_da(x[:3], x[3:6], x[6], x[7])
        
        i0 = int(index(0, module0))
        i1 = int(index(1, module1))
        i2 = int(index(2, module2))
        
        t[i0:][:6] += _t[ 0:][:6]
        t[i1:][:6] += _t[ 6:][:6]
        t[i2:][:6] += _t[12:][:6]
        
        M[i0:, i0:][:6, :6] += _M[ 0:,  0:][:6, :6]
        M[i0:, i1:][:6, :6] += _M[ 0:,  6:][:6, :6]
        M[i0:, i2:][:6, :6] += _M[ 0:, 12:][:6, :6]
        
        M[i1:, i0:][:6, :6] += _M[ 6:,  0:][:6, :6]
        M[i1:, i1:][:6, :6] += _M[ 6:,  6:][:6, :6]
        M[i1:, i2:][:6, :6] += _M[ 6:, 12:][:6, :6]
        
        M[i2:, i0:][:6, :6] += _M[12:,  0:][:6, :6]
        M[i2:, i1:][:6, :6] += _M[12:,  6:][:6, :6]
        M[i2:, i2:][:6, :6] += _M[12:, 12:][:6, :6]
    
    # Soft Mode Cut
    if soft_mode_cut is not None:
        # Initial constraint on the alignment parameters
        t += soft_mode_cut[0]
        M += np.diag(2 / soft_mode_cut[1] ** 2)
            
    return t, M



def solve_alignment_chi2_equation(t, M, output_dir, cutoff=1, plot=True):
    w, v = np.linalg.eig(M)
    w = np.real(w)
    v = np.real(v)
    
    if plot:
        plt.figure(dpi=120, figsize=(10, 4))
        plt.title("Spectral Decomposition")
        for i in range(0, 144, 6):            
            if i % 18:
                plt.axvline(i-0.5, color="black", lw=0.5, ls="-")
            else:
                plt.axvline(i-0.5, color="black", lw=0.5, ls="--")
        plt.plot(range(len(w)), w, marker="o", ls="", label="By Index")
        plt.plot(range(len(w)), sorted(w), marker="x", ls="", label="By Magnitude")
        plt.axhline(cutoff, ls=":", color="black", label="CutOff")
        plt.yscale("log")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue Magnitude")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"spectral_decomposition.pdf"))
        plt.savefig(os.path.join(output_dir, f"spectral_decomposition.png"))
        plt.close()
    
    w = np.where(w > cutoff, w, 0)
    winv = (w ** -1)
    winv[winv == np.inf] = 0

    Minv = v @ np.diag(winv) @ v.transpose()
    a = -Minv @ t
    return a, Minv



def station_wise(data, cutoff, output_dir, fix=None, plot=False, soft_mode_cut=None):
    t, M = build_alignment_chi2_equation(data, soft_mode_cut=soft_mode_cut)
    if fix is not None:
        for i in fix:
            t[i] = 0
            M[i, :] = 0
            M[:, i] = 0
    if plot:
        plot_matrix(M, output_dir)
    a, Minv = solve_alignment_chi2_equation(t, M, output_dir, cutoff=cutoff, plot=plot)
    # print("Global Chi2: ", (a - METEOROLOGY_alpha0).transpose() @ Minv @ (a - METEOROLOGY_alpha0))
    
    a *= np.array(3*8*[*RESCALING])
    Minv *= np.array(3*8*[*RESCALING]) ** 2
    MinvDiagonal = copy(Minv.diagonal())
    #MinvDiagonal *= np.array(3*8*[*RESCALING]) ** 2

    result_table = pd.DataFrame({
        r"M0: ($\mu$m or mrad)": unp.uarray(a[index(0, 0):index(0, 1)],  MinvDiagonal[index(0, 0):index(0, 1)] ** 0.5) * 1000,
        r"M1: ($\mu$m or mrad)": unp.uarray(a[index(0, 1):index(0, 2)],  MinvDiagonal[index(0, 1):index(0, 2)] ** 0.5) * 1000,
        r"M2: ($\mu$m or mrad)": unp.uarray(a[index(0, 2):index(0, 3)],  MinvDiagonal[index(0, 2):index(0, 3)] ** 0.5) * 1000,
        r"M3: ($\mu$m or mrad)": unp.uarray(a[index(0, 3):index(0, 4)],  MinvDiagonal[index(0, 3):index(0, 4)] ** 0.5) * 1000,
        r"M4: ($\mu$m or mrad)": unp.uarray(a[index(0, 4):index(0, 5)],  MinvDiagonal[index(0, 4):index(0, 5)] ** 0.5) * 1000,
        r"M5: ($\mu$m or mrad)": unp.uarray(a[index(0, 5):index(0, 6)],  MinvDiagonal[index(0, 5):index(0, 6)] ** 0.5) * 1000,
        r"M6: ($\mu$m or mrad)": unp.uarray(a[index(0, 6):index(0, 7)],  MinvDiagonal[index(0, 6):index(0, 7)] ** 0.5) * 1000,
        r"M7: ($\mu$m or mrad)": unp.uarray(a[index(0, 7):index(0, 8)],  MinvDiagonal[index(0, 7):index(0, 8)] ** 0.5) * 1000,
    }, index=result_cols)
    result_table.to_pickle(os.path.join(output_dir, "result_table.pkl"))
    
    for col in result_table.columns:
        result_table[col] = result_table[col].apply(lambda x: f"{x:.2f}")

    result_table.to_latex(os.path.join(output_dir, "result_table.tex"), escape=False)

    return result_table.transpose()