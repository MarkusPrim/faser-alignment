import os
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
from copy import copy

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


def What(z, sigma_x, sigma_y):
    _A = A(z)
    _W = np.diag([1/sigma_x**2, 1/sigma_x**2, 1/sigma_x**2, 1/sigma_y**2, 1/sigma_y**2, 1/sigma_y**2])
    return (
        np.identity(_A.shape[0]) - _A @ np.linalg.inv(
            _A.transpose() @ _W @ _A
        ) @ _A.transpose() @ _W
    ).transpose() @ _W


def dr_da_layer(x, y, mx, my):
    spacer = 6
    
    #drda_x = lambda x, y: np.array([0, 0, 0, 0, 0, 0]) * RESCALING
    #drda_y = lambda x, y: np.array([0, 0, 0, 0, 0, 0]) * RESCALING
    #drda_x = lambda x, y: np.array([-1, 0, 0, 0, 0, y]) * RESCALING
    #drda_y = lambda x, y: np.array([0, -1, 0, 0, 0, -x]) * RESCALING
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


def dr_da_module(x, y, mx, my):
    spacer = 6
    #drda_x = lambda x, y: np.array([0, 0, 0, 0, 0, 0]) * RESCALING
    #drda_y = lambda x, y: np.array([0, 0, 0, 0, 0, 0]) * RESCALING
    #drda_x = lambda x, y: np.array([-1, 0, 0, 0, 0, y]) * RESCALING
    #drda_y = lambda x, y: np.array([0, -1, 0, 0, 0, -x]) * RESCALING
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


def build_alignment_chi2_equation(df):
    dim = 18 + 144
    t =  np.zeros(dim)
    M = np.zeros((dim, dim))
    # Initalize as np array and sum while iterating
    for _, row in df.iterrows():
        x = row.values
        
        module0 = x[14]
        module1 = x[15]
        module2 = x[16]
        # Layer Components
        _t = dr_da_layer(x[20:23], x[23:26], x[7], x[6]).transpose() @ What(x[17:20], sigma_x, sigma_y) @ x[8:14]
        _M = dr_da_layer(x[20:23], x[23:26], x[7], x[6]).transpose() @ What(x[17:20], sigma_x, sigma_y) @ dr_da_layer(x[20:23], x[23:26], x[7], x[6])

        i0 = int(layer_index(0))
        i1 = int(layer_index(1))
        i2 = int(layer_index(2))

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

        # Module Components
        _t = dr_da_module(x[:3], x[3:6], x[6], x[7]).transpose() @ What(x[17:20], sigma_x, sigma_y) @ x[8:14]
        _M = dr_da_module(x[:3], x[3:6], x[6], x[7]).transpose() @ What(x[17:20], sigma_x, sigma_y) @ dr_da_module(x[:3], x[3:6], x[6], x[7])
        
        i0 = int(module_index(0, module0))
        i1 = int(module_index(1, module1))
        i2 = int(module_index(2, module2))
        
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
    
    return t, M


def apply_soft_mode_cut_to_chi2_equation(t, M, soft_mode_cut=None):
    # Soft Mode Cut
    # Initial constraint on the alignment parameters
    if soft_mode_cut is not None:
        t += soft_mode_cut[0]
        M += np.diag(2 / soft_mode_cut[1] ** 2)
    return t, M


def fix_parameters_in_chi2_equation(t, M, fix):
    if fix is not None:
        for i in fix:
            t[i] = 0
            M[i, :] = 0
            M[:, i] = 0
    return t, M


def solve_alignment_chi2_equation(t, M, output_dir, cutoff, plot=True):
    w, v = np.linalg.eig(M)
    w = np.real(w)
    v = np.real(v)
    
    if plot:
        plt.figure(dpi=120, figsize=(10, 4))
        plt.title("Spectral Decomposition")
        for i in range(0, 18+144, 6):            
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


def save_and_store_results(a, Minv, output_dir):
    print(len(a))
    a *= np.array((3+3*8)*[*RESCALING])
    Minv *= np.array((3+3*8)*[*RESCALING]) ** 2
    MinvDiagonal = copy(Minv.diagonal())

    result_table = pd.DataFrame({
        r"L0": unp.uarray(a[layer_index(0):layer_index(0)+6],  MinvDiagonal[layer_index(0):layer_index(0)+6] ** 0.5) * 1000,
        r"L1": unp.uarray(a[layer_index(1):layer_index(1)+6],  MinvDiagonal[layer_index(1):layer_index(1)+6] ** 0.5) * 1000,
        r"L2": unp.uarray(a[layer_index(2):layer_index(2)+6],  MinvDiagonal[layer_index(2):layer_index(2)+6] ** 0.5) * 1000,

        r"L0M0": unp.uarray(a[module_index(0, 0):module_index(0, 0)+6],  MinvDiagonal[module_index(0, 0):module_index(0, 0)+6] ** 0.5) * 1000,
        r"L1M0": unp.uarray(a[module_index(1, 0):module_index(1, 0)+6],  MinvDiagonal[module_index(1, 0):module_index(1, 0)+6] ** 0.5) * 1000,
        r"L2M0": unp.uarray(a[module_index(2, 0):module_index(2, 0)+6],  MinvDiagonal[module_index(2, 0):module_index(2, 0)+6] ** 0.5) * 1000,

        r"L0M1": unp.uarray(a[module_index(0, 1):module_index(0, 1)+6],  MinvDiagonal[module_index(0, 1):module_index(0, 1)+6] ** 0.5) * 1000,
        r"L1M1": unp.uarray(a[module_index(1, 1):module_index(1, 1)+6],  MinvDiagonal[module_index(1, 1):module_index(1, 1)+6] ** 0.5) * 1000,
        r"L2M1": unp.uarray(a[module_index(2, 1):module_index(2, 1)+6],  MinvDiagonal[module_index(2, 1):module_index(2, 1)+6] ** 0.5) * 1000,

        r"L0M2": unp.uarray(a[module_index(0, 2):module_index(0, 2)+6],  MinvDiagonal[module_index(0, 2):module_index(0, 2)+6] ** 0.5) * 1000,
        r"L1M2": unp.uarray(a[module_index(1, 2):module_index(1, 2)+6],  MinvDiagonal[module_index(1, 2):module_index(1, 2)+6] ** 0.5) * 1000,
        r"L2M2": unp.uarray(a[module_index(2, 2):module_index(2, 2)+6],  MinvDiagonal[module_index(2, 2):module_index(2, 2)+6] ** 0.5) * 1000,

        r"L0M3": unp.uarray(a[module_index(0, 3):module_index(0, 3)+6],  MinvDiagonal[module_index(0, 3):module_index(0, 3)+6] ** 0.5) * 1000,
        r"L1M3": unp.uarray(a[module_index(1, 3):module_index(1, 3)+6],  MinvDiagonal[module_index(1, 3):module_index(1, 3)+6] ** 0.5) * 1000,
        r"L2M3": unp.uarray(a[module_index(2, 3):module_index(2, 3)+6],  MinvDiagonal[module_index(2, 3):module_index(2, 3)+6] ** 0.5) * 1000,
        
        r"L0M4": unp.uarray(a[module_index(0, 4):module_index(0, 4)+6],  MinvDiagonal[module_index(0, 4):module_index(0, 4)+6] ** 0.5) * 1000,
        r"L1M4": unp.uarray(a[module_index(1, 4):module_index(1, 4)+6],  MinvDiagonal[module_index(1, 4):module_index(1, 4)+6] ** 0.5) * 1000,
        r"L2M4": unp.uarray(a[module_index(2, 4):module_index(2, 4)+6],  MinvDiagonal[module_index(2, 4):module_index(2, 4)+6] ** 0.5) * 1000,

        r"L0M5": unp.uarray(a[module_index(0, 5):module_index(0, 5)+6],  MinvDiagonal[module_index(0, 5):module_index(0, 5)+6] ** 0.5) * 1000,
        r"L1M5": unp.uarray(a[module_index(1, 5):module_index(1, 5)+6],  MinvDiagonal[module_index(1, 5):module_index(1, 5)+6] ** 0.5) * 1000,
        r"L2M5": unp.uarray(a[module_index(2, 5):module_index(2, 5)+6],  MinvDiagonal[module_index(2, 5):module_index(2, 5)+6] ** 0.5) * 1000,

        r"L0M6": unp.uarray(a[module_index(0, 6):module_index(0, 6)+6],  MinvDiagonal[module_index(0, 6):module_index(0, 6)+6] ** 0.5) * 1000,
        r"L1M6": unp.uarray(a[module_index(1, 6):module_index(1, 6)+6],  MinvDiagonal[module_index(1, 6):module_index(1, 6)+6] ** 0.5) * 1000,
        r"L2M6": unp.uarray(a[module_index(2, 6):module_index(2, 6)+6],  MinvDiagonal[module_index(2, 6):module_index(2, 6)+6] ** 0.5) * 1000,

        r"L0M7": unp.uarray(a[module_index(0, 7):module_index(0, 7)+6],  MinvDiagonal[module_index(0, 7):module_index(0, 7)+6] ** 0.5) * 1000,
        r"L1M7": unp.uarray(a[module_index(1, 7):module_index(1, 7)+6],  MinvDiagonal[module_index(1, 7):module_index(1, 7)+6] ** 0.5) * 1000,
        r"L2M7": unp.uarray(a[module_index(2, 7):module_index(2, 7)+6],  MinvDiagonal[module_index(2, 7):module_index(2, 7)+6] ** 0.5) * 1000,
        
    }, index=result_cols).transpose()

    result_table.to_pickle(os.path.join(output_dir, "result_table.pkl"))
    
    for col in result_table.columns:
        result_table[col] = result_table[col].apply(lambda x: f"{x:.2f}")

    result_table.to_latex(os.path.join(output_dir, "result_table.tex"), escape=False)

    return result_table