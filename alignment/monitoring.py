import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import os
from copy import copy


def plot_matrix(M, output_dir):
    X = copy(M)
    plt.figure(dpi=240)
    X[X == 0] = np.nan
    plt.imshow(X, interpolation = 'none')

    b = 18
    for i in range(1, 8):
        plt.axvline(i * b - 0.5, ls="-", lw=0.5, color="black")
        plt.axhline(i * b - 0.5, ls="-", lw=0.5, color="black")

    b=6
    for i in range(1, 3*8):
        plt.axvline(i * b - 0.5, ls=":", lw=0.3, color="black")
        plt.axhline(i * b - 0.5, ls=":", lw=0.3, color="black")

    ax = plt.gca()

    plt.xticks(np.array(range(0, 144, 18)) - 0.5, [f"$M_{i}$" for i in range(8)])
    # plt.yticks(np.array(range(8, 144, 18)) + 0.5, [f"M{i}" for i in range(8)])
    plt.yticks(np.array(range(0, 144, 6)) - 0.5, [f"$L_{i}$" for i in 8*list(range(3))])

    plt.colorbar()
    plt.title("Alignment Solution Matrix\n" + "$a = x, y, z, \\alpha, \\beta, \\gamma$ for each Module/Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"solution_matrix.pdf"))
    plt.savefig(os.path.join(output_dir, f"solution_matrix.png"))
    plt.show()
    plt.close()


def plot_residual_per_module(df, layer, axis, output_dir):
    fig, axs = plt.subplots(4, 2, dpi=120, sharex=True, sharey=True, gridspec_kw={"wspace": 0.025, "hspace": 0.05})
    xmin = df[f"initial_track_residual_{axis}{layer}"].min() * 1000
    xmax = df[f"initial_track_residual_{axis}{layer}"].max() * 1000
    boundary = max(abs(xmin), abs(xmax))
    xlim = (-boundary, boundary)
    if axis == "x": xlim = (-2000, 2000)
    if axis == "y": xlim = (-100, 100)
    bins = 14
    # bins = int(np.ceil((xmax - xmin) / (sigma_x * 1000))) * 5
    #if axis == "x": bins = int((xmax - xmin) / (sigma_x * 1000))
    #if axis == "y": bins = int((xmax - xmin) / (sigma_y * 1000))
    #print(xmax, xmin, xmax-xmin, bins)
    
    tmp = np.linspace(*xlim)
    for module in range(0, 8):
        i = module % 4
        j = module // 4
        series = df.query(f"{xlim[0]} < initial_track_residual_{axis}{layer} < {xlim[1]}").query(f"module{layer} == {module}")[f"initial_track_residual_{axis}{layer}"] * 1000
        _, edges, _ = axs[i, j].hist(series, bins=bins, range=xlim, density=True)
        axs[i, j].plot(tmp, scipy.stats.norm.pdf(tmp, loc=series.mean(), scale=series.std()), 
                       label=r"$\mu$" + f" = {series.mean():.3f}" + r" $\mu$m " + r"$\sigma$" + f" = {series.std():.3f}" + r" $\mu$m")
        axs[i, j].legend(loc=2, fontsize=8, frameon=False)
        axs[i, j].set_xlabel(axis + r" Residual [$\mu$m]")
        if j == 0: axs[i, j].set_ylabel("arb. units")
        axs[i, j].text(0.1, 0.5, f"M{module}", fontsize=8, transform = axs[i, j].transAxes)
    axs[i, j].set_ylim(0, axs[i, j].get_ylim()[1]*1.2)
    fig.suptitle(f"Layer {layer}")
    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"residual_per_module_L{layer}_{axis}.pdf"))
    plt.savefig(os.path.join(output_dir, f"residual_per_module_L{layer}_{axis}.png"))
    plt.show()
    plt.close()

    
def plot_chi2(df, cut, output_dir, title=None):
    fig = plt.figure(dpi=120)
    xlim = (0, 2*cut)
    
    tmp = np.linspace(*xlim)

    series = df[f"initial_track_chi2"] 
    _, edges, _ = plt.hist(series, bins=50, range=xlim, density=True)
    plt.plot(tmp, scipy.stats.chi2.pdf(tmp, df=2), label="$\chi^2(k=2)$")
    plt.legend(loc=0, fontsize=8, frameon=False)
    plt.xlabel(r"$\chi^2$")
    plt.axvline(cut, color="black", lw=1, ls=":")
    plt.ylabel("arb. units")
    plt.ylim(0, plt.gca().get_ylim()[1]*1.2)
    plt.text(0.1, 0.8, f"$N$ = {int(len(series))}", fontsize=8, transform = plt.gca().transAxes)
    if title is None:
        fig.suptitle(f"All Tracks")
    else:
        fig.suptitle(title)
    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"chi2.pdf"))
    plt.savefig(os.path.join(output_dir, f"chi2.png"))
    plt.show()
    plt.close()


def plot_hit_position_per_module(df, layer, axis, output_dir, local=False):
    fig, axs = plt.subplots(4, 2, dpi=120, sharex=True, sharey=True, gridspec_kw={"wspace": 0.025, "hspace": 0.05})
    if local: 
        var = f"{axis}{layer}_local"
    else: 
        var = f"{axis}{layer}"
    xmin = df[var].min()
    xmax = df[var].max()
    xlim = (xmin, xmax)
    
    tmp = np.linspace(*xlim)
    for module in range(0, 8):
        i = module % 4
        j = module // 4
        series = df.query(f"module{layer} == {module}")[var] 
        _, edges, _ = axs[i, j].hist(series, bins=200, range=xlim, density=True)
        #axs[i, j].plot(tmp, scipy.stats.norm.pdf(tmp, loc=series.mean(), scale=series.std()), 
        #               label=r"$\mu$" + f" = {series.mean() * 1000:.3f}" + r" $\mu$m " + r"$\sigma$" + f" = {series.std() * 1000:.3f}" + r" $\mu$m")
        #axs[i, j].legend(loc=2, fontsize=8, frameon=False)
        axs[i, j].text(0.1, 0.8, f"M{module} $\\bar{{x}}$ = {series.mean():.2f}, $\\tilde{{x}}$ = {series.median():.2f}", fontsize=8, transform = axs[i, j].transAxes)
        if local: 
            axs[i, j].set_xlabel(axis + " Local Position [mm]")
        else:
            axs[i, j].set_xlabel(axis + " Global Position [mm]")
        if j == 0: axs[i, j].set_ylabel("arb. units")
    axs[i, j].set_ylim(0, axs[i, j].get_ylim()[1]*1.2)
    fig.suptitle(f"Layer {layer}")
    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hit_position_per_module_L{layer}_{axis}.pdf"))
    plt.savefig(os.path.join(output_dir, f"hit_position_per_module_L{layer}_{axis}.png"))
    plt.show()
    plt.close()