import uproot3 as uproot
import pandas as pd
import numpy as np
import os
import sys

from alignment.monitoring import plot_chi2, plot_hit_position_per_module, plot_residual_per_module, plot_residual_per_layer
from alignment.equations import build_alignment_chi2_equation_module
from alignment.constants import runs


def prepare_data(df_, columns, chi2_max=10):
    
    data = pd.concat([
        df["spfit"].pandas.df(["initial_track_chi2"] + columns, entrystop=None, flatten=True) 
        for df in df_
    ], axis=0)

    # Required because of the way we define our coordinate systems
    data["x0_local"] += 0
    data["x1_local"] += -5
    data["x2_local"] += 5
    data["y0_local"] += 0
    data["y1_local"] += 0
    data["y2_local"] += 0

    # Sanitize data
    data[["module0", "module1", "module2"]] = data[["module0", "module1", "module2"]].astype(int)
    data = data[data.initial_track_chi2 < chi2_max]

    return data


if __name__ == "__main__":

    MONITORING = True
    CHI2_MAX=10
    
    for run in runs:

        # When going into local coordinates, global x and y are flipped. Hence we use mx --> my and my --> mx, 
        columns_module = [
            "x0_local", "x1_local", "x2_local",
            "y0_local", "y1_local", "y2_local",
            "z0", "z1", "z2",
            "initial_track_my",  # xy flipped in local coordinates
            "initial_track_mx", 
            "initial_track_residual_x0_local", "initial_track_residual_x1_local", "initial_track_residual_x2_local",
            "initial_track_residual_y0_local", "initial_track_residual_y1_local", "initial_track_residual_y2_local", 
            "module0", "module1", "module2",
            "sigma_x0_local", "sigma_x1_local", "sigma_x2_local",
            "sigma_y0_local", "sigma_y1_local", "sigma_y2_local",
            "rho0_local", "rho1_local", "rho2_local",
        ]


        print(f"Processing run {run}")
        print(runs[run])
        output_dir = os.path.join("results", sys.argv[1], str(run))
        os.makedirs(output_dir, exist_ok=True)
        data = prepare_data([uproot.open(f"{input_data}/trackerspfit.root") for input_data in runs[run]], columns=columns_module, chi2_max=CHI2_MAX)

        #for x0 in ["x0_local", "x1_local", "x2_local"]:
        #    data = data.query(f"-20 < {x0} < 20")
        
        #for y0 in ["y0_local", "y1_local", "y2_local"]:
        #    data = data.query(f"-60 < {y0} < 60")

        if MONITORING:
            plot_chi2(data, CHI2_MAX, output_dir)
            plot_residual_per_module(data, 0, "x", output_dir)
            plot_residual_per_module(data, 1, "x", output_dir)
            plot_residual_per_module(data, 2, "x", output_dir)
            plot_residual_per_module(data, 0, "y", output_dir)
            plot_residual_per_module(data, 1, "y", output_dir)
            plot_residual_per_module(data, 2, "y", output_dir)
            plot_hit_position_per_module(data, 0, "x", output_dir=output_dir, local=True)
            plot_hit_position_per_module(data, 1, "x", output_dir=output_dir, local=True)
            plot_hit_position_per_module(data, 2, "x", output_dir=output_dir, local=True)
            plot_hit_position_per_module(data, 0, "y", output_dir=output_dir, local=True)
            plot_hit_position_per_module(data, 1, "y", output_dir=output_dir, local=True)
            plot_hit_position_per_module(data, 2, "y", output_dir=output_dir, local=True)
            # plot_hit_position_per_module(data, 0, "x", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 1, "x", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 2, "x", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 0, "y", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 1, "y", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 2, "y", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 0, "z", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 1, "z", output_dir=output_dir, local=False)
            # plot_hit_position_per_module(data, 2, "z", output_dir=output_dir, local=False)

        print(f"run {run}")
        #print(f"cx {data.initial_track_cx.mean()}")
        #print(f"mx {data.initial_track_mx.mean()}")
        #print(f"cy {data.initial_track_cy.mean()}")
        #print(f"my {data.initial_track_my.mean()}")

        t, M = build_alignment_chi2_equation_module(data[columns_module])
        np.save(os.path.join(output_dir, "t_module.npy"), t)
        np.save(os.path.join(output_dir, "M_module.npy"), M)

