import uproot3 as uproot
import pandas as pd
import numpy as np
import os
import sys

from alignment.monitoring import plot_chi2, plot_hit_position_per_module, plot_residual_per_module
from alignment.equations import build_alignment_chi2_equation_module, build_alignment_chi2_equation_layer
from alignment.constants import runs


def prepare_data(df_, chi2_max=10):

    data = pd.concat([
        df["spfit"].pandas.df([
            "initial_track_chi2",
            "initial_track_cx",
            "initial_track_mx",
            "initial_track_cy",
            "initial_track_my",
            "initial_track_residual_x0",
            "initial_track_residual_x1",
            "initial_track_residual_x2",
            "initial_track_residual_y0",
            "initial_track_residual_y1",
            "initial_track_residual_y2",
            "x0",
            "x1",
            "x2",
            "y0",
            "y1",
            "y2",
            "z0",
            "z1",
            "z2",
            "x0_local",
            "x1_local",
            "x2_local",
            "y0_local",
            "y1_local",
            "y2_local",
            "module0",
            "module1",
            "module2",
    ], entrystop=None, flatten=True) 
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

    for run in runs:
        print(f"Processing run {run}")
        output_dir = os.path.join("results", sys.argv[1], str(run))
        os.makedirs(output_dir, exist_ok=True)
        data = prepare_data([uproot.open(f"{input_data}/trackerspfit.root") for input_data in runs[run][:3]])


        if MONITORING:
            plot_chi2(data, 10, output_dir)
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
            plot_hit_position_per_module(data, 0, "x", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 1, "x", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 2, "x", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 0, "y", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 1, "y", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 2, "y", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 0, "z", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 1, "z", output_dir=output_dir, local=False)
            plot_hit_position_per_module(data, 2, "z", output_dir=output_dir, local=False)


        columns_layer = [
            "x0", "x1", "x2",
            "y0", "y1", "y2",
            "z0", "z1", "z2",
            "initial_track_my",
            "initial_track_mx", 
            "initial_track_residual_x0", "initial_track_residual_x1", "initial_track_residual_x2", 
            "initial_track_residual_y0", "initial_track_residual_y1", "initial_track_residual_y2",
        ]

        t, M = build_alignment_chi2_equation_layer(data[columns_layer])
        np.save(os.path.join(output_dir, "t_layer.npy"), t)
        np.save(os.path.join(output_dir, "M_layer.npy"), M)

        columns_module = [
            "y0_local", "y1_local", "y2_local",  # Local xy flipped
            "x0_local", "x1_local", "x2_local",
            "z0", "z1", "z2",
            "initial_track_my",
            "initial_track_mx", 
            "initial_track_residual_x0", "initial_track_residual_x1", "initial_track_residual_x2", 
            "initial_track_residual_y0", "initial_track_residual_y1", "initial_track_residual_y2",
            "module0", "module1", "module2",
        ]

        t, M = build_alignment_chi2_equation_module(data[columns_module])
        np.save(os.path.join(output_dir, "t_module.npy"), t)
        np.save(os.path.join(output_dir, "M_module.npy"), M)

