import yaml
import json
import sys
import pandas as pd
import uncertainties.unumpy as unp

stations = [0, 1, 2, 3]
layers = [0, 1, 2]
modules = [0, 1, 2, 3, 4, 5, 6, 7]

alignment_config = {}

df = pd.read_pickle("results/study/3446/results_module/result_table.pkl")

c = {}
for station in stations:
    for layer in layers:
        for module in modules:
            c[f"{station}{layer}{module}"] = [
                -(unp.nominal_values(df.transpose()[f"L{layer}M{module}"][0]) + 0) / 1000,  # mm
                -(unp.nominal_values(df.transpose()[f"L{layer}M{module}"][1]) + 0) / 1000,  # mm
                -(unp.nominal_values(df.transpose()[f"L{layer}M{module}"][2]) + 0) / 1000,  # mm
                -(unp.nominal_values(df.transpose()[f"L{layer}M{module}"][3]) + 0) / 1000,  # rad
                -(unp.nominal_values(df.transpose()[f"L{layer}M{module}"][4]) + 0) / 1000,  # rad
                -(unp.nominal_values(df.transpose()[f"L{layer}M{module}"][5]) + 0) / 1000,  # rad
            ]

with open(f'{sys.argv[1]}.txt','w') as outfile:
    outfile.write(str(c))
