import yaml
import json
import sys
import scipy.stats

stations = [0, 1, 2, 3]
layers = [0, 1, 2]
modules = [0, 1, 2, 3, 4, 5, 6, 7]

def get_random_alignment(a=None):
    """Gets a reasonable random number for the requested parameter."""
    if a is None:
        return 0
    
    if a in ["x", "y", "z"]:
        return float(scipy.stats.norm.rvs(0, 1e-1))  # Assuming misalignment is approx. 100 mum
    
    if a in ["alpha", "beta", "gamma"]:
        return float(scipy.stats.norm.rvs(0, 2e-3))  # Assuming misalignment is approx. 2 mrad
    
    raise NotImplemented("""
    Either choose None (default), or any of the alignment parameters as config, 
    available are x, y, z, alpha, beta, gamma.
    """)



# If you want to introduce misalignment, replace None with PARAMETER, and a random parameter will be written in the configuration file.
c = {}
for station in stations:
    c[f"{station}"] = [
        get_random_alignment(None),
        get_random_alignment(None),
        get_random_alignment(None),
        get_random_alignment(None),
        get_random_alignment(None),
        get_random_alignment(None),
    ]
    for layer in layers:
        c[f"{station}{layer}"] = [
            get_random_alignment(None),
            get_random_alignment(None),
            get_random_alignment(None),
            get_random_alignment(None),
            get_random_alignment(None),
            get_random_alignment(None),
        ]
        for module in modules:
            c[f"{station}{layer}{module}"] = [
                get_random_alignment(None),
                get_random_alignment(None),
                get_random_alignment(None),
                get_random_alignment(None),
                get_random_alignment(None),
                get_random_alignment(None),
            ]

with open(f'{sys.argv[1]}.txt','w') as outfile:
    outfile.write(str(c))

