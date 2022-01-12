import numpy as np

RESCALING = np.array([1e-1, 1e-1, 1e-1, 2e-3, 2e-3, 2e-3])  # Scale everything to same order of magnitude

# TODO Write this out on data
sigma_x = 0.8160  # Tobias (confirmed) number # 0.570 # TDR Number
sigma_y = 0.0163  # Tobias (confirmed) number # 0.020 # TDR Number


result_cols = [
    r"Layer 0 - $x$", r"Layer 0 - $y$", r"Layer 0 - $z$", r"Layer 0 - $\alpha$", r"Layer 0 - $\beta$", r"Layer 0 - $\gamma$",
    r"Layer 1 - $x$", r"Layer 1 - $y$", r"Layer 1 - $z$", r"Layer 1 - $\alpha$", r"Layer 1 - $\beta$", r"Layer 1 - $\gamma$",
    r"Layer 2 - $x$", r"Layer 2 - $y$", r"Layer 2 - $z$", r"Layer 2 - $\alpha$", r"Layer 2 - $\beta$", r"Layer 2 - $\gamma$",
]